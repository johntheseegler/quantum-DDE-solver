"""Discretization and technical ingredients for solving the d-dimensional
periodic drift-diffusion equation (DDE).

Implements the preliminaries, forward-time centered-space (FTCS)
discretization, linear operator construction, and spectral/conditioning
analysis from:
    E. Devereux and A. Datta, "Quantum algorithms for solving a drift-diffusion
    equation: a complexity analysis", Phys. Rev. A 112, 042419 (2025)
    [arXiv:2505.21221v2].

Architecture
------------
This module defines ``DDEConfig`` and ``DDEDiscretization`` dataclasses plus
explicit layer functions (``layer_0_1`` .. ``layer_0_7``) designed so that
future solver modules (``classical_solvers.py``, ``quantum_measurement.py``,
and ``quantum_solvers.py``) consume a single ``DDEDiscretization`` object or
individual layer outputs cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq


@dataclass
class DDEConfig:
    """Configuration for the periodic d-dimensional drift-diffusion problem.

    PDE (Eq. 2):
        dp/dt = sum_{i=1}^d [ a * dp/dx_i + D * d^2 p / dx_i^2 ]
    on x in [-L, L]^d with periodic spatial boundary conditions and t in [0, T].
    """

    T: float
    L: float
    a: float
    D: float
    d: int = 1
    zeta: float = 1.0
    eps_c: float = 0.05
    eps_q: float = 0.05

    # Optional explicit grid overrides (when None, Corollary 2 Eqs. 20-21 apply).
    nx: Optional[int] = None
    nt: Optional[int] = None

    # Initial probability distribution p0(x) configuration (Assumption 1c & Eq. 23).
    p0_type: str = "gaussian"  # "gaussian", "point_origin", or "custom"
    x0: Optional[Sequence[float]] = None
    sigma0: float = 0.35
    custom_p0_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None


@dataclass
class DDEDiscretization:
    """Container holding outputs of all discretization & spectral layers.

    Every downstream classical solver (C1-C4) and quantum solver (Q1-Q4)
    receives this object so all grid metadata, initial states, operators,
    linear system blocks, and spectral bounds are shared consistently.
    """

    config: DDEConfig
    theory: Dict[str, Any]
    dx: float
    dt: float
    nx: int
    nt: int
    stability_param: float
    stochastic_stable: bool
    lemma11_Y: float
    a_over_D: float
    lemma11_satisfied: bool
    thm1_error_bound: float
    x_1d: np.ndarray
    t_grid: np.ndarray
    coords_dd: np.ndarray
    p0: np.ndarray
    ket_p0: np.ndarray
    p_exact_T: Optional[np.ndarray]
    H_1d: np.ndarray
    M: np.ndarray
    L_op: np.ndarray
    T_mat: np.ndarray
    A_mat: np.ndarray
    b_vec: np.ndarray
    spectra: Dict[str, Any] = field(default_factory=dict)


def _next_even_int(val: float) -> int:
    """Round up to the nearest positive even integer >= 2."""
    n = max(2, int(np.ceil(val)))
    return n if n % 2 == 0 else n + 1


def _unitary_dft_1d(nx: int) -> np.ndarray:
    """Construct the unitary inverse-DFT matrix F with F_{k,j} = exp(+2*pi*i*j*k/nx)/sqrt(nx).

    With this convention (matching Appendix B Eq. B3 and Eq. 34),
    F^dagger @ H @ F = diag(mu_0, ..., mu_{nx-1}) where F^dagger is the
    normalized forward DFT.
    """
    idx = np.arange(nx, dtype=float)
    j_grid, k_grid = np.meshgrid(idx, idx, indexing="xy")
    return np.exp(2j * np.pi * k_grid * j_grid / nx) / np.sqrt(nx)


def wrapped_gaussian_1d(
    x: np.ndarray,
    t: float,
    x0_val: float,
    sigma0: float,
    a: float,
    D: float,
    L: float,
    n_images: int = 4,
) -> np.ndarray:
    """Exact 1D periodic wrapped Gaussian solution to Eq. (2) on [-L, L).

    Note the PDE sign convention in Eq. (2):
        dp/dt = +a * dp/dx + D * d^2 p / dx^2
    corresponds to advection velocity v = -a, so the Gaussian center
    translates as mu(t) = x0 - a * t with variance sigma(t)^2 = sigma0^2 + 2*D*t
    and period 2*L.
    """
    period = 2.0 * L
    mu_t = x0_val - a * t
    sigma_t = np.sqrt(sigma0**2 + 2.0 * D * t)
    images = np.arange(-n_images, n_images + 1, dtype=float)
    diff = x[..., None] - mu_t + period * images
    return np.exp(-0.5 * (diff / sigma_t) ** 2).sum(axis=-1) / (
        sigma_t * np.sqrt(2.0 * np.pi)
    )


def wrapped_gaussian_dd(
    coords: np.ndarray,
    t: float,
    x0_vec: Sequence[float],
    sigma0: float,
    a: float,
    D: float,
    L: float,
    n_images: int = 4,
) -> np.ndarray:
    """Separable d-dimensional periodic wrapped Gaussian solution to Eq. (2)."""
    density = np.ones(coords.shape[0], dtype=float)
    for dim in range(coords.shape[1]):
        density *= wrapped_gaussian_1d(
            coords[:, dim], t, float(x0_vec[dim]), sigma0, a, D, L, n_images=n_images
        )
    return density


def layer_0_1_theoretical_parameters_and_finance_benchmark(
    config: DDEConfig,
) -> Dict[str, Any]:
    """Layer 0.1: Corollary 2 step sizes (Eqs. 19-21) and Finance Tables I-III.

    Computes the theoretical discretization bounds required to achieve infinity-norm
    discretization error <= eps_c, alongside the paper's Appendix A Eq. (A1)
    volume-scaled variant and the representative finance benchmark (Sec. I.C,
    Tables II & III).
    """
    T, L, a, D, d, zeta, eps_c = (
        config.T,
        config.L,
        config.a,
        config.D,
        config.d,
        config.zeta,
        config.eps_c,
    )
    poly = 3.0 * (a**2) * (L**2) + 8.0 * a * D * L + 4.0 * (D**2)

    # Corollary 2 (Eqs. 19-21)
    dx_max = np.sqrt((12.0 * eps_c * D) / (T * d * zeta * poly))
    dt_max = (6.0 * eps_c) / (T * (d**2) * zeta * poly)
    nt_theory = (T**2) * (d**2) * zeta * poly / (6.0 * eps_c)
    nx_theory = np.sqrt((T * d * zeta * (L**2) * poly) / (3.0 * eps_c * D))

    # Appendix A Eq. (A1) / Table II includes 1/L^d factor in error bound
    nt_theory_A1 = nt_theory / (L**d)
    nx_theory_A1_half = np.sqrt((T * d * zeta * (L**2) * poly) / (12.0 * eps_c * D * (L**d)))

    # Finance benchmark from Sec. I.C Table II:
    # T = 5000, L = 10, a = 0.2366, D = 0.2455, d = 3, zeta = 1
    fin = dict(T=5000.0, L=10.0, a=0.2366, D=0.2455, d=3, zeta=1.0)
    fin_poly = (
        3.0 * (fin["a"] ** 2) * (fin["L"] ** 2)
        + 8.0 * fin["a"] * fin["D"] * fin["L"]
        + 4.0 * (fin["D"] ** 2)
    )
    fin_aLD = fin["a"] * fin["L"] + fin["D"]
    fin_nt_prefactor_A1 = (
        (fin["T"] ** 2) * (fin["d"] ** 2) * fin["zeta"] * fin_poly / (6.0 * (fin["L"] ** fin["d"]))
    )
    fin_nx_prefactor_A1 = np.sqrt(
        fin["T"] * fin["d"] * fin["zeta"] * (fin["L"] ** 2) * fin_poly
        / (12.0 * fin["D"] * (fin["L"] ** fin["d"]))
    )

    # Table III asymptotic prefactors from Theorems 6, 7, 8, 10, 18, 19, 21, 22
    # (omitting logarithmic terms and setting eps_c = eps_q = 1)
    fT, fL, fd, fz, fD = fin["T"], fin["L"], fin["d"], fin["zeta"], fin["D"]
    finance_table_iii = {
        "nt_over_inv_eps_c": fin_nt_prefactor_A1,
        "nx_over_inv_sqrt_eps_c": fin_nx_prefactor_A1,
        "classical_linear_prefactor": (
            (fd ** (4.0 + fd / 2.0))
            * (fT ** (3.0 + fd / 2.0))
            * (fz ** ((3.0 + fd) / 2.0))
            * (fin_aLD ** (3.0 + fd))
            / (fD ** (fd / 2.0))
        ),
        "classical_time_prefactor": (
            (fd ** (fd / 2.0 + 3.0))
            * (fT ** (fd / 2.0 + 2.0))
            * (fz ** (fd / 2.0 + 1.0))
            * (fin_aLD ** (fd + 2.0))
            / (fD ** (fd / 2.0))
        ),
        "classical_rw_prefactor": (
            (fd ** (fd / 2.0 + 3.0))
            * (fT ** (fd / 2.0 + 2.0))
            * (fz ** (fd / 2.0 + 1.0))
            * (fin_aLD ** (fd + 2.0))
            / (fD ** (fd / 2.0))
        ),
        "classical_fft_prefactor": (
            (fd ** (fd / 2.0 + 1.0))
            * (fT ** (fd / 2.0))
            * (fz ** (fd / 2.0))
            * (fL ** fd)
            * (fin_aLD ** fd)
            / (fD ** (fd / 2.0))
        ),
        "quantum_linear_prefactor": (fd**5) * (fT**2) * fz * (fin_aLD**2),
        "quantum_time_prefactor": (
            (fd ** (fd / 2.0 + 3.0))
            * (fT ** (fd / 2.0 + 2.0))
            * (fz ** (fd / 4.0 + 1.0))
            * (fL**2)
            * (fin_aLD ** (fd / 2.0))
        ),
        "quantum_rw_prefactor": (
            (fd ** ((fd + 7.0) / 2.0))
            * (fT ** (fd / 2.0 + 1.0))
            * (fz ** (fd / 4.0 + 0.5))
            * (fin_aLD ** (fd / 2.0 + 1.0))
        ),
        "quantum_qft_prefactor": (
            (fd ** (fd / 2.0 + 2.0))
            * (fT ** (fd / 2.0))
            * (fz ** (fd / 4.0))
            * (fin_aLD ** (fd / 2.0))
        ),
        "aLD": fin_aLD,
        "D": fD,
    }

    return {
        "poly_factor": poly,
        "dx_max_cor2": dx_max,
        "dt_max_cor2": dt_max,
        "nt_theory_cor2": nt_theory,
        "nx_theory_cor2": nx_theory,
        "nt_theory_appA": nt_theory_A1,
        "nx_theory_appA": nx_theory_A1_half,
        "suggested_nx_even": _next_even_int(nx_theory),
        "suggested_nt": max(1, int(np.ceil(nt_theory))),
        "finance_benchmark": finance_table_iii,
    }


def layer_0_2_build_grid_and_verify_stability(
    config: DDEConfig, theory: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Layer 0.2: Construct spatial/temporal grid G and check stability & Theorem 1 bounds.

    Resolves nx (must be even) and nt, computes dx = 2L / nx and dt = T / nt,
    builds 1D and d-D coordinate arrays, and verifies:
      - Theorem 1 stochastic stability (Eq. A2): 2*d*D*dt/dx^2 <= 1
      - Explicit centered-stencil positivity: a*dx/(2*D) <= 1 (Sec. III.E)
      - Lemma 11 conditioning criteria: Y = D*dt/dx^2 <= 1/5 and a/D < 2*sqrt(10)
      - Theorem 1 theoretical error bound (Eq. 18)
    """
    if theory is None:
        theory = layer_0_1_theoretical_parameters_and_finance_benchmark(config)

    nx = config.nx if config.nx is not None else theory["suggested_nx_even"]
    if nx < 2 or nx % 2 != 0:
        raise ValueError(f"nx must be a positive even integer (Sec. II.A), got {nx}.")

    dx = 2.0 * config.L / nx

    if config.nt is not None:
        nt = int(config.nt)
    else:
        # Choose nt so both Corollary 2 and Theorem 1 stability hold
        min_nt_stability = int(np.ceil(2.0 * config.d * config.D * config.T / (dx**2)))
        nt = max(theory["suggested_nt"], min_nt_stability, 1)

    if nt < 1:
        raise ValueError(f"nt must be >= 1, got {nt}.")

    dt = config.T / nt

    # Periodic grid x in [-L, L): x_0 = -L, ..., x_{nx/2} = 0, ..., x_{nx-1} = L - dx
    x_1d = -config.L + np.arange(nx, dtype=float) * dx
    t_grid = np.linspace(0.0, config.T, nt + 1)

    # d-dimensional coordinates in row-major tensor order (x_1, ..., x_d)
    mesh = np.meshgrid(*([x_1d] * config.d), indexing="ij")
    coords_dd = np.column_stack([m.ravel() for m in mesh])

    stability_param = 2.0 * config.d * config.D * dt / (dx**2)
    peclet_mesh = config.a * dx / (2.0 * config.D)
    stochastic_stable = bool(stability_param <= 1.0 + 1e-12 and peclet_mesh <= 1.0 + 1e-12)

    lemma11_Y = config.D * dt / (dx**2)
    a_over_D = config.a / config.D
    lemma11_satisfied = bool(
        lemma11_Y <= 0.2 + 1e-12 and a_over_D < 2.0 * np.sqrt(10.0)
    )

    thm1_error_bound = (
        config.T
        * (config.d * config.zeta / 2.0)
        * (
            config.d * dt * ((config.a * config.L + config.D) ** 2)
            + (dx**2 / 3.0) * (config.a * config.L + config.D / 2.0)
        )
    )

    return {
        "nx": nx,
        "nt": nt,
        "dx": dx,
        "dt": dt,
        "x_1d": x_1d,
        "t_grid": t_grid,
        "coords_dd": coords_dd,
        "n_space": nx**config.d,
        "stability_param": stability_param,
        "peclet_mesh": peclet_mesh,
        "stochastic_stable": stochastic_stable,
        "lemma11_Y": lemma11_Y,
        "a_over_D": a_over_D,
        "lemma11_satisfied": lemma11_satisfied,
        "thm1_error_bound": thm1_error_bound,
    }


def layer_0_3_initial_and_analytic_distributions(
    config: DDEConfig, grid_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Layer 0.3: Build normalized initial distribution p0 (Eq. 23), |p0> (Eq. 6),
    and the analytic reference distribution at t = T.
    """
    coords_dd = grid_info["coords_dd"]
    nx = grid_info["nx"]
    d = config.d
    n_space = grid_info["n_space"]

    if config.x0 is None:
        x0_vec = [0.0] * d
    else:
        if len(config.x0) != d:
            raise ValueError(f"Expected x0 of length d={d}, got {len(config.x0)}.")
        x0_vec = [float(v) for v in config.x0]

    p_exact_T: Optional[np.ndarray] = None

    if config.p0_type == "gaussian":
        raw_p0 = wrapped_gaussian_dd(
            coords_dd, 0.0, x0_vec, config.sigma0, config.a, config.D, config.L
        )
        raw_pT = wrapped_gaussian_dd(
            coords_dd, config.T, x0_vec, config.sigma0, config.a, config.D, config.L
        )
        # Normalize discrete probability mass on grid G (Eq. 23: ||p||_1 = 1)
        p0 = raw_p0 / np.sum(raw_p0)
        p_exact_T = raw_pT / np.sum(raw_pT)
    elif config.p0_type == "point_origin":
        p0 = np.zeros(n_space, dtype=float)
        # Index nx//2 along every axis corresponds to x_j = 0
        origin_multi = tuple([nx // 2] * d)
        origin_flat = int(np.ravel_multi_index(origin_multi, tuple([nx] * d)))
        p0[origin_flat] = 1.0
        # Green's function at t = T from point source at origin
        raw_pT = wrapped_gaussian_dd(
            coords_dd, config.T, [0.0] * d, 1e-12, config.a, config.D, config.L
        )
        p_exact_T = raw_pT / np.sum(raw_pT)
    elif config.p0_type == "custom":
        if config.custom_p0_fn is None:
            raise ValueError("custom_p0_fn must be provided when p0_type='custom'.")
        raw_p0 = np.asarray(config.custom_p0_fn(coords_dd), dtype=float).ravel()
        if raw_p0.shape != (n_space,):
            raise ValueError(
                f"custom_p0_fn returned shape {raw_p0.shape}, expected ({n_space},)."
            )
        if np.any(raw_p0 < 0) or np.sum(raw_p0) <= 0:
            raise ValueError("Initial distribution p0 must be non-negative with positive sum.")
        p0 = raw_p0 / np.sum(raw_p0)
    else:
        raise ValueError(
            f"Unsupported p0_type='{config.p0_type}'. Choose 'gaussian', 'point_origin', or 'custom'."
        )

    ket_p0 = p0 / np.linalg.norm(p0, ord=2)

    return {
        "x0_vec": x0_vec,
        "p0": p0,
        "ket_p0": ket_p0,
        "p_exact_T": p_exact_T,
        "p0_l1_norm": float(np.sum(np.abs(p0))),
        "ket_p0_l2_norm": float(np.linalg.norm(ket_p0, ord=2)),
    }


def layer_0_4_build_spatial_operators_H_and_M(
    config: DDEConfig, grid_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Layer 0.4: Construct 1D circulant spatial matrix H (Eq. 26) and
    d-dimensional Kronecker sum M (Eq. 28).
    """
    nx = grid_info["nx"]
    dx = grid_info["dx"]
    d = config.d
    D = config.D
    a = config.a

    w_center = -2.0 * D / (dx**2)
    w_plus = D / (dx**2) + a / (2.0 * dx)   # multiplies p(x + dx) -> superdiagonal
    w_minus = D / (dx**2) - a / (2.0 * dx)  # multiplies p(x - dx) -> subdiagonal

    H_1d = np.zeros((nx, nx), dtype=float)
    idx = np.arange(nx)
    H_1d[idx, idx] = w_center
    H_1d[idx, (idx + 1) % nx] = w_plus
    H_1d[idx, (idx - 1) % nx] = w_minus

    # Build M = sum_{j=1}^d I_{nx}^{otimes (j-1)} otimes H otimes I_{nx}^{otimes (d-j)}
    n_space = nx**d
    M = np.zeros((n_space, n_space), dtype=float)
    I_1d = np.eye(nx, dtype=float)
    for j in range(d):
        term = np.array([[1.0]], dtype=float)
        for m in range(d):
            term = np.kron(term, H_1d if m == j else I_1d)
        M += term

    return {
        "w_center": w_center,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "H_1d": H_1d,
        "M": M,
    }


def layer_0_5_build_transition_operator_L(
    config: DDEConfig,
    grid_info: Dict[str, Any],
    spatial_ops: Dict[str, Any],
    init_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Layer 0.5: Build FTCS transition operator L = I + dt * M (Eqs. 16 & 29)
    and verify stochasticity and mass conservation.
    """
    n_space = grid_info["n_space"]
    dt = grid_info["dt"]
    M = spatial_ops["M"]

    L_op = np.eye(n_space, dtype=float) + dt * M

    min_entry = float(np.min(L_op))
    row_sums = L_op.sum(axis=1)
    col_sums = L_op.sum(axis=0)
    is_stochastic = bool(
        min_entry >= -1e-14
        and np.allclose(row_sums, 1.0, atol=1e-12)
        and np.allclose(col_sums, 1.0, atol=1e-12)
    )

    p1_l1_norm = None
    if init_info is not None:
        p1 = L_op @ init_info["p0"]
        p1_l1_norm = float(np.sum(np.abs(p1)))

    return {
        "L_op": L_op,
        "min_entry": min_entry,
        "max_row_sum_err": float(np.max(np.abs(row_sums - 1.0))),
        "max_col_sum_err": float(np.max(np.abs(col_sums - 1.0))),
        "is_doubly_stochastic": is_stochastic,
        "p1_l1_norm": p1_l1_norm,
    }


def layer_0_6_build_spacetime_system_A(
    grid_info: Dict[str, Any],
    spatial_ops: Dict[str, Any],
    L_info: Dict[str, Any],
    init_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Layer 0.6: Construct time-shift matrix T (Eq. 31), full space-time system
    matrix A (Eqs. 17, 27, 30), and RHS vector b = (L*p0, 0, ..., 0)^T.
    """
    nt = grid_info["nt"]
    dt = grid_info["dt"]
    n_space = grid_info["n_space"]
    M = spatial_ops["M"]
    L_op = L_info["L_op"]
    p0 = init_info["p0"]

    # Eq. (31): nt x nt lower-bidiagonal shift matrix T
    T_mat = np.eye(nt, dtype=float) - np.eye(nt, k=-1, dtype=float)

    # Eq. (30): A = T otimes L - dt * I_{nt} otimes M
    I_nt = np.eye(nt, dtype=float)
    A_mat = np.kron(T_mat, L_op) - dt * np.kron(I_nt, M)

    # Direct block construction from Eqs. (17) and (27) for equivalence verification
    I_space = np.eye(n_space, dtype=float)
    A_block_ref = np.kron(I_nt, I_space) - np.kron(np.eye(nt, k=-1, dtype=float), L_op)
    tensor_block_diff = float(np.max(np.abs(A_mat - A_block_ref)))

    # Eq. (17): b = [L * p0, 0, ..., 0]^T
    b_vec = np.zeros(nt * n_space, dtype=float)
    b_vec[:n_space] = L_op @ p0

    # Sparsity per row (used in Theorem 6 & Theorem 17/18)
    nnz_per_row = int(np.max(np.count_nonzero(np.abs(A_mat) > 1e-15, axis=1)))

    return {
        "T_mat": T_mat,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "tensor_vs_block_max_diff": tensor_block_diff,
        "A_shape": A_mat.shape,
        "max_nnz_per_row": nnz_per_row,
    }


def _solve_lemma4_secular_roots(abs_l: float, nt: int) -> np.ndarray:
    """Solve Lemma 4 secular equation (Eq. 38) for all nt eigenvalues of A_j A_j^dagger.

    Secular equation:
        |l_j| * sin(nt * theta) + sin((nt + 1) * theta) = 0,  theta != k * pi.
    For all |l_j| < (nt + 1) / nt (including every stable mode |l_j| <= 1),
    all nt roots lie cleanly in the real open intervals (m*pi/nt, (m+1)*pi/nt)
    for m = 0, ..., nt - 1. If |l_j| > (nt + 1) / nt, one root moves onto the
    hyperbolic branch theta = pi + i * phi, which is handled as a fallback.
    """
    if abs_l < 1e-14:
        # A_j = I_nt -> all eigenvalues alpha = 1
        return np.ones(nt, dtype=float)

    def f_real(theta: float) -> float:
        return abs_l * np.sin(nt * theta) + np.sin((nt + 1) * theta)

    roots_alpha = []
    eps = 1e-11
    for m in range(nt):
        lo = m * np.pi / nt + eps
        hi = (m + 1) * np.pi / nt - eps
        if lo >= hi:
            continue
        if f_real(lo) * f_real(hi) < 0:
            theta_star = brentq(f_real, lo, hi, xtol=1e-13)
            alpha = 1.0 + abs_l**2 + 2.0 * abs_l * np.cos(theta_star)
            roots_alpha.append(float(alpha))

    # Hyperbolic fallback branch theta = pi + i*phi when |l_j| > (nt + 1)/nt
    if len(roots_alpha) < nt:
        def f_hyp(phi: float) -> float:
            return np.sinh((nt + 1) * phi) - abs_l * np.sinh(nt * phi)

        phi_hi = 1.0
        while f_hyp(phi_hi) < 0 and phi_hi < 500.0:
            phi_hi *= 2.0
        if f_hyp(1e-9) * f_hyp(phi_hi) < 0:
            phi_star = brentq(f_hyp, 1e-9, phi_hi, xtol=1e-13)
            alpha = 1.0 + abs_l**2 - 2.0 * abs_l * np.cosh(phi_star)
            roots_alpha.append(float(alpha))

    return np.sort(np.asarray(roots_alpha, dtype=float))


def layer_0_7_spectral_and_conditioning_analysis(
    config: DDEConfig,
    grid_info: Dict[str, Any],
    spatial_ops: Dict[str, Any],
    L_info: Dict[str, Any],
    A_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Layer 0.7: Analytical vs numerical spectral checks and condition numbers.

    Verifies:
      - Lemma 3 (Eq. 33): Eigenvalues mu_j of H and mu_{j_1}+...+mu_{j_d} of M;
        diagonalization F^dagger H F = diag(mu).
      - Eq. (32) / Lemma 9 (Eq. 48): Eigenvalues l_j = 1 + dt * mu_j of L.
      - Lemma 4 (Eqs. 34-38): Block diagonalization (I otimes F^dagger) A (I otimes F),
        tridiagonal blocks A_j A_j^dagger, and secular roots alpha_j.
      - Theorem 5 (Eqs. C1-C4): Singular values sigma_min, sigma_max, ||A||, ||A^-1||, kappa_A.
      - Lemma 11 (Eqs. D1-D4): Conditioning of L (kappa_L <= 5 in restricted regime).
      - Lemma 12 (Eq. 56, D5-D10): Lower bound ||L^tau |0>||_2^2 >= (4*sqrt(tau))^{-d}.
    """
    nx = grid_info["nx"]
    nt = grid_info["nt"]
    dx = grid_info["dx"]
    dt = grid_info["dt"]
    d = config.d
    a = config.a
    D = config.D
    T = config.T

    H_1d = spatial_ops["H_1d"]
    L_op = L_info["L_op"]
    T_mat = A_info["T_mat"]
    A_mat = A_info["A_mat"]

    # 1) Lemma 3: 1D eigenvalues mu_j and unitary DFT diagonalization check
    j_idx = np.arange(nx, dtype=float)
    mu_1d = (
        -4.0 * (D / (dx**2)) * (np.sin(np.pi * j_idx / nx) ** 2)
        + 1j * (a / dx) * np.sin(2.0 * np.pi * j_idx / nx)
    )
    F_1d = _unitary_dft_1d(nx)
    H_diag_check = F_1d.conj().T @ H_1d @ F_1d
    H_diag_err = float(np.max(np.abs(H_diag_check - np.diag(mu_1d))))

    # Build d-D eigenvalues mu_{j_1} + ... + mu_{j_d}
    grids_mu = np.meshgrid(*([mu_1d] * d), indexing="ij")
    mu_dd = np.sum(np.stack(grids_mu, axis=0), axis=0).ravel()
    l_dd = 1.0 + dt * mu_dd

    # Numerical vs analytical eigenvalues of L_op
    l_1d = 1.0 + dt * mu_1d
    F_dd = F_1d
    for _ in range(d - 1):
        F_dd = np.kron(F_dd, F_1d)
    L_diag_check = F_dd.conj().T @ L_op @ F_dd
    L_diag_err = float(np.max(np.abs(L_diag_check - np.diag(l_dd))))

    # 2) Lemma 4: Blocks A_j and A_j A_j^dagger across all d-D Fourier modes
    I_nt = np.eye(nt, dtype=complex)
    block_max_err = 0.0
    secular_max_err = 0.0
    all_block_singular_values = []

    for idx_mode, (mu_val, l_val) in enumerate(zip(mu_dd, l_dd)):
        # Eq. (35): A_j = (1 + dt*mu_j)*T - dt*mu_j*I
        A_j = l_val * T_mat - dt * mu_val * I_nt
        # Eq. (36): explicit tridiagonal formula for A_j A_j^dagger
        AA_dag_num = A_j @ A_j.conj().T
        AA_dag_formula = np.diag([1.0] + [1.0 + abs(l_val) ** 2] * (nt - 1))
        if nt > 1:
            AA_dag_formula = AA_dag_formula.astype(complex)
            AA_dag_formula += np.diag([-np.conj(l_val)] * (nt - 1), k=1)
            AA_dag_formula += np.diag([-l_val] * (nt - 1), k=-1)
        block_max_err = max(block_max_err, float(np.max(np.abs(AA_dag_num - AA_dag_formula))))

        # Numerical eigenvalues of A_j A_j^dagger
        eigs_num = np.sort(np.maximum(np.linalg.eigvalsh(AA_dag_num), 0.0))
        all_block_singular_values.append(np.sqrt(eigs_num))

        # Check secular equation (Eq. 38) on the first few modes + extreme modes
        if idx_mode < 8 or idx_mode == len(l_dd) // 2:
            eigs_sec = _solve_lemma4_secular_roots(float(abs(l_val)), nt)
            if len(eigs_sec) == nt:
                secular_max_err = max(secular_max_err, float(np.max(np.abs(eigs_num - eigs_sec))))

    sv_blocks = np.sort(np.concatenate(all_block_singular_values))
    sv_full = np.sort(np.linalg.svd(A_mat, compute_uv=False))
    svd_block_vs_full_err = float(np.max(np.abs(sv_blocks - sv_full)))

    sigma_min_A = float(sv_full[0])
    sigma_max_A = float(sv_full[-1])
    norm_A = sigma_max_A
    norm_A_inv = 1.0 / sigma_min_A
    kappa_A = sigma_max_A / sigma_min_A

    # Theorem 5 theoretical bounds (Eqs. C1-C4)
    sigma_min_lower_C1 = 2.0 / (nt * np.pi)
    regime_param_thm5 = (a**2) * T / (2.0 * nt * D)
    if regime_param_thm5 <= 1.0:
        kappa_A_thm5_bound = 2.0 * nt
    else:
        kappa_A_thm5_bound = nt * (1.0 + np.sqrt(regime_param_thm5))

    # 3) Lemma 11: Condition number kappa_L and Eq. (D1)/(D2) formula
    Y = D * dt / (dx**2)
    X = a * dt / (2.0 * dx)
    Z = 2.0 * np.pi * j_idx / nx
    lam_LLdag_1d_D2 = (
        (1.0 - 2.0 * Y) * ((1.0 - 2.0 * Y) + 4.0 * Y * np.cos(Z))
        + 4.0 * (Y**2) * (np.cos(Z) ** 2)
        + 4.0 * (X**2) * (np.sin(Z) ** 2)
    )
    sv_L_1d_formula = np.sqrt(np.maximum(lam_LLdag_1d_D2, 0.0))
    sv_L_dd = np.sort(np.abs(l_dd))
    sv_L_svd = np.sort(np.linalg.svd(L_op, compute_uv=False))
    L_sv_err = float(np.max(np.abs(sv_L_dd - sv_L_svd)))

    sigma_min_L = float(sv_L_svd[0])
    sigma_max_L = float(sv_L_svd[-1])
    kappa_L = sigma_max_L / sigma_min_L if sigma_min_L > 0 else np.inf
    kappa_L_1d_formula = float(np.max(sv_L_1d_formula) / np.min(sv_L_1d_formula))

    # 4) Lemma 12: Origin return / L2 collision norm bound for tau = 1..nt
    # Evaluate for point source at origin |0> (index nx//2 in each dimension)
    origin_multi = tuple([nx // 2] * d)
    origin_idx = int(np.ravel_multi_index(origin_multi, tuple([nx] * d)))
    ket_origin = np.zeros(nx**d, dtype=float)
    ket_origin[origin_idx] = 1.0

    taus = np.arange(1, nt + 1, dtype=int)
    norm_L_tau_origin_sq = np.zeros(nt, dtype=float)
    return_prob_2tau = np.zeros(nt, dtype=float)
    lemma12_lower_bound = 1.0 / ((4.0 * np.sqrt(taus.astype(float))) ** d)

    state = ket_origin.copy()
    for idx_t, tau in enumerate(taus):
        state = L_op @ state
        norm_L_tau_origin_sq[idx_t] = float(np.dot(state, state))
        # Also compute <0 | L^{2*tau} | 0> via Fourier eigenvalues (equals norm_L_tau_origin_sq when a=0)
        return_prob_2tau[idx_t] = float(np.mean(l_dd ** (2 * tau)).real)

    lemma12_satisfied = bool(np.all(norm_L_tau_origin_sq + 1e-14 >= lemma12_lower_bound))

    return {
        "mu_1d": mu_1d,
        "mu_dd": mu_dd,
        "l_1d": l_1d,
        "l_dd": l_dd,
        "H_diag_err": H_diag_err,
        "L_diag_err": L_diag_err,
        "block_AAdag_formula_err": block_max_err,
        "secular_eq_max_err": secular_max_err,
        "svd_block_vs_full_err": svd_block_vs_full_err,
        "sigma_min_A": sigma_min_A,
        "sigma_max_A": sigma_max_A,
        "norm_A": norm_A,
        "norm_A_inv": norm_A_inv,
        "kappa_A": kappa_A,
        "sigma_min_lower_C1": sigma_min_lower_C1,
        "regime_param_thm5": regime_param_thm5,
        "kappa_A_thm5_bound": kappa_A_thm5_bound,
        "lam_LLdag_1d_D2": lam_LLdag_1d_D2,
        "L_singular_values_err": L_sv_err,
        "sigma_min_L": sigma_min_L,
        "sigma_max_L": sigma_max_L,
        "kappa_L": kappa_L,
        "kappa_L_1d_formula": kappa_L_1d_formula,
        "taus": taus,
        "norm_L_tau_origin_sq": norm_L_tau_origin_sq,
        "return_prob_2tau": return_prob_2tau,
        "lemma12_lower_bound": lemma12_lower_bound,
        "lemma12_satisfied": lemma12_satisfied,
    }


def assemble_discretization(config: DDEConfig) -> DDEDiscretization:
    """Run Layers 0.1 through 0.7 in sequence and return a DDEDiscretization object."""
    theory = layer_0_1_theoretical_parameters_and_finance_benchmark(config)
    grid_info = layer_0_2_build_grid_and_verify_stability(config, theory=theory)
    init_info = layer_0_3_initial_and_analytic_distributions(config, grid_info)
    spatial_ops = layer_0_4_build_spatial_operators_H_and_M(config, grid_info)
    L_info = layer_0_5_build_transition_operator_L(
        config, grid_info, spatial_ops, init_info=init_info
    )
    A_info = layer_0_6_build_spacetime_system_A(grid_info, spatial_ops, L_info, init_info)
    spectra = layer_0_7_spectral_and_conditioning_analysis(
        config, grid_info, spatial_ops, L_info, A_info
    )

    return DDEDiscretization(
        config=config,
        theory=theory,
        dx=grid_info["dx"],
        dt=grid_info["dt"],
        nx=grid_info["nx"],
        nt=grid_info["nt"],
        stability_param=grid_info["stability_param"],
        stochastic_stable=grid_info["stochastic_stable"],
        lemma11_Y=grid_info["lemma11_Y"],
        a_over_D=grid_info["a_over_D"],
        lemma11_satisfied=grid_info["lemma11_satisfied"],
        thm1_error_bound=grid_info["thm1_error_bound"],
        x_1d=grid_info["x_1d"],
        t_grid=grid_info["t_grid"],
        coords_dd=grid_info["coords_dd"],
        p0=init_info["p0"],
        ket_p0=init_info["ket_p0"],
        p_exact_T=init_info["p_exact_T"],
        H_1d=spatial_ops["H_1d"],
        M=spatial_ops["M"],
        L_op=L_info["L_op"],
        T_mat=A_info["T_mat"],
        A_mat=A_info["A_mat"],
        b_vec=A_info["b_vec"],
        spectra=spectra,
    )


def verify_discretization_test_case(disc: DDEDiscretization) -> Dict[str, Any]:
    """Verify the discretized linear system A * p = b against explicit L^{nt} * p0
    and compare against the analytical periodic solution p_exact_T.
    """
    n_space = disc.nx**disc.config.d
    p_spacetime = np.linalg.solve(disc.A_mat, disc.b_vec)
    p_T_from_A = p_spacetime[-n_space:]

    p_T_from_L = disc.p0.copy()
    for _ in range(disc.nt):
        p_T_from_L = disc.L_op @ p_T_from_L

    max_diff_A_vs_L = float(np.max(np.abs(p_T_from_A - p_T_from_L)))
    l1_final = float(np.sum(np.abs(p_T_from_L)))
    min_prob = float(np.min(p_T_from_L))

    err_inf_vs_exact = None
    if disc.p_exact_T is not None:
        err_inf_vs_exact = float(np.max(np.abs(p_T_from_L - disc.p_exact_T)))

    return {
        "p_spacetime": p_spacetime,
        "p_T_from_A": p_T_from_A,
        "p_T_from_L": p_T_from_L,
        "max_diff_A_vs_L": max_diff_A_vs_L,
        "l1_final": l1_final,
        "min_prob": min_prob,
        "err_inf_vs_exact": err_inf_vs_exact,
        "thm1_error_bound": disc.thm1_error_bound,
    }
