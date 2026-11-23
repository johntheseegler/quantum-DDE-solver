"""Classical algorithms (Sec. III) for solving the d-dimensional periodic
drift-diffusion equation (DDE).

Implements every stage separately for all four classical methods described in:
    E. Devereux and A. Datta, "Quantum algorithms for solving a drift-diffusion
    equation: a complexity analysis", Phys. Rev. A 112, 042419 (2025)
    [arXiv:2505.21221v2].

Methods Implemented
-------------------
- C1 (Sec. III.B, Theorem 6, Fig. 1): System of linear equations via Conjugate Gradient
- C2 (Sec. III.C, Theorem 7, Fig. 2): Forward-time evolution L^{nt} p0
- C3 (Sec. III.D, Theorem 8, Fig. 3): Random walk / Coupled From The Past (CFTP) exact sampling
- C4 (Sec. III.E, Lemma 9, Theorem 10, Fig. 4): Diagonalization by Fast Fourier Transform (FFT)

All functions consume a ``DDEDiscretization`` object produced by ``discretization.py``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from complete_code.discretization import DDEDiscretization


# ======================================================================
# C1: Classical Linear Equations via Conjugate Gradient (Theorem 6, Fig. 1)
# ======================================================================


def c1_stage1_build_rhs(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 1 Stage 1: Compute b = (L*p0, 0, ..., 0)^T in O(d * nx^d) operations."""
    n_space = disc.nx**disc.config.d
    nt = disc.nt

    # Compute L * p0 exploiting the sparse (2d+1)-point stencil
    Lp0 = disc.L_op @ disc.p0
    b_vec = np.zeros(nt * n_space, dtype=float)
    b_vec[:n_space] = Lp0

    max_diff_from_disc = float(np.max(np.abs(b_vec - disc.b_vec)))
    return {
        "Lp0": Lp0,
        "b_vec": b_vec,
        "b_norm_2": float(np.linalg.norm(b_vec, ord=2)),
        "max_diff_from_disc_b": max_diff_from_disc,
    }


def c1_stage2_choose_delta_and_setup_normal_system(
    disc: DDEDiscretization, rhs_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Fig. 1 Stage 3 & Eq. (41): Choose tolerance delta = eps_c / nt and build
    the sparse SPD normal-equation operators A^T A and A^T b for Conjugate Gradient.
    """
    delta = disc.config.eps_c / float(disc.nt)

    A_csr = sparse.csr_matrix(disc.A_mat)
    AT_csr = A_csr.transpose().tocsr()
    b_vec = rhs_info["b_vec"]
    b_spd = AT_csr.dot(b_vec)

    return {
        "delta": delta,
        "A_csr": A_csr,
        "AT_csr": AT_csr,
        "b_vec": b_vec,
        "b_spd": b_spd,
        "system_size": int(A_csr.shape[0]),
        "nnz_A": int(A_csr.nnz),
    }


def c1_stage3_conjugate_gradient_solve(
    disc: DDEDiscretization,
    stage2_info: Dict[str, Any],
    max_iter: Optional[int] = None,
    tol_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Fig. 1 Stage 2: Execute Conjugate Gradient on A^T A p = A^T b.

    Stops when the residual in the A-norm of the normal system
    ||A x_k - b||_2 = ||x_k - x*||_{A^T A} <= delta (or tol_override),
    guaranteeing Eq. (41) 2-norm / inf-norm accuracy O(eps_c).
    """
    A_csr = stage2_info["A_csr"]
    AT_csr = stage2_info["AT_csr"]
    b_vec = stage2_info["b_vec"]
    b_spd = stage2_info["b_spd"]
    delta = stage2_info["delta"]
    target_tol = float(tol_override) if tol_override is not None else float(delta)

    n = stage2_info["system_size"]
    if max_iter is None:
        max_iter = max(2 * n, 500)

    def matvec_spd(v: np.ndarray) -> np.ndarray:
        return AT_csr.dot(A_csr.dot(v))

    x = np.zeros(n, dtype=float)
    r = b_spd.copy()
    p = r.copy()
    rs_old = float(np.dot(r, r))

    res_spd_history: List[float] = [float(np.sqrt(rs_old))]
    res_orig_history: List[float] = [float(np.linalg.norm(b_vec - A_csr.dot(x), ord=2))]

    iters = 0
    converged = res_orig_history[-1] <= target_tol

    while not converged and iters < max_iter:
        Ap = matvec_spd(p)
        denom = float(np.dot(p, Ap))
        if denom <= 1e-30:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.dot(r, r))
        iters += 1

        res_spd_history.append(float(np.sqrt(rs_new)))
        orig_res = float(np.linalg.norm(b_vec - A_csr.dot(x), ord=2))
        res_orig_history.append(orig_res)

        if orig_res <= target_tol:
            converged = True
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    n_space = disc.nx**disc.config.d
    p_spacetime = x.reshape(disc.nt, n_space)
    p_T = p_spacetime[-1]

    # Reference exact solve of A x = b for discretization-solver error checking
    x_exact_solve = np.linalg.solve(disc.A_mat, b_vec)
    solver_err_2 = float(np.linalg.norm(x - x_exact_solve, ord=2))
    solver_err_inf = float(np.max(np.abs(p_T - x_exact_solve[-n_space:])))

    err_inf_vs_exact = None
    if disc.p_exact_T is not None:
        err_inf_vs_exact = float(np.max(np.abs(p_T - disc.p_exact_T)))

    return {
        "p_spacetime": p_spacetime,
        "p_T": p_T,
        "iterations": iters,
        "converged": converged,
        "target_tol": target_tol,
        "final_orig_residual": res_orig_history[-1],
        "res_spd_history": np.asarray(res_spd_history, dtype=float),
        "res_orig_history": np.asarray(res_orig_history, dtype=float),
        "solver_err_2_vs_direct": solver_err_2,
        "solver_err_inf_vs_direct": solver_err_inf,
        "err_inf_vs_exact": err_inf_vs_exact,
        "l1_norm_T": float(np.sum(np.abs(p_T))),
    }


# ======================================================================
# C2: Classical Time Evolution (Theorem 7, Fig. 2)
# ======================================================================


def c2_stage1_apply_single_step_L(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 2 Stage 1: Apply transition matrix L once to initial vector p0."""
    p1 = disc.L_op @ disc.p0
    return {
        "p1": p1,
        "l1_norm_p1": float(np.sum(np.abs(p1))),
        "min_p1": float(np.min(p1)),
    }


def c2_stage2_repeat_nt_steps(
    disc: DDEDiscretization, stage1_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Fig. 2 Stage 2: Repeat L application for all remaining nt - 1 steps."""
    n_space = disc.nx**disc.config.d
    trajectory = np.zeros((disc.nt + 1, n_space), dtype=float)
    trajectory[0] = disc.p0
    trajectory[1] = stage1_info["p1"]

    l1_norms = [float(np.sum(np.abs(trajectory[0]))), stage1_info["l1_norm_p1"]]
    curr = trajectory[1].copy()
    for step in range(2, disc.nt + 1):
        curr = disc.L_op @ curr
        trajectory[step] = curr
        l1_norms.append(float(np.sum(np.abs(curr))))

    p_T = trajectory[-1]
    err_inf_vs_exact = None
    if disc.p_exact_T is not None:
        err_inf_vs_exact = float(np.max(np.abs(p_T - disc.p_exact_T)))

    return {
        "trajectory": trajectory,
        "p_T": p_T,
        "l1_norms": np.asarray(l1_norms, dtype=float),
        "max_l1_drift": float(np.max(np.abs(np.asarray(l1_norms) - 1.0))),
        "err_inf_vs_exact": err_inf_vs_exact,
    }


# ======================================================================
# C3: Classical Random Walk via CFTP / Exact Sampling (Theorem 8, Fig. 3)
# ======================================================================


def c3_stage1_extract_markov_kernel(disc: DDEDiscretization) -> Dict[str, Any]:
    """Sec. III.D Stage 1: Extract discrete Markov jump probabilities from L.

    Each column j of L defines the transition probabilities out of state j.
    In d dimensions on the periodic lattice Z_{nx}^d:
      - Stay at current coordinate with probability: 1 - 2*d*D*dt/dx^2
      - Shift -1 along axis m with probability: dt * (D/dx^2 + a/(2*dx))
      - Shift +1 along axis m with probability: dt * (D/dx^2 - a/(2*dx))
    """
    d = disc.config.d
    a = disc.config.a
    D = disc.config.D
    dx = disc.dx
    dt = disc.dt

    p_stay = 1.0 - 2.0 * d * D * dt / (dx**2)
    p_minus = dt * (D / (dx**2) + a / (2.0 * dx))  # step -1 along axis m
    p_plus = dt * (D / (dx**2) - a / (2.0 * dx))   # step +1 along axis m

    offsets: List[Tuple[int, ...]] = [tuple([0] * d)]
    probs: List[float] = [p_stay]
    for axis in range(d):
        step_minus = [0] * d
        step_minus[axis] = -1
        offsets.append(tuple(step_minus))
        probs.append(p_minus)

        step_plus = [0] * d
        step_plus[axis] = +1
        offsets.append(tuple(step_plus))
        probs.append(p_plus)

    offsets_arr = np.asarray(offsets, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    if np.any(probs_arr < -1e-12):
        raise ValueError(f"Negative transition probabilities detected: {probs_arr}")
    probs_arr = np.maximum(probs_arr, 0.0)
    probs_arr /= probs_arr.sum()

    # Cross-check that reconstructed Markov matrix matches disc.L_op identically
    n_space = disc.nx**d
    shape_dd = tuple([disc.nx] * d)
    L_check = np.zeros((n_space, n_space), dtype=float)
    coords_idx = np.array(np.unravel_index(np.arange(n_space), shape_dd)).T
    for off, prob in zip(offsets_arr, probs_arr):
        dest_idx = (coords_idx + off[None, :]) % disc.nx
        dest_flat = np.ravel_multi_index(dest_idx.T, shape_dd)
        L_check[dest_flat, np.arange(n_space)] += prob
    kernel_vs_L_err = float(np.max(np.abs(L_check - disc.L_op)))

    return {
        "p_stay": float(p_stay),
        "p_minus": float(p_minus),
        "p_plus": float(p_plus),
        "offsets": offsets_arr,
        "probs": probs_arr,
        "cdf": np.cumsum(probs_arr),
        "kernel_vs_L_err": kernel_vs_L_err,
    }


def c3_stage2_cftp_single_coupled_horizon(
    disc: DDEDiscretization, kernel_info: Dict[str, Any], seed: int = 313
) -> Dict[str, Any]:
    """Fig. 3 Stage 1: Coupled random-walk mapping over hitting horizon H = nt.

    Draws a single sequence of nt step offsets (t = -nt, ..., -1) and applies
    the coupled transition map Phi_{-nt}^0 to all n_x^d lattice sites simultaneously.
    Also records the trajectory of a single representative walker starting at x0.
    """
    rng = np.random.default_rng(seed)
    d = disc.config.d
    nx = disc.nx
    nt = disc.nt
    n_space = nx**d
    shape_dd = tuple([nx] * d)

    offsets = kernel_info["offsets"]
    cdf = kernel_info["cdf"]

    # Sample nt coupled transitions using uniform random numbers in [0, 1] (Assumption 1b)
    u_seq = rng.random(nt)
    step_choices = np.searchsorted(cdf, u_seq, side="right")
    step_choices = np.clip(step_choices, 0, len(offsets) - 1)
    step_vectors = offsets[step_choices]  # shape (nt, d)

    # Coupled map of all n_x^d grid states over horizon H = nt
    start_multi = np.array(np.unravel_index(np.arange(n_space), shape_dd)).T
    total_shift = np.sum(step_vectors, axis=0)
    end_multi = (start_multi + total_shift[None, :]) % nx
    coupled_map = np.ravel_multi_index(end_multi.T, shape_dd)

    # Track one representative walker starting from peak of p0
    start_peak = int(np.argmax(disc.p0))
    start_peak_multi = np.array(np.unravel_index(start_peak, shape_dd), dtype=int)
    path_multi = np.zeros((nt + 1, d), dtype=int)
    path_multi[0] = start_peak_multi
    for t_idx in range(nt):
        path_multi[t_idx + 1] = (path_multi[t_idx] + step_vectors[t_idx]) % nx
    path_flat = np.ravel_multi_index(path_multi.T, shape_dd)

    return {
        "u_seq": u_seq,
        "step_vectors": step_vectors,
        "total_shift": total_shift,
        "coupled_map": coupled_map,
        "rep_path_multi": path_multi,
        "rep_path_flat": path_flat,
    }


def c3_stage3_repeat_across_grid_and_aggregate(
    disc: DDEDiscretization,
    kernel_info: Dict[str, Any],
    samples_per_grid_list: Sequence[int] = (20, 80, 320, 1280, 5120),
    seed: int = 313,
) -> Dict[str, Any]:
    """Fig. 3 Stage 2 & Theorem 8: Repeat random-walk sampling across all nx^d
    starting points at t = 0 weighted by p0(x_i) to reconstruct p_tilde(x, T).

    Computes both:
      1) Exact Markov pushforward across all nx^d starting points (L^{nt} p0), and
      2) Empirical CFTP Monte Carlo estimates across increasing sample budgets M
         to verify O(M^{-1/2}) sampling convergence.
    """
    rng = np.random.default_rng(seed)
    d = disc.config.d
    nx = disc.nx
    nt = disc.nt
    n_space = nx**d
    shape_dd = tuple([nx] * d)

    offsets = kernel_info["offsets"]
    cdf = kernel_info["cdf"]
    start_coords = np.array(np.unravel_index(np.arange(n_space), shape_dd)).T

    # Exact pushforward summing over all nx^d initial sites: sum_i p0[i] * (L^{nt} e_i)
    L_nt = np.linalg.matrix_power(disc.L_op, nt)
    exact_markov_pushforward = np.zeros(n_space, dtype=float)
    for i in range(n_space):
        exact_markov_pushforward += disc.p0[i] * L_nt[:, i]

    # Empirical CFTP / random-walk sampling across all grid points (stratified by site i)
    max_M = int(max(samples_per_grid_list))
    # Draw independent nt-step walk shifts for M total trajectories distributed across sites
    # Allocate trajectories across all nx^d initial points proportional to p0
    # plus at least 1 baseline evaluation per grid site for exactness check
    p_emp_by_M: Dict[int, np.ndarray] = {}
    err_inf_vs_L_by_M: Dict[int, float] = {}
    err_inf_vs_exact_by_M: Dict[int, float] = {}

    for M in samples_per_grid_list:
        M_int = int(M)
        # Sample initial site index i in {0, ..., nx^d - 1} according to p0
        # (equivalently initiating walks from all grid points weighted by p0)
        init_sites = rng.choice(n_space, size=M_int, p=disc.p0)
        init_multi = start_coords[init_sites]

        # Simulate nt random walk steps via uniform numbers (Assumption 1b)
        u_mat = rng.random((M_int, nt))
        choices = np.searchsorted(cdf, u_mat, side="right")
        choices = np.clip(choices, 0, len(offsets) - 1)
        total_disp = offsets[choices].sum(axis=1)  # shape (M_int, d)

        final_multi = (init_multi + total_disp) % nx
        final_flat = np.ravel_multi_index(final_multi.T, shape_dd)
        counts = np.bincount(final_flat, minlength=n_space)
        p_emp = counts.astype(float) / float(M_int)

        p_emp_by_M[M_int] = p_emp
        err_inf_vs_L_by_M[M_int] = float(np.max(np.abs(p_emp - exact_markov_pushforward)))
        if disc.p_exact_T is not None:
            err_inf_vs_exact_by_M[M_int] = float(np.max(np.abs(p_emp - disc.p_exact_T)))

    finest_M = int(max_M)
    p_T_empirical = p_emp_by_M[finest_M]

    # Also compute log-log convergence slope of sampling error vs M
    M_arr = np.asarray(sorted(err_inf_vs_L_by_M.keys()), dtype=float)
    err_arr = np.asarray([err_inf_vs_L_by_M[int(m)] for m in M_arr], dtype=float)
    slope = float(np.polyfit(np.log(M_arr), np.log(np.maximum(err_arr, 1e-16)), 1)[0])

    return {
        "exact_markov_pushforward": exact_markov_pushforward,
        "p_T_empirical": p_T_empirical,
        "finest_M": finest_M,
        "M_values": M_arr,
        "err_inf_vs_L_by_M": err_inf_vs_L_by_M,
        "err_inf_vs_exact_by_M": err_inf_vs_exact_by_M,
        "emp_errors_vs_L": err_arr,
        "sampling_slope": slope,
        "l1_norm_empirical": float(np.sum(p_T_empirical)),
    }


# ======================================================================
# C4: Classical Diagonalization by FFT (Lemma 9, Theorem 10, Fig. 4)
# ======================================================================


def _taylor_sin(x: np.ndarray, num_terms: int) -> np.ndarray:
    """Evaluate sin(x) using its Maclaurin series truncated at ``num_terms`` odd terms."""
    result = np.zeros_like(x, dtype=float)
    term = x.copy()
    for m in range(num_terms):
        if m > 0:
            term = -term * (x * x) / ((2 * m) * (2 * m + 1))
        result = result + term
    return result


def c4_stage1_choose_delta(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 4 Stage 1 & Lemma 9: Choose delta = eps_c and determine the Taylor
    truncation order K = O(log(nt / delta)) for trigonometric eigenvalues.
    """
    delta = float(disc.config.eps_c)
    delta_prime = delta / float(disc.nt)

    # Sensitivity bound: |l_tilde - l| <= d * dt * (8*D/dx^2 + a/dx) * err_sin
    # (factor 2 in 8D comes from |sin^2(u) - sin_tilde^2(u)| <= 2 * |sin - sin_tilde|)
    sens = disc.config.d * disc.dt * (8.0 * disc.config.D / (disc.dx**2) + disc.config.a / disc.dx)
    req_sin_err = delta_prime / max(sens, 1e-15)

    # Remainder bound for sin(x) on [0, 2*pi] with K terms is (2*pi)^(2K+1) / (2K+1)!
    K = 1
    while True:
        rem_bound = ((2.0 * np.pi) ** (2 * K + 1)) / math.factorial(2 * K + 1)
        if rem_bound <= req_sin_err or K >= 40:
            break
        K += 1

    return {
        "delta": delta,
        "delta_prime": delta_prime,
        "sensitivity_factor": sens,
        "required_sin_accuracy": req_sin_err,
        "taylor_terms_K": K,
        "taylor_remainder_bound": rem_bound,
    }


def c4_stage2_fft_initial_state(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 4 Stage 2: Apply d-dimensional FFT to initial distribution p0."""
    shape_dd = tuple([disc.nx] * disc.config.d)
    p0_grid = disc.p0.reshape(shape_dd)
    psi0 = np.fft.fftn(p0_grid)
    return {
        "shape_dd": shape_dd,
        "p0_grid": p0_grid,
        "psi0": psi0,
    }


def c4_stage3_build_eigenvalues_and_apply_Lambda_nt(
    disc: DDEDiscretization,
    stage1_info: Dict[str, Any],
    stage2_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Fig. 4 Stage 3 & Lemma 9 (Eq. 48): Compute eigenvalues of L^{nt} via
    truncated Taylor series and exact trig, verify ||Lambda_tilde^{nt} - Lambda^{nt}|| <= delta,
    and multiply psi_0 by Lambda^{nt}.
    """
    nx = disc.nx
    nt = disc.nt
    d = disc.config.d
    dx = disc.dx
    dt = disc.dt
    a = disc.config.a
    D = disc.config.D
    K = stage1_info["taylor_terms_K"]

    j_idx = np.arange(nx, dtype=float)
    arg1 = np.pi * j_idx / nx
    arg2 = 2.0 * np.pi * j_idx / nx

    # Exact 1D eigenvalues mu_j (Lemma 3, Eq. 33)
    mu_1d_exact = -4.0 * (D / (dx**2)) * (np.sin(arg1) ** 2) + 1j * (a / dx) * np.sin(arg2)

    # Lemma 9 Taylor-approximated 1D eigenvalues
    sin1_taylor = _taylor_sin(arg1, K)
    sin2_taylor = _taylor_sin(arg2, K)
    mu_1d_taylor = -4.0 * (D / (dx**2)) * (sin1_taylor**2) + 1j * (a / dx) * sin2_taylor

    # Build d-dimensional eigenvalue tensors l_{j_1,...,j_d} = 1 + dt * sum_r mu_{j_r}
    mu_grids_exact = np.meshgrid(*([mu_1d_exact] * d), indexing="ij")
    mu_dd_exact = np.sum(np.stack(mu_grids_exact, axis=0), axis=0)
    Lambda_exact = 1.0 + dt * mu_dd_exact

    mu_grids_taylor = np.meshgrid(*([mu_1d_taylor] * d), indexing="ij")
    mu_dd_taylor = np.sum(np.stack(mu_grids_taylor, axis=0), axis=0)
    Lambda_taylor = 1.0 + dt * mu_dd_taylor

    Lambda_nt_exact = Lambda_exact**nt
    Lambda_nt_taylor = Lambda_taylor**nt

    lambda_nt_diff = float(np.max(np.abs(Lambda_nt_taylor - Lambda_nt_exact)))

    psi0 = stage2_info["psi0"]
    psi_nt_exact = Lambda_nt_exact * psi0
    psi_nt_taylor = Lambda_nt_taylor * psi0

    return {
        "Lambda_exact": Lambda_exact,
        "Lambda_taylor": Lambda_taylor,
        "Lambda_nt_exact": Lambda_nt_exact,
        "Lambda_nt_taylor": Lambda_nt_taylor,
        "lambda_nt_max_diff": lambda_nt_diff,
        "lemma9_bound_ok": bool(lambda_nt_diff <= stage1_info["delta"]),
        "psi_nt_exact": psi_nt_exact,
        "psi_nt_taylor": psi_nt_taylor,
    }


def c4_stage4_ifft_reconstruction(
    disc: DDEDiscretization,
    stage1_info: Dict[str, Any],
    stage3_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Fig. 4 Stage 4 & Theorem 10 (Eq. 51): Apply d-dimensional Inverse FFT to
    recover p_tilde_{nt}, verify Eq. (52), and check accuracy vs analytic reference.
    """
    p_T_exact_fft = np.fft.ifftn(stage3_info["psi_nt_exact"]).real.ravel()
    p_T_taylor_fft = np.fft.ifftn(stage3_info["psi_nt_taylor"]).real.ravel()

    eq52_l2_diff = float(np.linalg.norm(p_T_taylor_fft - p_T_exact_fft, ord=2))
    eq52_bound_ok = bool(eq52_l2_diff <= stage1_info["delta"])

    err_inf_vs_exact = None
    if disc.p_exact_T is not None:
        err_inf_vs_exact = float(np.max(np.abs(p_T_taylor_fft - disc.p_exact_T)))

    return {
        "p_T": p_T_taylor_fft,
        "p_T_exact_trig": p_T_exact_fft,
        "eq52_l2_diff": eq52_l2_diff,
        "eq52_delta": stage1_info["delta"],
        "eq52_bound_ok": eq52_bound_ok,
        "l1_norm_T": float(np.sum(np.abs(p_T_taylor_fft))),
        "err_inf_vs_exact": err_inf_vs_exact,
    }


def run_all_classical_solvers(
    disc: DDEDiscretization, samples_per_grid_list: Sequence[int] = (50, 200, 800, 3200)
) -> Dict[str, Any]:
    """Convenience runner executing all 4 classical solvers end-to-end on ``disc``."""
    c1_s1 = c1_stage1_build_rhs(disc)
    c1_s2 = c1_stage2_choose_delta_and_setup_normal_system(disc, c1_s1)
    c1_s3 = c1_stage3_conjugate_gradient_solve(disc, c1_s2)

    c2_s1 = c2_stage1_apply_single_step_L(disc)
    c2_s2 = c2_stage2_repeat_nt_steps(disc, c2_s1)

    c3_s1 = c3_stage1_extract_markov_kernel(disc)
    c3_s2 = c3_stage2_cftp_single_coupled_horizon(disc, c3_s1)
    c3_s3 = c3_stage3_repeat_across_grid_and_aggregate(
        disc, c3_s1, samples_per_grid_list=samples_per_grid_list
    )

    c4_s1 = c4_stage1_choose_delta(disc)
    c4_s2 = c4_stage2_fft_initial_state(disc)
    c4_s3 = c4_stage3_build_eigenvalues_and_apply_Lambda_nt(disc, c4_s1, c4_s2)
    c4_s4 = c4_stage4_ifft_reconstruction(disc, c4_s1, c4_s3)

    return {
        "C1_linear_cg": c1_s3,
        "C2_time_evolution": c2_s2,
        "C3_random_walk": c3_s3,
        "C4_fft": c4_s4,
    }
