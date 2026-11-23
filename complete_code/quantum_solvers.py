"""Quantum algorithms (Sec. IV.B - IV.E) for solving the d-dimensional periodic
drift-diffusion equation (DDE), implemented with Qiskit ``QuantumCircuit``
and ``Statevector`` simulation.

Implements every stage separately for all four quantum algorithms from:
    E. Devereux and A. Datta, "Quantum algorithms for solving a drift-diffusion
    equation: a complexity analysis", Phys. Rev. A 112, 042419 (2025)
    [arXiv:2505.21221v2].

Algorithms Implemented
----------------------
- Q1 (Sec. IV.B, Theorems 17-18, Fig. 5): Quantum linear systems solver (HHL)
- Q2 (Sec. IV.C, Theorem 19, Fig. 6): Quantum time evolution via Hamiltonian sim + LCU
- Q3 (Sec. IV.D, Theorems 20-21, Fig. 7): Quantum random walk fast-forwarding (Chebyshev)
- Q4 (Sec. IV.E, Theorem 22, Fig. 8): Diagonalization via Quantum Fourier Transform (QFT)

All four methods extract the classical probability distribution p_tilde_tilde
using the Theorem 16 Multidimensional Measurement Protocol in ``quantum_measurement.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, QFTGate, UCRYGate, UnitaryGate
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from scipy.special import comb

from complete_code.classical_solvers import _taylor_sin, c4_stage1_choose_delta
from complete_code.discretization import DDEDiscretization, _unitary_dft_1d
from complete_code.quantum_measurement import run_multidimensional_measurement_protocol


# ======================================================================
# Shared Qubit-Embedding & Unitary Block-Encoding Helpers
# ======================================================================


def _pad_power_of_two(n: int) -> Tuple[int, int]:
    """Return (q_qubits, dim_padded) with dim_padded = 2**q_qubits >= n."""
    q = max(1, int(np.ceil(np.log2(max(n, 2)))))
    return q, 2**q


def _embed_vector(vec: np.ndarray, dim_padded: int) -> np.ndarray:
    """Zero-pad a vector to length dim_padded, clean sub-1e-13 noise, and normalize."""
    out = np.zeros(dim_padded, dtype=complex)
    out[: vec.size] = vec
    out[np.abs(out) < 1e-13] = 0.0
    if np.max(np.abs(out.imag)) < 1e-12:
        out = out.real.astype(complex)
    norm = float(np.linalg.norm(out, ord=2))
    if norm > 0.0:
        out /= norm
    return out


def _embed_contractive_block_unitary(A: np.ndarray, dim_padded: int) -> Tuple[np.ndarray, float]:
    """Construct an exact 1-ancilla unitary dilation U of size (2*dim_padded)^2
    whose ancilla-|0> top-left block equals A_padded / alpha, where alpha = max(||A||_2, 1.0).

    When the ancilla qubit is placed at the MSB (index q_sys), U[:dim_padded, :dim_padded]
    implements A_padded / alpha cleanly.
    """
    n = A.shape[0]
    A_pad = np.zeros((dim_padded, dim_padded), dtype=complex)
    A_pad[:n, :n] = A

    U_svd, s_vals, Vh_svd = np.linalg.svd(A_pad, full_matrices=True)
    alpha = max(float(np.max(s_vals)), 1.0)
    s_scaled = np.clip(s_vals / alpha, 0.0, 1.0)
    c_vals = np.sqrt(np.maximum(1.0 - s_scaled**2, 0.0))

    S_mat = np.diag(s_scaled)
    C_mat = np.diag(c_vals)

    top_left = U_svd @ S_mat @ Vh_svd
    top_right = U_svd @ C_mat @ Vh_svd
    bot_left = U_svd @ C_mat @ Vh_svd
    bot_right = -U_svd @ S_mat @ Vh_svd

    U_dil = np.block([[top_left, top_right], [bot_left, bot_right]])
    return U_dil, alpha


def _embed_dd_tensor_to_qubits(arr_dd: np.ndarray, nx: int, d: int, q_axis: int) -> np.ndarray:
    """Embed a d-dimensional array of shape (nx,)*d into (2**q_axis,)*d flattened."""
    dim_axis = 2**q_axis
    if dim_axis == nx:
        flat = arr_dd.ravel().astype(complex)
    else:
        padded = np.zeros(tuple([dim_axis] * d), dtype=complex)
        slices = tuple([slice(0, nx)] * d)
        padded[slices] = arr_dd
        flat = padded.ravel()
    flat[np.abs(flat) < 1e-13] = 0.0
    if np.max(np.abs(flat.imag)) < 1e-12:
        flat = flat.real.astype(complex)
    norm = float(np.linalg.norm(flat, ord=2))
    if norm > 0.0:
        flat /= norm
    return flat


def _extract_dd_tensor_from_qubits(vec_padded: np.ndarray, nx: int, d: int, q_axis: int) -> np.ndarray:
    """Extract the physical (nx,)*d subspace from a padded (2**q_axis,)*d statevector."""
    dim_axis = 2**q_axis
    if dim_axis == nx:
        return vec_padded[: nx**d]
    reshaped = vec_padded.reshape(tuple([dim_axis] * d))
    slices = tuple([slice(0, nx)] * d)
    return reshaped[slices].ravel()


def _axis_qubit_groups(d: int, q_axis: int) -> List[List[int]]:
    """Return Qiskit little-endian qubit indices for each spatial axis 0..d-1.

    In row-major flat indexing (i_0, ..., i_{d-1}), axis 0 is the most significant
    stride and axis d-1 is the least significant stride (qubit 0..q_axis-1).
    """
    return [
        list(range((d - 1 - axis) * q_axis, (d - axis) * q_axis))
        for axis in range(d)
    ]


def _axis_qft_gate(nx: int, q_axis: int, inverse: bool = False):
    """Return native QFTGate(q_axis) when nx is a power of 2, or an embedded
    unitary DFT gate on q_axis qubits when nx is not a power of 2.
    """
    dim_axis = 2**q_axis
    if dim_axis == nx:
        gate = QFTGate(q_axis)
        return gate.inverse() if inverse else gate

    # Embed exact nx-point unitary DFT F_nx (where F_nx^dag = forward FFT)
    F_nx = _unitary_dft_1d(nx)
    U_embed = np.eye(dim_axis, dtype=complex)
    U_embed[:nx, :nx] = F_nx.conj().T if inverse else F_nx
    label = f"DFT_{nx}_inv" if inverse else f"DFT_{nx}"
    return UnitaryGate(U_embed, label=label)


# ======================================================================
# Q1: Quantum Linear Systems Solver / HHL (Theorems 17 & 18, Fig. 5)
# ======================================================================


def q1_stage1a_calculate_b_via_L(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 5 Stage 1a & Lemma 11: Prepare |p0> in a Qiskit circuit and apply
    the 1-ancilla block-encoded transition operator L / ||L||_2.
    """
    n_space = disc.nx**disc.config.d
    q_space, dim_pad = _pad_power_of_two(n_space)

    U_dil, alpha_L = _embed_contractive_block_unitary(disc.L_op, dim_pad)
    p0_pad = _embed_vector(disc.ket_p0, dim_pad)

    # Circuit: qubits 0..q_space-1 = space, qubit q_space = block-encoding ancilla
    qc = QuantumCircuit(q_space + 1, name="q1_stage1a_L_block_encode")
    qc.prepare_state(p0_pad, list(range(q_space)))
    qc.append(UnitaryGate(U_dil, label="U_L"), list(range(q_space + 1)))

    sv_full = Statevector.from_instruction(qc).data
    # Ancilla |0> block corresponds to indices 0 .. dim_pad-1 (MSB = 0)
    block0 = sv_full[:n_space]
    Lp0_from_circuit = alpha_L * block0 * np.linalg.norm(disc.p0, ord=2)
    Lp0_exact = disc.L_op @ disc.p0
    block_encode_err = float(np.max(np.abs(Lp0_from_circuit.real - Lp0_exact)))

    norm_L_p0 = float(np.linalg.norm(block0, ord=2))
    ket_Lp0 = block0 / max(norm_L_p0, 1e-15)

    return {
        "lemma11_satisfied": disc.lemma11_satisfied,
        "kappa_L": disc.spectra["kappa_L"],
        "alpha_L": alpha_L,
        "q_space": q_space,
        "circuit": qc,
        "Lp0_from_circuit": Lp0_from_circuit.real,
        "ket_Lp0": ket_Lp0,
        "postselect_prob_L": norm_L_p0**2,
        "block_encode_err": block_encode_err,
    }


def q1_stage1b_initialize_b_state(
    disc: DDEDiscretization, stage1a_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Fig. 5 Stage 1b: Build the normalized space-time RHS state
    |b> = |0>_t otimes |L*p0>_x on q_st = ceil(log2(nt * nx^d)) qubits.
    """
    n_space = disc.nx**disc.config.d
    n_st = disc.nt * n_space
    q_space, dim_space = _pad_power_of_two(n_space)
    q_st, dim_st = _pad_power_of_two(n_st)

    # Because |b> only populates indices 0..n_space-1, preparing |L*p0> on the
    # lower q_space qubits leaves all upper time qubits in |0> identically.
    b_space_pad = _embed_vector(stage1a_info["ket_Lp0"], dim_space)

    qc = QuantumCircuit(q_st, name="q1_stage1b_init_b")
    qc.prepare_state(b_space_pad, list(range(q_space)))
    sv = Statevector.from_instruction(qc).data[:n_st]

    b_ref = disc.b_vec / np.linalg.norm(disc.b_vec, ord=2)
    state_err = float(np.linalg.norm(sv - b_ref, ord=2))

    return {
        "n_st": n_st,
        "q_st": q_st,
        "circuit": qc,
        "ket_b": sv,
        "ket_b_err": state_err,
    }


def q1_stage2_hhl_solve(
    disc: DDEDiscretization,
    stage1b_info: Dict[str, Any],
    eta: Optional[float] = None,
) -> Dict[str, Any]:
    """Fig. 5 Stage 2 & Theorems 17-18: Block-encoded HHL linear system solve
    A^{-1} |b> with singular-value phase quantization accuracy eta = eps_c (Eq. 61).
    """
    eta_val = float(eta) if eta is not None else float(disc.config.eps_c)
    ket_b = stage1b_info["ket_b"]
    n_st = stage1b_info["n_st"]
    n_space = disc.nx**disc.config.d

    # SVD of A: A = U * diag(sigma) * V^dagger => A^{-1} |b> = sum (beta_k / sigma_k) |v_k>
    U_A, sigmas, Vh_A = np.linalg.svd(disc.A_mat, full_matrices=False)
    betas = U_A.conj().T @ ket_b

    # Quantize singular values to b_clock bits matching Theorem 17 precision eta
    kappa_A = float(sigmas[0] / sigmas[-1])
    b_clock = max(8, int(np.ceil(np.log2(max(kappa_A / eta_val, 2.0)))) + 2)
    # Quantize reciprocals sigma_min / sigma_k onto 2^{-b_clock} grid
    sigma_min = float(sigmas[-1])
    scale_step = 1.0 / float(2**b_clock)
    recip_exact = sigma_min / sigmas
    recip_quant = np.round(recip_exact / scale_step) * scale_step
    recip_quant = np.maximum(recip_quant, scale_step)

    # HHL ancilla-|0> unnormalized output state: sum_k (beta_k * recip_quant_k) |v_k>
    coeffs_hhl = betas * recip_quant
    psi_anc0 = Vh_A.conj().T @ coeffs_hhl
    p_success_hhl = float(np.vdot(psi_anc0, psi_anc0).real)
    ket_p_spacetime = psi_anc0 / max(np.sqrt(p_success_hhl), 1e-15)

    # Exact normalized solution |A^{-1} b> for Eq. (61) error verification
    sol_exact = np.linalg.solve(disc.A_mat, ket_b)
    ket_exact_spacetime = sol_exact / np.linalg.norm(sol_exact, ord=2)
    # Align global phase for norm comparison
    phase_align = np.vdot(ket_exact_spacetime, ket_p_spacetime)
    if abs(phase_align) > 0:
        ket_p_spacetime *= np.exp(-1j * np.angle(phase_align))
    eq61_state_err = float(np.linalg.norm(ket_p_spacetime - ket_exact_spacetime, ord=2))

    # Build Qiskit circuit representing the prepared HHL output state
    q_st = stage1b_info["q_st"]
    _, dim_st = _pad_power_of_two(n_st)
    qc_out = QuantumCircuit(q_st, name="q1_stage2_hhl_state")
    qc_out.prepare_state(_embed_vector(ket_p_spacetime, dim_st), list(range(q_st)))

    # Project onto final time slice t = T (block nt - 1) and extract ||p_tilde(T)||_2
    slice_T = np.real(ket_p_spacetime[-n_space:])
    slice_T_pos = np.maximum(slice_T, 0.0)
    ket_p_T = slice_T / max(np.linalg.norm(slice_T, ord=2), 1e-15)
    # Since ||p_T||_1 = 1, ||p_T||_2 = ||slice_T||_2 / sum(slice_T)
    p_T_norm2 = float(np.linalg.norm(slice_T_pos, ord=2) / max(np.sum(slice_T_pos), 1e-15))

    return {
        "eta": eta_val,
        "b_clock_qubits": b_clock,
        "kappa_A": kappa_A,
        "hhl_postselect_prob": p_success_hhl,
        "aa_rounds_O_kappa": int(np.ceil(np.pi / (4.0 * np.arcsin(np.sqrt(min(p_success_hhl, 1.0)))))),
        "ket_p_spacetime": ket_p_spacetime,
        "ket_exact_spacetime": ket_exact_spacetime,
        "eq61_state_err": eq61_state_err,
        "eq61_bound_ok": bool(eq61_state_err <= eta_val),
        "ket_p_T": ket_p_T,
        "p_T_norm2": p_T_norm2,
        "circuit_out": qc_out,
    }


def q1_stage3_measurement(
    disc: DDEDiscretization, stage2_info: Dict[str, Any], seed: int = 313
) -> Dict[str, Any]:
    """Fig. 5 Stage 3: Extract p_tilde_tilde(x, T) via Theorem 16 measurement protocol."""
    meas = run_multidimensional_measurement_protocol(
        stage2_info["ket_p_T"],
        p_norm2=stage2_info["p_T_norm2"],
        eps_q=disc.config.eps_q,
        seed=seed,
    )
    p_out = meas["p_extracted"]
    err_vs_exact = (
        float(np.max(np.abs(p_out - disc.p_exact_T)))
        if disc.p_exact_T is not None
        else None
    )
    return {
        "measurement": meas,
        "p_T": p_out,
        "err_inf_vs_ptilde": meas["err_inf_vs_ptilde"],
        "err_inf_vs_exact": err_vs_exact,
        "total_eps_bound": disc.config.eps_c + disc.config.eps_q,
    }


# ======================================================================
# Q2: Quantum Time Evolution via Hamiltonian Sim + LCU (Theorem 19, Fig. 6)
# ======================================================================


def q2_stage1_initialize_state(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 6 Stage 1: Prepare |p0> on q_space spatial qubits using Qiskit."""
    n_space = disc.nx**disc.config.d
    q_space, dim_pad = _pad_power_of_two(n_space)
    qc = QuantumCircuit(q_space, name="q2_stage1_init_p0")
    qc.prepare_state(_embed_vector(disc.ket_p0, dim_pad), list(range(q_space)))
    sv = Statevector.from_instruction(qc).data[:n_space]
    return {
        "q_space": q_space,
        "dim_pad": dim_pad,
        "circuit": qc,
        "ket_p0": sv,
    }


def q2_stage2a_build_lcu_hamiltonian_and_shift(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 6 Stage 2a (Sec. IV.C, Ref. [13, 14]): Decompose M into drift
    Hamiltonian H_drift = i * M_drift (with unitary evolution e^{-i H_drift dt})
    and diffusion operator, and build the (2d+1)-term LCU Qiskit block-encoding
    circuit (PREPARE + SELECT + PREPARE^dag) for L.
    """
    d = disc.config.d
    nx = disc.nx
    n_space = nx**d
    dx = disc.dx
    dt = disc.dt
    a = disc.config.a
    D = disc.config.D

    # Skew-Hermitian drift / symmetric diffusion decomposition of M
    M_drift = 0.5 * (disc.M - disc.M.T)
    M_diff = 0.5 * (disc.M + disc.M.T)
    H_drift = 1j * M_drift
    drift_hermitian_err = float(np.max(np.abs(H_drift - H_drift.conj().T)))
    U_drift_dt = expm(-1j * H_drift * dt)

    # Exact (2d + 1)-term LCU of L = sum_m alpha_m U_m using cyclic shift unitaries
    shape_dd = tuple([nx] * d)
    coords = np.array(np.unravel_index(np.arange(n_space), shape_dd)).T

    alphas: List[float] = [1.0 - 2.0 * d * D * dt / (dx**2)]
    shift_unitaries: List[np.ndarray] = [np.eye(n_space, dtype=complex)]

    p_minus = dt * (D / (dx**2) + a / (2.0 * dx))
    p_plus = dt * (D / (dx**2) - a / (2.0 * dx))

    for axis in range(d):
        for step_dir, weight in [(-1, p_minus), (+1, p_plus)]:
            dest = coords.copy()
            dest[:, axis] = (dest[:, axis] + step_dir) % nx
            dest_flat = np.ravel_multi_index(dest.T, shape_dd)
            U_shift = np.zeros((n_space, n_space), dtype=complex)
            U_shift[dest_flat, np.arange(n_space)] = 1.0
            alphas.append(float(weight))
            shift_unitaries.append(U_shift)

    alphas_arr = np.asarray(alphas, dtype=float)
    lcu_lambda = float(np.sum(np.abs(alphas_arr)))  # = 1.0 when stochastic

    # Build Qiskit LCU circuit: ancilla register (q_anc) + spatial register (q_space)
    q_space, dim_pad = _pad_power_of_two(n_space)
    num_terms = len(alphas_arr)
    q_anc, dim_anc = _pad_power_of_two(num_terms)

    prep_vec = np.zeros(dim_anc, dtype=complex)
    prep_vec[:num_terms] = np.sqrt(np.maximum(alphas_arr / lcu_lambda, 0.0))

    # Build PREPARE unitary via Householder / QR completion of prep_vec as column 0
    V_prep = np.eye(dim_anc, dtype=complex)
    e0 = np.zeros(dim_anc, dtype=complex)
    e0[0] = 1.0
    if np.linalg.norm(prep_vec - e0) > 1e-14:
        # QR on [prep_vec, I] gives orthonormal basis with first col proportional to prep_vec
        Q_mat, R_mat = np.linalg.qr(np.column_stack([prep_vec, np.eye(dim_anc, dtype=complex)]))
        phase = np.vdot(prep_vec, Q_mat[:, 0])
        Q_mat[:, 0] *= np.exp(-1j * np.angle(phase))
        V_prep = Q_mat

    # Build SELECT block unitary on (q_anc + q_space) qubits:
    # ancilla qubits = MSB [q_space .. q_space + q_anc - 1], space = LSB [0 .. q_space - 1]
    U_select = np.eye(dim_anc * dim_pad, dtype=complex)
    for m in range(num_terms):
        U_pad = np.eye(dim_pad, dtype=complex)
        U_pad[:n_space, :n_space] = shift_unitaries[m]
        start = m * dim_pad
        end = (m + 1) * dim_pad
        U_select[start:end, start:end] = U_pad

    qc_lcu = QuantumCircuit(q_space + q_anc, name="q2_lcu_step")
    anc_qubits = list(range(q_space, q_space + q_anc))
    all_qubits = list(range(q_space + q_anc))
    qc_lcu.append(UnitaryGate(V_prep, label="PREP"), anc_qubits)
    qc_lcu.append(UnitaryGate(U_select, label="SELECT"), all_qubits)
    qc_lcu.append(UnitaryGate(V_prep.conj().T, label="PREP_dag"), anc_qubits)

    # Verify ancilla-|0> block reproduces L / lcu_lambda
    U_lcu_full = Operator(qc_lcu).data
    L_block = lcu_lambda * U_lcu_full[:n_space, :n_space]
    lcu_block_err = float(np.max(np.abs(L_block.real - disc.L_op)))

    return {
        "H_drift": H_drift,
        "U_drift_dt": U_drift_dt,
        "drift_hermitian_err": drift_hermitian_err,
        "alphas": alphas_arr,
        "lcu_lambda": lcu_lambda,
        "num_lcu_terms": num_terms,
        "q_anc": q_anc,
        "q_space": q_space,
        "dim_pad": dim_pad,
        "qc_lcu_step": qc_lcu,
        "U_lcu_step_block": U_lcu_full[:n_space, :n_space],
        "lcu_block_err": lcu_block_err,
    }


def q2_stage2b_evolve_and_amplify(
    disc: DDEDiscretization,
    stage1_info: Dict[str, Any],
    stage2a_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Fig. 6 Stage 2b & Theorem 19: Evolve nt steps using the LCU block-encoding
    operator, track survival probability ||L^k |p0>||_2^2, and apply amplitude
    amplification to prepare |p_tilde_{nt}>.
    """
    n_space = disc.nx**disc.config.d
    q_space = stage1_info["q_space"]
    dim_pad = stage1_info["dim_pad"]
    L_step = stage2a_info["U_lcu_step_block"]
    lcu_lambda = stage2a_info["lcu_lambda"]

    state = stage1_info["ket_p0"].astype(complex).copy()
    norm_history = [1.0]
    for _ in range(disc.nt):
        state = lcu_lambda * (L_step @ state)
        norm_history.append(float(np.linalg.norm(state, ord=2)))

    norm_final = norm_history[-1]
    aa_rounds = int(np.ceil(np.pi / (4.0 * np.arcsin(min(norm_final, 1.0)))))
    ket_p_T = state / max(norm_final, 1e-15)

    # Build final amplified Qiskit statevector circuit
    qc_final = QuantumCircuit(q_space, name="q2_stage2b_amplified")
    qc_final.prepare_state(_embed_vector(ket_p_T, dim_pad), list(range(q_space)))

    # Lemma 12 reference bound on ||L^{nt} |p0>||_2
    exact_L_p0 = np.linalg.matrix_power(disc.L_op, disc.nt) @ disc.ket_p0
    ket_exact = exact_L_p0 / np.linalg.norm(exact_L_p0, ord=2)
    state_err = float(np.linalg.norm(ket_p_T - ket_exact, ord=2))

    # ||p_T||_2 needed to map amplitudes |p_tilde> to L1 probability distribution
    p_T_exact_l1 = np.linalg.matrix_power(disc.L_op, disc.nt) @ disc.p0
    p_T_norm2 = float(np.linalg.norm(p_T_exact_l1, ord=2))

    return {
        "norm_history": np.asarray(norm_history, dtype=float),
        "norm_final": norm_final,
        "postselect_prob": norm_final**2,
        "aa_rounds": aa_rounds,
        "ket_p_T": ket_p_T,
        "p_T_norm2": p_T_norm2,
        "state_err_vs_Lnt": state_err,
        "circuit_final": qc_final,
    }


def q2_stage3_measurement(
    disc: DDEDiscretization, stage2b_info: Dict[str, Any], seed: int = 313
) -> Dict[str, Any]:
    """Fig. 6 Stage 3: Extract p_tilde_tilde(x, T) via Theorem 16 measurement protocol."""
    meas = run_multidimensional_measurement_protocol(
        stage2b_info["ket_p_T"],
        p_norm2=stage2b_info["p_T_norm2"],
        eps_q=disc.config.eps_q,
        seed=seed,
    )
    p_out = meas["p_extracted"]
    err_vs_exact = (
        float(np.max(np.abs(p_out - disc.p_exact_T)))
        if disc.p_exact_T is not None
        else None
    )
    return {
        "measurement": meas,
        "p_T": p_out,
        "err_inf_vs_ptilde": meas["err_inf_vs_ptilde"],
        "err_inf_vs_exact": err_vs_exact,
        "total_eps_bound": disc.config.eps_c + disc.config.eps_q,
    }


# ======================================================================
# Q3: Quantum Random Walk Fast-Forwarding (Theorems 20 & 21, Fig. 7)
# ======================================================================


def q3_stage1_initialize_state(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 7 Stage 1: Prepare |p0> on q_space spatial qubits using Qiskit."""
    return q2_stage1_initialize_state(disc)


def q3_stage2a_build_quantum_walk_chebyshev_lcu(
    disc: DDEDiscretization,
    stage1_info: Dict[str, Any],
    eta: Optional[float] = None,
) -> Dict[str, Any]:
    """Fig. 7 Stage 2a & Theorem 20 (Eq. 80): Compute fast-forwarded quantum walk
    Chebyshev cutoff M_walk = O(sqrt(nt * log(1 / (eta * ||L^nt |p0>||_2)))) and
    the binomial Chebyshev weights c_m for L^{nt} approx sum_{m=0}^{M_walk} c_m T_m(L).
    """
    eta_val = float(eta) if eta is not None else float(disc.config.eps_c)
    nt = disc.nt

    # Compute exact ||L^{nt} |p0>||_2 for Theorem 20 step bound Eq. (80)
    L_nt_ket = np.linalg.matrix_power(disc.L_op, nt) @ disc.ket_p0
    norm_Lnt_ket = float(np.linalg.norm(L_nt_ket, ord=2))

    # Eq. (80) fast-forwarding step count
    arg_log = max(2.0 / (eta_val * max(norm_Lnt_ket, 1e-12)), 2.0)
    M_walk = min(nt, int(np.ceil(np.sqrt(2.0 * nt * np.log(arg_log)))))

    # Exact expansion: x^{nt} = sum_{k=0}^{nt} (binom(nt, k) / 2^{nt}) * T_{|nt - 2k|}(x)
    # Note: each index m = |nt - 2k| collects contributions from k and nt - k (except m=0).
    cheb_coeffs = np.zeros(M_walk + 1, dtype=float)
    for k in range(nt + 1):
        m = abs(nt - 2 * k)
        if m <= M_walk:
            cheb_coeffs[m] += float(comb(nt, k, exact=False)) / float(2.0**nt)

    return {
        "eta": eta_val,
        "nt": nt,
        "norm_Lnt_ket": norm_Lnt_ket,
        "M_walk": M_walk,
        "speedup_ratio_nt_over_M": float(nt) / float(max(M_walk, 1)),
        "cheb_coeffs": cheb_coeffs,
        "cheb_l1_weight": float(np.sum(np.abs(cheb_coeffs))),
    }


def q3_stage2b_fast_forward_and_amplify(
    disc: DDEDiscretization,
    stage1_info: Dict[str, Any],
    stage2a_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Fig. 7 Stage 2b & Theorem 21 (Eq. 79): Evaluate the M_walk-step Chebyshev
    quantum walk polynomial W_M(L) |p0>, embed W_M(L) in a Qiskit 1-ancilla
    block-encoding circuit, and verify || |p_tilde> - L^nt|p0>/||L^nt|p0>|| ||_2 <= eta.
    """
    n_space = disc.nx**disc.config.d
    q_space = stage1_info["q_space"]
    dim_pad = stage1_info["dim_pad"]
    M_walk = stage2a_info["M_walk"]
    coeffs = stage2a_info["cheb_coeffs"]

    # Build Chebyshev walk operator W_M(L) = sum_{m=0}^{M_walk} c_m T_m(L)
    # using the 3-term quantum-walk recurrence requiring only M_walk applications of L
    I_sp = np.eye(n_space, dtype=float)
    T_prev = I_sp.copy()
    W_op = coeffs[0] * T_prev

    if M_walk >= 1:
        T_curr = disc.L_op.copy()
        W_op = W_op + coeffs[1] * T_curr
        for m in range(2, M_walk + 1):
            T_next = 2.0 * (disc.L_op @ T_curr) - T_prev
            W_op = W_op + coeffs[m] * T_next
            T_prev, T_curr = T_curr, T_next

    # Block-encode W_op into a (q_space + 1)-qubit Qiskit circuit
    U_dil, alpha_W = _embed_contractive_block_unitary(W_op, dim_pad)
    qc = QuantumCircuit(q_space + 1, name="q3_walk_fast_forward")
    qc.prepare_state(_embed_vector(stage1_info["ket_p0"], dim_pad), list(range(q_space)))
    qc.append(UnitaryGate(U_dil, label=f"U_Walk(M={M_walk})"), list(range(q_space + 1)))

    sv_full = Statevector.from_instruction(qc).data
    anc0_block = sv_full[:n_space] * alpha_W
    norm_walk = float(np.linalg.norm(anc0_block, ord=2))
    ket_p_T = anc0_block / max(norm_walk, 1e-15)

    # Compare against exact normalized L^{nt} |p0> (Eq. 79)
    exact_vec = np.linalg.matrix_power(disc.L_op, disc.nt) @ disc.ket_p0
    ket_exact = exact_vec / np.linalg.norm(exact_vec, ord=2)
    eq79_err = float(np.linalg.norm(ket_p_T - ket_exact, ord=2))

    # Also compute error convergence curve for m = 2 .. nt (for visualization)
    m_sweep = sorted(set(list(range(2, disc.nt + 1, max(1, disc.nt // 10))) + [M_walk, disc.nt]))
    sweep_errs = []
    l_dd = disc.spectra["l_dd"]
    exact_pow = l_dd**disc.nt
    for m_lim in m_sweep:
        tot = np.zeros_like(l_dd)
        for k in range(disc.nt + 1):
            m = abs(disc.nt - 2 * k)
            if m <= m_lim:
                tot += (float(comb(disc.nt, k, exact=False)) / float(2.0**disc.nt)) * np.cos(
                    m * np.arccos(l_dd + 0j)
                )
        sweep_errs.append(float(np.max(np.abs(tot - exact_pow))))

    aa_rounds = int(np.ceil(np.pi / (4.0 * np.arcsin(min(norm_walk, 1.0)))))
    p_T_exact_l1 = np.linalg.matrix_power(disc.L_op, disc.nt) @ disc.p0
    p_T_norm2 = float(np.linalg.norm(p_T_exact_l1, ord=2))

    return {
        "circuit": qc,
        "alpha_W": alpha_W,
        "norm_walk": norm_walk,
        "aa_rounds": aa_rounds,
        "ket_p_T": ket_p_T,
        "p_T_norm2": p_T_norm2,
        "eq79_state_err": eq79_err,
        "eq79_bound_ok": bool(eq79_err <= stage2a_info["eta"]),
        "m_sweep": np.asarray(m_sweep, dtype=int),
        "sweep_errs": np.asarray(sweep_errs, dtype=float),
    }


def q3_stage3_measurement(
    disc: DDEDiscretization, stage2b_info: Dict[str, Any], seed: int = 313
) -> Dict[str, Any]:
    """Fig. 7 Stage 3: Extract p_tilde_tilde(x, T) via Theorem 16 measurement protocol."""
    return q2_stage3_measurement(disc, stage2b_info, seed=seed)


# ======================================================================
# Q4: Diagonalization by Quantum Fourier Transform (Theorem 22, Fig. 8)
# ======================================================================


def q4_stage1_initialize_state(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 8 Stage 1: Build a (q_space + 1)-qubit Qiskit circuit preparing |p0>
    on the spatial register and |0> on the ancilla qubit (qubit q_space).
    """
    d = disc.config.d
    nx = disc.nx
    q_axis, dim_axis = _pad_power_of_two(nx)
    q_space = d * q_axis
    dim_pad = 2**q_space

    p0_dd = disc.ket_p0.reshape(tuple([nx] * d))
    p0_embedded = _embed_dd_tensor_to_qubits(p0_dd, nx, d, q_axis)

    sites = list(range(q_space))
    ancilla = q_space
    qc = QuantumCircuit(q_space + 1, name="q4_qft_diagonalization")
    qc.prepare_state(p0_embedded, sites)

    sv_full = Statevector.from_instruction(qc).data
    ket_p0_qiskit = _extract_dd_tensor_from_qubits(sv_full[:dim_pad], nx, d, q_axis)

    return {
        "d": d,
        "nx": nx,
        "q_axis": q_axis,
        "dim_axis": dim_axis,
        "q_space": q_space,
        "dim_pad": dim_pad,
        "sites": sites,
        "ancilla": ancilla,
        "uses_native_QFTGate": bool(dim_axis == nx),
        "circuit_s1": qc,
        "ket_p0_qiskit": ket_p0_qiskit,
    }


def q4_stage2_apply_qft(
    disc: DDEDiscretization, stage1_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Fig. 8 Stage 2 (Eq. 95): Apply d-dimensional forward QFT (QFTGate.inverse()
    on each axis register) to transform |p0> -> |psi_0>.
    """
    d = stage1_info["d"]
    nx = stage1_info["nx"]
    q_axis = stage1_info["q_axis"]
    dim_pad = stage1_info["dim_pad"]

    qc = stage1_info["circuit_s1"].copy(name="q4_stage2_qft")
    axis_groups = _axis_qubit_groups(d, q_axis)
    qft_gate_1d = _axis_qft_gate(nx, q_axis, inverse=True)

    for grp in axis_groups:
        qc.append(qft_gate_1d, grp)

    sv_full = Statevector.from_instruction(qc).data
    psi0_phys = _extract_dd_tensor_from_qubits(sv_full[:dim_pad], nx, d, q_axis)

    # Cross-check against unitary d-D FFT
    p0_grid = disc.ket_p0.reshape(tuple([nx] * d))
    psi0_ref = np.fft.fftn(p0_grid, norm="ortho").ravel()
    qft_err = float(np.max(np.abs(psi0_phys - psi0_ref)))

    return {
        "circuit_s2": qc,
        "axis_groups": axis_groups,
        "psi0_phys": psi0_phys,
        "psi0_l2_norm": float(np.linalg.norm(psi0_phys, ord=2)),
        "qft_vs_fft_err": qft_err,
    }


def q4_stage3a_calculate_eigenvalues(disc: DDEDiscretization) -> Dict[str, Any]:
    """Fig. 8 Stage 3a & Lemma 9: Precompute eigenvalues l_j^{nt} up to accuracy
    delta = eps_c using truncated Maclaurin series, and extract phases & UCRY angles.
    """
    s1 = c4_stage1_choose_delta(disc)
    K = s1["taylor_terms_K"]

    nx = disc.nx
    nt = disc.nt
    d = disc.config.d
    dx = disc.dx
    dt = disc.dt
    a = disc.config.a
    D = disc.config.D

    j_idx = np.arange(nx, dtype=float)
    arg1 = np.pi * j_idx / nx
    arg2 = 2.0 * np.pi * j_idx / nx

    mu_1d_taylor = (
        -4.0 * (D / (dx**2)) * (_taylor_sin(arg1, K) ** 2)
        + 1j * (a / dx) * _taylor_sin(arg2, K)
    )
    mu_grids = np.meshgrid(*([mu_1d_taylor] * d), indexing="ij")
    Lambda_taylor = 1.0 + dt * np.sum(np.stack(mu_grids, axis=0), axis=0)
    Lambda_nt_taylor = Lambda_taylor**nt

    # Clip magnitudes to [0, 1] for exact unitary ancilla rotation angle 2*arccos(|l^{nt}|)
    mags = np.clip(np.abs(Lambda_nt_taylor), 0.0, 1.0)
    phases = np.angle(Lambda_nt_taylor)

    # Compare vs exact eigenvalues from disc.spectra
    l_dd_exact_nt = (disc.spectra["l_dd"].reshape(tuple([nx] * d))) ** nt
    eig_err = float(np.max(np.abs(Lambda_nt_taylor - l_dd_exact_nt)))

    return {
        "taylor_terms_K": K,
        "Lambda_nt_taylor": Lambda_nt_taylor,
        "magnitudes": mags,
        "phases": phases,
        "eig_max_err_vs_exact": eig_err,
    }


def q4_stage3bc_ancilla_rotation_and_amplitude_amplification(
    disc: DDEDiscretization,
    stage1_info: Dict[str, Any],
    stage2_info: Dict[str, Any],
    stage3a_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Fig. 8 Stages 3b & 3c (Eq. 96): Apply DiagonalGate(exp(i*phi)) on the
    Fourier register and UCRYGate(2*arccos(|l^{nt}|)) targeting the ancilla qubit,
    measure success probability P(0) = ||L^{nt} |p0>||_2^2, and perform
    Amplitude Amplification to isolate |psi_{nt}>.
    """
    d = stage1_info["d"]
    nx = stage1_info["nx"]
    q_axis = stage1_info["q_axis"]
    dim_pad = stage1_info["dim_pad"]
    sites = stage1_info["sites"]
    ancilla = stage1_info["ancilla"]

    # Embed phases and magnitudes onto dim_pad = 2**q_space (padded entries get mag=0, phase=0)
    phases_dd = stage3a_info["phases"]
    mags_dd = stage3a_info["magnitudes"]

    diag_phases = np.ones(tuple([2**q_axis] * d), dtype=complex)
    diag_phases[tuple([slice(0, nx)] * d)] = np.exp(1j * phases_dd)

    mags_pad = np.zeros(tuple([2**q_axis] * d), dtype=float)
    mags_pad[tuple([slice(0, nx)] * d)] = mags_dd

    angles = (2.0 * np.arccos(np.clip(mags_pad.ravel(), 0.0, 1.0))).tolist()

    qc = stage2_info["circuit_s2"].copy(name="q4_stage3bc_ancilla_rot")
    qc.append(DiagonalGate(diag_phases.ravel().tolist()), sites)
    qc.append(UCRYGate(angles), [ancilla, *sites])

    sv_rot = Statevector.from_instruction(qc).data
    unitarity_norm = float(np.linalg.norm(sv_rot, ord=2))

    # Ancilla |0> block is sv_rot[:dim_pad]
    anc0_fourier_pad = sv_rot[:dim_pad]
    P_zero = float(np.vdot(anc0_fourier_pad, anc0_fourier_pad).real)
    aa_rounds = int(np.ceil(np.pi / (4.0 * np.arcsin(np.sqrt(min(P_zero, 1.0))))))

    # Amplified Fourier state |psi_{nt}>
    psi_nt_pad = anc0_fourier_pad / max(np.sqrt(P_zero), 1e-15)
    psi_nt_phys = _extract_dd_tensor_from_qubits(psi_nt_pad, nx, d, q_axis)

    return {
        "circuit_s3": qc,
        "unitarity_norm": unitarity_norm,
        "P_zero": P_zero,
        "aa_rounds": aa_rounds,
        "psi_nt_phys": psi_nt_phys,
    }


def q4_stage4_apply_iqft(
    disc: DDEDiscretization,
    stage1_info: Dict[str, Any],
    stage3bc_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Fig. 8 Stage 4 & Theorem 22: Apply Inverse QFT (QFTGate on each axis) to
    transform |psi_{nt}> -> |p_tilde_{nt}>, and transpile the full Q4 Qiskit
    circuit to ["u", "cx"] gates to measure circuit depth and CX count.
    """
    d = stage1_info["d"]
    nx = stage1_info["nx"]
    q_axis = stage1_info["q_axis"]
    dim_pad = stage1_info["dim_pad"]

    qc = stage3bc_info["circuit_s3"].copy(name="q4_full_circuit")
    axis_groups = _axis_qubit_groups(d, q_axis)
    iqft_gate_1d = _axis_qft_gate(nx, q_axis, inverse=False)

    for grp in axis_groups:
        qc.append(iqft_gate_1d, grp)

    sv_final = Statevector.from_instruction(qc).data
    anc0_pad = sv_final[:dim_pad]
    anc0_phys = _extract_dd_tensor_from_qubits(anc0_pad, nx, d, q_axis)

    P_zero_check = float(np.vdot(anc0_phys, anc0_phys).real)
    ket_p_T = anc0_phys / max(np.sqrt(P_zero_check), 1e-15)

    # Transpile to basis gates ["u", "cx"] to report quantum circuit resources
    compiled = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
    cx_count = int(compiled.count_ops().get("cx", 0))

    # Compare QFT statevector against exact L^{nt} |p0> / norm
    pT_exact_l1 = np.linalg.matrix_power(disc.L_op, disc.nt) @ disc.p0
    pT_norm2 = float(np.linalg.norm(pT_exact_l1, ord=2))
    ket_ref = pT_exact_l1 / pT_norm2
    state_err_vs_Lnt = float(np.linalg.norm(ket_p_T - ket_ref, ord=2))

    # Also evaluate exact-sum reconstruction (matching drift_diffusion_report.pdf)
    p_T_statevector_exact = np.real(anc0_phys) / np.sum(np.real(anc0_phys))

    return {
        "circuit_full": qc,
        "circuit_compiled": compiled,
        "num_qubits": int(compiled.num_qubits),
        "compiled_depth": int(compiled.depth()),
        "compiled_size": int(compiled.size()),
        "compiled_cx_count": cx_count,
        "P_zero": P_zero_check,
        "ket_p_T": ket_p_T,
        "p_T_norm2": pT_norm2,
        "state_err_vs_Lnt": state_err_vs_Lnt,
        "p_T_statevector_exact": p_T_statevector_exact,
    }


def q4_stage5_measurement(
    disc: DDEDiscretization, stage4_info: Dict[str, Any], seed: int = 313
) -> Dict[str, Any]:
    """Fig. 8 Stage 5: Extract p_tilde_tilde(x, T) via Theorem 16 measurement protocol."""
    meas = run_multidimensional_measurement_protocol(
        stage4_info["ket_p_T"],
        p_norm2=stage4_info["p_T_norm2"],
        eps_q=disc.config.eps_q,
        seed=seed,
    )
    p_out = meas["p_extracted"]
    err_vs_exact = (
        float(np.max(np.abs(p_out - disc.p_exact_T)))
        if disc.p_exact_T is not None
        else None
    )
    return {
        "measurement": meas,
        "p_T": p_out,
        "err_inf_vs_ptilde": meas["err_inf_vs_ptilde"],
        "err_inf_vs_exact": err_vs_exact,
        "total_eps_bound": disc.config.eps_c + disc.config.eps_q,
    }


def run_all_quantum_solvers(disc: DDEDiscretization, seed: int = 313) -> Dict[str, Any]:
    """Convenience runner executing Q1, Q2, Q3, and Q4 end-to-end on ``disc``."""
    q1_1a = q1_stage1a_calculate_b_via_L(disc)
    q1_1b = q1_stage1b_initialize_b_state(disc, q1_1a)
    q1_2 = q1_stage2_hhl_solve(disc, q1_1b)
    q1_3 = q1_stage3_measurement(disc, q1_2, seed=seed)

    q2_1 = q2_stage1_initialize_state(disc)
    q2_2a = q2_stage2a_build_lcu_hamiltonian_and_shift(disc)
    q2_2b = q2_stage2b_evolve_and_amplify(disc, q2_1, q2_2a)
    q2_3 = q2_stage3_measurement(disc, q2_2b, seed=seed)

    q3_1 = q3_stage1_initialize_state(disc)
    q3_2a = q3_stage2a_build_quantum_walk_chebyshev_lcu(disc, q3_1)
    q3_2b = q3_stage2b_fast_forward_and_amplify(disc, q3_1, q3_2a)
    q3_3 = q3_stage3_measurement(disc, q3_2b, seed=seed)

    q4_1 = q4_stage1_initialize_state(disc)
    q4_2 = q4_stage2_apply_qft(disc, q4_1)
    q4_3a = q4_stage3a_calculate_eigenvalues(disc)
    q4_3bc = q4_stage3bc_ancilla_rotation_and_amplitude_amplification(
        disc, q4_1, q4_2, q4_3a
    )
    q4_4 = q4_stage4_apply_iqft(disc, q4_1, q4_3bc)
    q4_5 = q4_stage5_measurement(disc, q4_4, seed=seed)

    return {
        "Q1_linear_hhl": {"s1a": q1_1a, "s1b": q1_1b, "s2": q1_2, "s3": q1_3},
        "Q2_time_lcu": {"s1": q2_1, "s2a": q2_2a, "s2b": q2_2b, "s3": q2_3},
        "Q3_random_walk": {"s1": q3_1, "s2a": q3_2a, "s2b": q3_2b, "s3": q3_3},
        "Q4_qft_diag": {"s1": q4_1, "s2": q4_2, "s3a": q4_3a, "s3bc": q4_3bc, "s4": q4_4, "s5": q4_5},
    }
