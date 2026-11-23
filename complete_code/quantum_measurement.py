"""Multidimensional Quantum Probability Distribution Measurement Protocol
(Section IV.A.1, Lemmas 13-15, Theorem 16, Eqs. 57-60).

Implements the four-layer quantum readout pipeline from:
    E. Devereux and A. Datta, "Quantum algorithms for solving a drift-diffusion
    equation: a complexity analysis", Phys. Rev. A 112, 042419 (2025)
    [arXiv:2505.21221v2].

Each layer executes real Qiskit ``QuantumCircuit`` primitives (Hadamards,
phase-kickback rotations, and ``QFTGate.inverse()``) simulated with
``Statevector``, and can be invoked separately in individual notebook cells
or chained via ``run_multidimensional_measurement_protocol``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector


def qm_layer1_threshold_coordinate_sampling(
    ket_p: np.ndarray,
    p_norm2: Optional[float],
    eps_q: float,
    delta: float = 0.01,
    seed: int = 313,
) -> Dict[str, Any]:
    """Lemma 15 & Theorem 16 Step 1: Identify coordinates with p_tilde[i] >= eps_q.

    Draws N_samples = O(log(n / delta) / eps_q) computational-basis samples from
    the prepared quantum state |p_tilde> to isolate the active support S of size
    r = |S| <= ceil(1 / eps_q). Coordinates outside S are truncated to 0,
    contributing at most eps_q error in the infinity norm.
    """
    ket_arr = np.asarray(ket_p, dtype=complex).ravel()
    n = ket_arr.size

    ket_real = np.real(ket_arr)
    if p_norm2 is None:
        # Recover ||p_tilde||_2 from sum_i p_i = 1 => ||p_tilde||_2 = 1 / sum_i <i|p_tilde>
        amp_sum = float(np.sum(np.maximum(ket_real, 0.0)))
        if amp_sum <= 0.0:
            raise ValueError("Cannot infer p_norm2 from state with non-positive amplitude sum.")
        p_norm2 = 1.0 / amp_sum

    p_tilde = np.maximum(p_norm2 * ket_real, 0.0)
    if np.sum(p_tilde) > 0:
        p_tilde = p_tilde / np.sum(p_tilde)

    # Exact heavy coordinates satisfying Lemma 15 threshold p_i >= eps_q
    heavy_exact = np.where(p_tilde >= eps_q)[0]

    # Sample from |p_tilde> Born probabilities (Theorem 16 Step 1)
    born_probs = np.abs(ket_arr) ** 2
    born_probs /= np.sum(born_probs)

    # Lemma 15 sample count: N = ceil(log(n / (delta * eps_q)) / eps_q)
    # Include a modest constant factor inside O(log(n/delta)/eps_q) so quadratic
    # Born weights (p_i^2 / ||p||_2^2) reliably hit all coordinates >= eps_q.
    min_born_at_thresh = (eps_q**2) / max(p_norm2**2, 1e-12)
    effective_rate = min(eps_q, max(min_born_at_thresh, 0.25 * eps_q))
    n_samples = int(np.ceil(np.log(max(n, 2) / (delta * eps_q)) / effective_rate))

    rng = np.random.default_rng(seed)
    shots = rng.choice(n, size=n_samples, p=born_probs)
    unique_seen, counts_seen = np.unique(shots, return_counts=True)
    count_map = dict(zip(unique_seen.tolist(), counts_seen.tolist()))

    # Active coordinate set S: heavy coordinates discovered in sampling
    discovered_heavy = np.array([idx for idx in heavy_exact if idx in count_map], dtype=int)
    active_indices = discovered_heavy if discovered_heavy.size > 0 else heavy_exact
    if active_indices.size == 0:
        # Fallback if distribution is flatter than eps_q everywhere: retain top coordinate
        active_indices = np.array([int(np.argmax(p_tilde))], dtype=int)

    r = int(active_indices.size)
    max_trunc_err = float(np.max(p_tilde[np.setdiff1d(np.arange(n), active_indices)])) if r < n else 0.0

    return {
        "n_coords": n,
        "p_norm2": float(p_norm2),
        "p_tilde_ref": p_tilde,
        "born_probs": born_probs,
        "n_samples": n_samples,
        "unique_sampled_count": int(unique_seen.size),
        "heavy_exact_indices": heavy_exact,
        "active_indices": active_indices,
        "r_active": r,
        "max_allowed_r": int(np.ceil(1.0 / eps_q)),
        "truncation_err_inf": max_trunc_err,
        "all_heavy_discovered": bool(np.array_equal(discovered_heavy, heavy_exact)),
    }


def qm_layer2_build_hadamard_registers(
    layer1_info: Dict[str, Any], eps_q: float
) -> Dict[str, Any]:
    """Theorem 16 Steps 2 & 3 (Eq. 57): Build r registers of k_tilde qubits each
    and apply Hadamard gates H^{otimes k_tilde} to prepare uniform superpositions.
    """
    tilde_k = max(4, int(np.ceil(np.log2(4.0 / eps_q))))
    r = layer1_info["r_active"]
    dim_reg = 2**tilde_k

    hadamard_circuits: List[QuantumCircuit] = []
    hadamard_states: List[np.ndarray] = []

    for _ in range(r):
        qc = QuantumCircuit(tilde_k, name="qm_hadamard_reg")
        qc.h(range(tilde_k))
        sv = Statevector.from_instruction(qc).data
        hadamard_circuits.append(qc)
        hadamard_states.append(sv)

    return {
        "tilde_k": tilde_k,
        "dim_reg": dim_reg,
        "r_active": r,
        "total_ancilla_qubits": r * tilde_k,
        "phase_resolution": 4.0 / float(dim_reg),
        "hadamard_circuits": hadamard_circuits,
        "hadamard_states": hadamard_states,
    }


def qm_layer3_phase_oracle_query(
    layer1_info: Dict[str, Any],
    layer2_info: Dict[str, Any],
    eps_q: float,
) -> Dict[str, Any]:
    """Lemmas 13 & 14, Theorem 16 Step 4 (Eqs. 58-60): Encode p_tilde[S_j] into
    the phase of each k_tilde-qubit Qiskit register via single-qubit phase kickback
    e^{2*pi*i * x_j * p_tilde[S_j]}.
    """
    tilde_k = layer2_info["tilde_k"]
    active_indices = layer1_info["active_indices"]
    p_tilde = layer1_info["p_tilde_ref"]

    phase_circuits: List[QuantumCircuit] = []
    phase_states: List[np.ndarray] = []

    for j_pos, coord_idx in enumerate(active_indices):
        qc = layer2_info["hadamard_circuits"][j_pos].copy(name=f"qm_phase_reg_{coord_idx}")
        p_val = float(p_tilde[coord_idx])
        # For x = sum_{b=0}^{k-1} x_b * 2^b, apply P(2*pi * 2^b * p_val) to qubit b
        for bit in range(tilde_k):
            qc.p(2.0 * np.pi * (2**bit) * p_val, bit)
        sv = Statevector.from_instruction(qc).data
        phase_circuits.append(qc)
        phase_states.append(sv)

    # Verify Eq. (60) state accuracy for first register
    x_grid = np.arange(2**tilde_k, dtype=float)
    max_oracle_state_err = 0.0
    for j_pos, coord_idx in enumerate(active_indices):
        expected_sv = np.exp(2j * np.pi * x_grid * p_tilde[coord_idx]) / np.sqrt(2**tilde_k)
        err = float(np.max(np.abs(phase_states[j_pos] - expected_sv)))
        max_oracle_state_err = max(max_oracle_state_err, err)

    # Theorem 16 asymptotic U_p query complexity estimate:
    # O(log(n / delta) * log(1 / (eps_q * delta)) / eps_q)
    n = layer1_info["n_coords"]
    Up_calls_bound = int(np.ceil(np.log(max(n, 2) / 0.01) * np.log(1.0 / (eps_q * 0.01)) / eps_q))

    return {
        "phase_circuits": phase_circuits,
        "phase_states": phase_states,
        "max_oracle_state_err": max_oracle_state_err,
        "theorem16_Up_calls_order": Up_calls_bound,
    }


def qm_layer4_iqft_and_median_extraction(
    layer1_info: Dict[str, Any],
    layer2_info: Dict[str, Any],
    layer3_info: Dict[str, Any],
    eps_q: float,
    delta: float = 0.01,
    seed: int = 313,
) -> Dict[str, Any]:
    """Theorem 16 Step 5 & Median Boosting: Apply k_tilde-qubit Inverse QFT
    (``QFTGate(tilde_k).inverse()``) to each register, repeat R = O(log(1/(delta*eps_q)))
    times, and take the coordinate-wise median to extract p_tilde_tilde.
    """
    tilde_k = layer2_info["tilde_k"]
    dim_reg = layer2_info["dim_reg"]
    active_indices = layer1_info["active_indices"]
    p_tilde = layer1_info["p_tilde_ref"]
    n = layer1_info["n_coords"]

    iqft_gate = QFTGate(tilde_k).inverse()
    num_median_reps = max(7, int(np.ceil(4.0 * np.log(1.0 / (delta * eps_q)))))
    rng = np.random.default_rng(seed + 1)

    p_extracted = np.zeros(n, dtype=float)
    iqft_circuits: List[QuantumCircuit] = []
    iqft_probs_list: List[np.ndarray] = []
    per_coord_estimates: Dict[int, float] = {}

    for j_pos, coord_idx in enumerate(active_indices):
        qc = layer3_info["phase_circuits"][j_pos].copy(name=f"qm_iqft_reg_{coord_idx}")
        qc.append(iqft_gate, range(tilde_k))
        sv = Statevector.from_instruction(qc).data
        probs_m = np.abs(sv) ** 2
        probs_m /= np.sum(probs_m)

        # Sample R repetitions and take median (Theorem 16 confidence boost to 1 - delta)
        m_draws = rng.choice(dim_reg, size=num_median_reps, p=probs_m)
        p_est = float(np.median(m_draws.astype(float) / float(dim_reg)))

        p_extracted[coord_idx] = p_est
        per_coord_estimates[int(coord_idx)] = p_est
        iqft_circuits.append(qc)
        iqft_probs_list.append(probs_m)

    err_inf_vs_ptilde = float(np.max(np.abs(p_extracted - p_tilde)))
    err_2_vs_ptilde = float(np.linalg.norm(p_extracted - p_tilde, ord=2))

    return {
        "p_extracted": p_extracted,
        "p_extracted_normalized": p_extracted / max(np.sum(p_extracted), 1e-15),
        "num_median_reps": num_median_reps,
        "iqft_circuits": iqft_circuits,
        "iqft_probs_list": iqft_probs_list,
        "per_coord_estimates": per_coord_estimates,
        "err_inf_vs_ptilde": err_inf_vs_ptilde,
        "err_2_vs_ptilde": err_2_vs_ptilde,
        "within_eps_q": bool(err_inf_vs_ptilde <= eps_q + 1e-12),
        "l1_norm_raw": float(np.sum(p_extracted)),
    }


def run_multidimensional_measurement_protocol(
    ket_p: np.ndarray,
    p_norm2: Optional[float],
    eps_q: float,
    delta: float = 0.01,
    seed: int = 313,
) -> Dict[str, Any]:
    """Execute all 4 layers of the Theorem 16 quantum measurement protocol."""
    l1 = qm_layer1_threshold_coordinate_sampling(
        ket_p, p_norm2=p_norm2, eps_q=eps_q, delta=delta, seed=seed
    )
    l2 = qm_layer2_build_hadamard_registers(l1, eps_q=eps_q)
    l3 = qm_layer3_phase_oracle_query(l1, l2, eps_q=eps_q)
    l4 = qm_layer4_iqft_and_median_extraction(
        l1, l2, l3, eps_q=eps_q, delta=delta, seed=seed
    )
    return {
        "layer1_sampling": l1,
        "layer2_hadamard": l2,
        "layer3_phase_oracle": l3,
        "layer4_iqft_median": l4,
        "p_extracted": l4["p_extracted"],
        "p_extracted_normalized": l4["p_extracted_normalized"],
        "err_inf_vs_ptilde": l4["err_inf_vs_ptilde"],
        "within_eps_q": l4["within_eps_q"],
    }
