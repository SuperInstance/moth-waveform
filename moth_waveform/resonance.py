"""Quantum resonance hearing for spline-tension curvature fields.

Division of labor, named honestly:
    CLASSICAL — the temporal geometry. The tension spline and the
    Walsh–Hadamard (WH) transform of its curvature field run on numpy/scipy.
    Time enters as knot geometry (the ducks), not as sample indices; the WH
    spectrum is the "sound" of that geometry.
    QUANTUM   — the ear and the judge, on the quantum-audio substrate
    (quantumaudio QPAM scheme API + qiskit Aer):
      * ear:    the WH spectrum is prepared as probability amplitudes via the
                scheme's own value_setting() extension point and measured in
                the computational basis — the substrate carries the spectrum
                as a waveform and the measurement reads its entropy.
      * judge:  two spectra (a candidate future vs the actual continuation)
                are prepared as quantum states and compared with a measured
                swap test — overlap = 2·P(ancilla 0) − 1 = |⟨ψ|φ⟩|². This is
                quantum-native similarity judging: interference, not dot
                product. (At 6 qubits it is simulable; the protocol is the
                hardware-honest part and runs unchanged on real QPUs.)

Calibration receipt (2000 shots, AerSimulator, seed 42/1/13 — see repo
README for the raw numbers): planted-periodic curvature 1.98 bits,
phase-shuffled 4.54 bits, spline-smoothed noise 3.98 bits of 6. Thresholds
below sit one-sided inside that >2-bit gap.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import quantumaudio as qaudio
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from scipy.interpolate import CubicSpline

# calibrated thresholds — see module docstring
_EAR_STRUCTURE_MAX = 3.0   # below: the board is singing (regime structure)
_EAR_NOISE_MIN = 3.5       # above: no tension worth hearing (refuse to guess)
_SWAP_RES_MIN = 0.3        # below: candidate resonates with nothing — refuse

_sim = AerSimulator()
_sch = qaudio.load_scheme("QPAM")


# -- classical temporal geometry ----------------------------------------------


def curvature_field(series, n_grid: int = 64) -> np.ndarray:
    """Tension-spline curvature resampled to a power-of-two grid, normalized.

    This field is the first-class temporal variable: prediction and hearing
    operate on it, never on raw samples.
    """
    y = np.asarray(series, dtype=float)
    n = y.size
    n_ducks = min(n, max(8, n // 4))
    idx = np.unique(np.linspace(0, n - 1, n_ducks).astype(int))
    t = np.linspace(0.0, 1.0, n)
    cs = CubicSpline(t[idx], y[idx], bc_type="natural")
    curv = cs(t, 2)
    grid = np.linspace(0.0, 1.0, n_grid)
    field = CubicSpline(t, curv)(grid)
    return field / (np.max(np.abs(field)) + 1e-12)


def wh_spectrum(field: np.ndarray) -> np.ndarray:
    """Walsh–Hadamard power spectrum of the curvature field (probabilities)."""
    a = np.asarray(field, dtype=float)
    a = a / np.linalg.norm(a)
    h = a.copy()
    step = 1
    while step < len(h):
        for i in range(0, len(h), step * 2):
            for j in range(i, i + step):
                x, y = h[j], h[j + step]
                h[j], h[j + step] = (x + y) / np.sqrt(2), (x - y) / np.sqrt(2)
        step *= 2
    p = h ** 2
    return p / p.sum()


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return -float(np.sum(nz * np.log2(nz)))


# -- quantum ear ---------------------------------------------------------------


def _encode_probs(p: np.ndarray) -> QuantumCircuit:
    """Prepare probabilities as amplitudes via the QPAM scheme's own extension
    point (value_setting is the library's seam for custom amplitude vectors —
    we bypass encode() deliberately: its (x+1)/2 offset answers a different
    question than the one the ear asks)."""
    n = int(np.log2(len(p)))
    circ = _sch.initialize_circuit(n, 0)
    _sch.value_setting(circ, np.sqrt(p))
    return circ


def _measure_probs(p: np.ndarray, shots: int, seed: int) -> np.ndarray:
    circ = _encode_probs(p).copy()
    _sch.measure(circ)  # library measure path; decode() wants encode() metadata
    counts = _sim.run(transpile(circ, _sim), shots=shots, seed=seed).result().get_counts()
    probs = np.zeros(len(p))
    total = max(1, sum(counts.values()))
    for bitstr, c in counts.items():
        probs[int(bitstr, 2)] = c / total
    return probs


def _swap_overlap(pa: np.ndarray, pb: np.ndarray, shots: int, seed: int) -> float:
    """Measured swap test: |⟨ψ|φ⟩|² from ancilla statistics."""
    n = int(np.log2(len(pa)))
    joint = np.kron(np.sqrt(pa), np.sqrt(pb))
    anc = QuantumRegister(1, "anc")
    qr = QuantumRegister(2 * n, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, anc, cr)
    qc.initialize(joint, list(range(2 * n)))
    qc.h(anc[0])
    for i in range(n):
        qc.cswap(anc[0], qr[i], qr[i + n])
    qc.h(anc[0])
    qc.measure(anc[0], cr[0])
    counts = _sim.run(transpile(qc, _sim), shots=shots, seed=seed).result().get_counts()
    p0 = counts.get("0", 0) / max(1, sum(counts.values()))
    return 2.0 * p0 - 1.0


# -- public API ----------------------------------------------------------------


def curvature_entropy(series, shots: int = 2000, seed: int = 42) -> float:
    """The ear on one series: measured spectrum entropy of its curvature field.

    Low (< _EAR_STRUCTURE_MAX): periodic tension — the board is singing.
    High (> _EAR_NOISE_MIN): no structure to hear.
    """
    return _entropy(_measure_probs(wh_spectrum(curvature_field(series)), shots, seed))


def resonance_gate(past_series, shots: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Is there tension worth hearing? The structure check with a receipt."""
    ent = curvature_entropy(past_series, shots=shots, seed=seed)
    return {
        "structured": ent <= _EAR_STRUCTURE_MAX,
        "entropy": ent,
        "threshold": _EAR_STRUCTURE_MAX,
        "reason": (
            f"curvature spectrum entropy {ent:.2f} bits ≤ {_EAR_STRUCTURE_MAX} — regime structure audible"
            if ent <= _EAR_STRUCTURE_MAX
            else f"curvature spectrum entropy {ent:.2f} bits > {_EAR_STRUCTURE_MAX} — no tension to hear"
        ),
    }


def judge_extrapolation(
    decomposition,
    candidates: Dict[str, np.ndarray],
    continuation: np.ndarray,
    shots: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Predictive engining, with receipts.

    1. Hear the past: no structure → REFUSED with a named reason (never guess).
    2. Hear each candidate: append it to the past, take the curvature of the
       extended region, and ask the swap test how much it resonates with the
       actual continuation's curvature. The candidates never judge themselves;
       the board does.
    3. If no candidate clears _SWAP_RES_MIN → REFUSED (nothing resonates).
    """
    past = decomposition.signal
    gate = resonance_gate(past, shots=shots, seed=seed)
    if not gate["structured"]:
        return {"verdict": "REFUSED", "winner": None, "resonances": {},
                "past_entropy": gate["entropy"], "reason": gate["reason"]}

    past_signal = np.asarray(past, dtype=float)
    actual_cont = np.asarray(continuation, dtype=float)
    actual_spec = wh_spectrum(curvature_field(np.concatenate([past_signal, actual_cont])))

    resonances: Dict[str, float] = {}
    for name, cand in candidates.items():
        extended = np.concatenate([past_signal, np.asarray(cand, dtype=float)])
        cand_spec = wh_spectrum(curvature_field(extended))
        resonances[name] = _swap_overlap(cand_spec, actual_spec, shots, seed)

    winner = max(resonances, key=resonances.get)
    best = resonances[winner]
    if best < _SWAP_RES_MIN:
        return {
            "verdict": "REFUSED", "winner": None, "resonances": resonances,
            "past_entropy": gate["entropy"],
            "reason": (
                f"best candidate '{winner}' resonance {best:.3f} < {_SWAP_RES_MIN} — "
                "no continuation resonates with the heard tension"
            ),
        }
    return {
        "verdict": "CONFIRMED", "winner": winner, "resonances": resonances,
        "past_entropy": gate["entropy"],
        "reason": (
            f"candidate '{winner}' resonance {best:.3f} with the actual continuation's "
            f"curvature spectrum (structure gate at {gate['entropy']:.2f} bits)"
        ),
    }
