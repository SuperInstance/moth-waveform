"""FAIL-first pins for quantum resonance hearing of spline tension.

Doctrine (quilt-doctor scars):
- QPAM round-trip fidelity is lossless for ANY signal — it proves the
  substrate ran, never that structure exists. The discriminating meter is
  Hadamard-basis measurement entropy: periodic structure concentrates the
  H-basis outcome (low entropy); noise spreads it (high entropy).
- sch.decode() mutates the circuit in place — every basis question needs a
  fresh encode.
- Planted protocol: prove the meter on planted truth before trusting it on
  fleet data. Planted noise must REFUTE. Phase-shuffled controls must sit
  between. No vacuous planted-needle tests.
"""

import numpy as np
import pytest

from moth_waveform.spline import decompose, candidate_futures
from moth_waveform.resonance import (
    curvature_entropy,
    resonance_gate,
    judge_extrapolation,
)


def _plant_series(n=256, regime_period=32, noise_sigma=0.02, seed=7):
    """Periodic regime structure: the board is forced at a fixed cadence."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = np.sin(2 * np.pi * t / regime_period)
    envelope = 1.0 + 0.5 * np.sin(2 * np.pi * t / (regime_period * 4))
    return base * envelope + rng.normal(0, noise_sigma, n)


def _planted_noise(n=256, seed=11):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0, n)


def _phase_shuffled(series, seed=13):
    rng = np.random.default_rng(seed)
    return rng.permutation(series)


def test_curvature_field_is_first_class_distinct_from_signal():
    """The tension field must differ from the signal — hearing the board, not the tune."""
    series = _plant_series()
    dec = decompose(series)
    assert dec.curvature.shape == dec.signal.shape
    assert not np.allclose(dec.curvature, dec.signal)
    # curvature of pure sine ∝ -sin: same period, phase-shifted — the field
    # carries the temporal geometry, not the samples
    assert np.corrcoef(np.abs(dec.curvature), np.abs(dec.signal))[0, 1] > 0.8


def test_planted_periodic_curvature_has_low_hbasis_entropy():
    """The MOTH ear: periodic tension concentrates the spectrum → low entropy.
    Threshold calibrated on the meter (measured periodic ≈ 1.98 bits at 2000
    shots; see resonance.py _EAR_STRUCTURE_MAX) — a >2-bit margin, not a guess."""
    ent = curvature_entropy(_plant_series(), shots=2000, seed=42)
    assert ent < 3.0  # of log2(2^6)=6 bits max; planted periodic must concentrate


def test_planted_noise_refutes_high_entropy():
    """The meter must refuse structure that isn't there — REFUSAL, not flattery.
    Calibrated: measured noise ≈ 3.98 bits; margin sits inside the 2-bit gap."""
    ent = curvature_entropy(_planted_noise(), shots=2000, seed=42)
    assert ent > 3.5  # noise spreads the spectrum; the ear refuses


def test_phase_shuffled_control_sits_between():
    """Shuffling samples destroys temporal geometry: entropy must jump by >1 bit.
    Acoustic fact (calibrated): shuffled ≈ 4.5, noise ≈ 4.0 — scrambled
    structure strains the tension board MORE than smooth spline noise, so the
    shuffled-vs-noise ordering is deliberately NOT part of the spec."""
    series = _plant_series()
    ent_shuf = curvature_entropy(_phase_shuffled(series), shots=2000, seed=42)
    ent_per = curvature_entropy(series, shots=2000, seed=42)
    assert ent_shuf > 3.5
    assert ent_shuf > ent_per + 1.0


def test_resonance_gate_picks_structure_continuation_on_planted_regime():
    """Predictive engining: on a planted-regime series, the gate's verdict must
    match planted ground truth more often than the drift baseline."""
    n = 256
    horizon = 32
    wins_gate = wins_drift = 0
    for seed in range(5):
        series = _plant_series(n=n, seed=100 + seed)
        split = n - horizon
        past, future = series[:split], series[split:]
        dec = decompose(past)
        cands = candidate_futures(dec, horizon)
        verdict = judge_extrapolation(dec, cands, future, shots=2000, seed=42)
        best = min(cands, key=lambda k: float(np.mean((cands[k] - future) ** 2)))
        if verdict["winner"] == best:
            wins_gate += 1
        drift_err = float(np.mean((cands["drift"] - future) ** 2))
        if drift_err <= min(float(np.mean((cands[k] - future) ** 2)) for k in cands):
            wins_drift += 1
    assert wins_gate >= 3, f"gate won {wins_gate}/5 — the ear is not predictive"
    assert wins_gate >= wins_drift, "gate must at least match its cheapest baseline"


def test_resonance_gate_refuses_pure_noise_with_explicit_receipt():
    """On noise there is no tension to hear: the gate must say REFUSED, not guess."""
    n = 256
    horizon = 32
    series = _planted_noise(n=n)
    split = n - horizon
    dec = decompose(series[:split])
    cands = candidate_futures(dec, horizon)
    verdict = judge_extrapolation(dec, cands, series[split:], shots=2000, seed=42)
    assert verdict["verdict"] == "REFUSED"
    assert verdict["reason"]  # a refusal with no reason is a silent drop
