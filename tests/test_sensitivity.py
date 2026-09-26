"""FAIL-first pins for duck-sensitivity receipts (Phase 2).

Doctrine transposed from Phase 1 and the fleet scars:
- The noise floor is MEASURED in-run (no-op kick + sigma_H from repeated
  preparations), never a constant — a gate that passes without a live floor
  measurement passes vacuously (fleet-murmur scar).
- Operator discipline: K_T (temporal kick — the thesis) and K_V (value kick —
  the control) are separate instruments. A receipt that conflates them hears
  the thesis everywhere.
- A carrier must out-kick its COHORT, not just the shot noise. Live failure
  receipt (2026-09-27, probe + suite): with sigma_E estimated from only 2
  self-swaps the echo floor collapsed to 0 and every real kick cleared both
  gates — "ducks [0..15] all clear floor 0.07 bits" on P1 AND on P4. Two
  in-run bars with divided labor now decide: the cohort bar (median +
  2.5·MAD, the sensitive DETECTOR) admits candidates, and the tie test
  (top-two within 2σ_H, the specific DISCRIMINATOR) decides whether one
  duck is named. At 3·MAD the verdict hinged on a 0.001-bit coincidence —
  each test must do one job.
- The no-op health gate runs at 4·sigma_H, not 2: sigma_H from 6 preps
  carries ~5 dof, and a 2-sigma health check false-refused a healthy
  instrument live (probe receipt: noop_dH -0.121 vs floor 0.065 -> spurious
  REFUSED; also flipped P4 between suite and standalone runs).
- Planted protocol before fleet data: P1 plants a carrier duck; P2 plants
  noise; P3 plants a broken instrument (no-op leak must self-refuse); P4
  plants a value carrier; P5 phase-shuffles P1.
- CALIBRATED_* constants quote the calibration run
  (tests/calibrate_sensitivity.py, same seeds/shots as the pins) with a
  named safety fraction — provenance, not guesses, same discipline as
  resonance.py's thresholds. Current values were initially written as
  pre-calibration estimates, watched FAIL, then set from the measured
  margins (the Phase 1 pattern, second rotation).
"""

import numpy as np
import pytest

from moth_waveform.spline import decompose
from moth_waveform.resonance import curvature_field, wh_spectrum
from moth_waveform.sensitivity import (
    CALIBRATED_CARRIER_MARGIN,
    CALIBRATED_NOISE_ENTROPY_MIN,
    _assemble_report,
    duck_sensitivity,
    field_from_knots,
    kick_duck,
)

# the no-op health gate factor, restated here so the pin dies if the
# instrument silently loosens it (value quoted from sensitivity.NOOP_K at
# calibration time; a change there must re-open this pin deliberately)
_NOOP_HEALTH_K = 4.0


def _plant_carrier_series(n=128, n_ducks=16, carrier_idx=7, seed=7):
    """One duck carries the regime: a sharp smooth notch centered exactly at
    knot `carrier_idx`'s position on a smooth periodic board. Kicking that
    knot in time misaligns the board's sharpest feature; kicking any other
    knot merely re-smooths its own neighborhood.

    Classical screen receipt (no Aer, 2026-09-27, depth 1.2): |ΔH_KT| 0.99
    at knot 7 vs 0.66 runner (knot 6), cohort bar 0.88 — knot 7 alone clears;
    K_V clears nothing (top 0.34). Knot 7 is a cliff duck: one direction
    re-smooths, the other scrambles — the design's asymmetry receipt, live.
    Depth is the SNR lever (the sub-gap width is invisible to a 16-knot
    spline — only knot VALUES reach the ear): 0.9 left a 0.25-bit top gap;
    1.5-2.0 woke knot 8 into a secondary carrier.

    (Three earlier plants died by screen and are named, not buried: (1) the
    plateau+spike plant put the attribution on the plateau's right EDGE —
    knot 8 out-kicked knot 7, 1.32 vs 1.01 in the quantum probe; (2) twin
    bumps at t7 and t7+1/3 left a 0.11 margin spread over four ducks; (3)
    dip-dominated boards (base amp ≤ 0.5) blew the pre-gate (H0 > 3.5) or
    made knot 8 the top responder — kicking it drags the dip.
    Planted what survives, named what died.)"""
    t = np.linspace(0.0, 1.0, n)
    idx = np.unique(np.linspace(0, n - 1, n_ducks).astype(int))
    x = idx / (n - 1)
    gap = float(np.mean(np.diff(x)))
    base = np.sin(2 * np.pi * 3 * t)
    s = base - 1.2 * np.exp(-((t - x[carrier_idx]) ** 2) / (2 * (gap / 6) ** 2))
    return s + np.random.default_rng(seed).normal(0, 0.01, n)


def _plant_value_carrier_series(n=128, n_ducks=16, carrier_idx=7, seed=7):
    """A one-duck value spike (deep V) on a smooth periodic board — the
    value-outlier plant.

    FALSIFICATION, named: the Phase 2 design expected this plant to yield
    VALUE-CARRIER. The calibration refuted the expectation: the WH-entropy
    meter is max-normalized (amplitude-cancelling) and translation-dominated,
    so a one-duck value feature is inaudible — the spike duck's responses sit
    at/below the measurement floor while far ducks' K_T responses dominate.
    test_p4 pins the falsified reality, not the design's hope."""
    t = np.linspace(0.0, 1.0, n)
    idx = np.unique(np.linspace(0, n - 1, n_ducks).astype(int))
    base = np.sin(2 * np.pi * 5 * t)
    base[idx[carrier_idx]] -= 1.0
    return base + np.random.default_rng(seed).normal(0, 0.01, n)


def _planted_noise(n=128, seed=11):
    return np.random.default_rng(seed).normal(0, 1.0, n)


# -- P0: the instruments are tied to the Phase 1 ear (classical, fast) ---------


def test_field_from_knots_matches_phase1_ear_on_unkicked_decomposition():
    """No drift between sensitivity's field builder and resonance's ear:
    the un-kicked decomposition must reproduce curvature_field() exactly."""
    dec = decompose(_plant_carrier_series())
    a = field_from_knots(dec.knots, dec.duck_values, dec.signal.size)
    b = curvature_field(dec.signal)
    assert np.allclose(a, b, atol=1e-12)


def test_noop_kick_is_a_classical_identity():
    """A δ=0 kick must rebuild the identical spline — the no-op's only
    randomness is measurement shot noise, which is what sigma_H measures."""
    dec = decompose(_plant_carrier_series())
    kicked = kick_duck(dec, 3, 0.0, operator="temporal")
    assert np.allclose(kicked.knots, dec.knots, atol=0)
    a = wh_spectrum(field_from_knots(kicked.knots, kicked.duck_values, dec.signal.size))
    b = wh_spectrum(field_from_knots(dec.knots, dec.duck_values, dec.signal.size))
    assert np.allclose(a, b, atol=1e-12)


# -- P1: the planted carrier must be named -------------------------------------


def test_planted_carrier_duck_is_named_with_margin_over_runner_up():
    """P1 positive control: the instrument must name duck 7 as THE carrier —
    a deaf meter returning DISTRIBUTED here fails the suite, and so does a
    meter that crowns the whole board (the floor-only live failure)."""
    dec = decompose(_plant_carrier_series(), n_ducks=16)
    rep = duck_sensitivity(dec, shots=1000, r_preps=6, seed=42)
    assert rep["verdict"]["verdict"] == "CARRIER", rep["verdict"]["reason"]
    assert rep["verdict"]["duck"] == 7
    margin = rep["verdict"]["runner_up_margin_bits"]
    assert margin >= CALIBRATED_CARRIER_MARGIN, (
        f"carrier margin {margin:.2f} < calibrated floor {CALIBRATED_CARRIER_MARGIN}"
    )


# -- P2: noise must not produce a carrier --------------------------------------


def test_planted_noise_never_yields_a_named_carrier():
    """P2 negative control: white noise is either refused up front (no
    structure to attribute) or honestly DISTRIBUTED — never a named duck."""
    dec = decompose(_planted_noise(), n_ducks=16)
    rep = duck_sensitivity(dec, shots=1000, r_preps=6, seed=42)
    v = rep["verdict"]["verdict"]
    assert v in ("REFUSED", "DISTRIBUTED"), v
    assert "CARRIER" not in v
    assert rep["verdict"]["reason"]  # a refusal with no reason is a silent drop
    # provenance of the refusal: past entropy must exceed the calibrated band
    if v == "REFUSED":
        assert rep["past_entropy"]["measured"] >= CALIBRATED_NOISE_ENTROPY_MIN


# -- P3: the floor is measured in-run and a broken instrument refuses ----------


def test_floor_is_measured_in_run_and_noop_stays_under_it():
    """P3a: the live report must carry a floor measured THIS run (sigma_H,
    sigma_E from >=3 self-preparations, no-op delta-H, k) and the no-op must
    sit under the 4-sigma health gate on the planted case."""
    dec = decompose(_plant_carrier_series(), n_ducks=16)
    rep = duck_sensitivity(dec, shots=1000, r_preps=6, seed=42)
    fl = rep["floor"]
    assert fl["r_preps"] >= 2 and fl["k"] > 0
    assert fl.get("e_preps", 0) >= 3, (
        "sigma_E from <3 self-swaps is noise-on-noise (live failure: 2 swaps "
        "-> sigma_E 0.0 -> echo gate fully vacuous)"
    )
    assert fl["sigma_H"] > 0, "sigma_H=0 means the floor was assumed, not measured"
    assert fl["sigma_E"] is not None, "sigma_E missing — echo floor assumed"
    assert fl["noop_dH"] <= _NOOP_HEALTH_K * fl["sigma_H"], (
        "no-op leaked on a healthy instrument"
    )


def test_noop_leak_self_refuses():
    """P3b (the vacuous-pass trap): if the no-op kick produces signal above
    the floor, the report must refuse ITSELF — named, not silent. Branch
    tested directly (no Aer): the pin dies if the branch doesn't exist."""
    records = {
        "temporal": {3: {0.5: {"dH": 1.4, "E": 0.2, "reach": {0: 0.1}},
                         -0.5: {"dH": 1.2, "E": 0.3, "reach": {0: 0.1}}}},
        "value": {3: {0.5: {"dH": 0.1, "E": 0.9, "reach": {}},
                      -0.5: {"dH": -0.1, "E": 0.9, "reach": {}}}},
    }
    rep = _assemble_report(
        past_entropy_measured=1.9,
        floor={"sigma_H": 0.05, "sigma_E": 0.01, "k": 2.0, "floor_H": 0.1,
               "floor_E": 0.02, "noop_dH": 0.5, "r_preps": 6},
        records=records,
        n_ducks=16,
        shots=1000,
        seed=42,
    )
    assert rep["verdict"]["verdict"] == "REFUSED"
    assert "no-op" in rep["verdict"]["reason"]


# -- P4: the value plant is heard honestly (a falsification pin) ---------------


def test_planted_value_spike_is_not_misheard_as_temporal_carrier():
    """P4, as falsified: the design expected VALUE-CARRIER on the one-duck
    value spike; calibration refuted it — the meter is max-normalized and
    translation-dominated, so the spike itself is quiet (the 4-delta sweep
    measures its K_T response at ~0.10 bits — above the 2σ shot floor but
    8× under the ~0.86 cohort bar; the earlier "deep under the floor" claim
    was a 2-delta artifact, named and corrected). What the pin holds the
    instrument to: (a) the spike duck stays far under the carrier bar — the
    board never resolves duck 7 as carrying anything; (b) the verdict does
    not name duck 7 as any kind of carrier; (c) both operators stay
    recorded — an instrument that drops K_V has conflated the operators by
    omission."""
    dec = decompose(_plant_value_carrier_series(), n_ducks=16)
    rep = duck_sensitivity(dec, shots=1000, r_preps=6, seed=42)
    v = rep["verdict"]
    # (a) the spike duck's best response is far under the cohort bar
    kt7 = rep["per_duck"][7]["temporal"]["max_abs_dH"]
    kv7 = rep["per_duck"][7]["value"]["max_abs_dH"]
    bars = rep["cohort_bars"]
    assert kt7 < bars["temporal"] and kv7 < bars["value"], (
        f"spike duck cleared a carrier bar (KT {kt7:.3f} vs {bars['temporal']:.3f}, "
        f"KV {kv7:.3f} vs {bars['value']:.3f}) — the value spike was heard "
        "as a carrier"
    )
    # (b) no verdict names duck 7 as any kind of carrier
    assert v["duck"] != 7, v["reason"]
    assert 7 not in (v["carriers"] or []), v["reason"]
    # (c) both operators recorded at the spike duck
    for op in ("temporal", "value"):
        assert "max_abs_dH" in rep["per_duck"][7][op]


# -- P5: phase-shuffled carrier sits behind the pre-gate ------------------------


def test_phase_shuffled_carrier_is_refused_before_kicks():
    """P5 (Phase 1's scar, inherited): shuffling scrambles the temporal
    geometry — entropy lands above the noise band, so the pre-gate refuses
    before any duck is kicked. Thresholds are not inherited silently: the
    refusal must quote the measured entropy."""
    series = _plant_carrier_series()
    shuffled = np.random.default_rng(13).permutation(series)
    dec = decompose(shuffled, n_ducks=16)
    rep = duck_sensitivity(dec, shots=1000, r_preps=6, seed=42)
    assert rep["verdict"]["verdict"] == "REFUSED"
    assert "no structure" in rep["verdict"]["reason"]
    assert rep["past_entropy"]["measured"] >= CALIBRATED_NOISE_ENTROPY_MIN
    assert rep["per_duck"] == {}, "kicks ran behind a REFUSED pre-gate — wasted spend"
