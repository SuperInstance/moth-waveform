"""Calibration run for the duck-sensitivity instrument (Phase 2).

Planted truth first — the same discipline as Phase 1's table in the README.
Run directly:  python3 tests/calibrate_sensitivity.py
Its measured numbers set CALIBRATED_* in moth_waveform/sensitivity.py and the
pins' provenance in tests/test_sensitivity.py. Seeds/shots match the pins.

What this run measures and quotes:
- P1 planted carrier: verdict, margin over runner-up, floor (sigma_H, sigma_E,
  no-op), cohort bars, per-duck K_T/K_V responses + asymmetry A_i.
- P2 planted noise: verdict + provenance of the refusal.
- P4 planted value spike: the falsified design's honest outcome.
- P5 phase-shuffled carrier: pre-gate refusal provenance.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from moth_waveform.spline import decompose
from moth_waveform.sensitivity import (
    COHORT_K,
    NOOP_K,
    _CALIBRATION_SAFETY,
    duck_sensitivity,
)

from test_sensitivity import (
    _plant_carrier_series,
    _plant_value_carrier_series,
    _planted_noise,
)


def main() -> None:
    shots, r_preps, seed = 1000, 6, 42

    print(f"== P1: planted temporal carrier (notch pinned to knot 7 of 16) "
          f"[kick k=2.0, noop health k={NOOP_K}, cohort k={COHORT_K}] ==")
    dec = decompose(_plant_carrier_series(), n_ducks=16)
    rep = duck_sensitivity(dec, shots=shots, r_preps=r_preps, seed=seed)
    v = rep["verdict"]
    print(f"verdict={v['verdict']} duck={v['duck']} "
          f"margin={v['runner_up_margin_bits']}")
    print(f"past_entropy={rep['past_entropy']['measured']:.3f}")
    print(f"floor={rep['floor']}")
    print(f"cohort_bars={rep['cohort_bars']}")
    ranked = sorted(
        ((max(rep['per_duck'][i]['temporal']['max_abs_dH'],
              rep['per_duck'][i]['value']['max_abs_dH']), i)
         for i in rep['per_duck']),
        reverse=True)
    print("ranked max|dH|: " + ", ".join(f"d{i}={m:.3f}" for m, i in ranked[:6]))
    if v["duck"] is not None and v["runner_up_margin_bits"] is not None:
        gap = v["runner_up_margin_bits"]
        print(f"suggested CALIBRATED_CARRIER_MARGIN = "
              f"{_CALIBRATION_SAFETY * gap:.3f}  (gap {gap:.3f} x safety "
              f"{_CALIBRATION_SAFETY})")
    for i in sorted(rep["per_duck"]):
        e = rep["per_duck"][i]
        kt, kv = e["temporal"], e["value"]
        at = kt["asymmetry_A"]
        print(f"  duck {i:2d}: KT {kt['dH_at_max']:+.3f} "
              f"(max|{kt['max_abs_dH']:.3f}|, E {kt['E_at_max']}, "
              f"A {at if at is None else round(at, 2)}) | "
              f"KV {kv['dH_at_max']:+.3f} (max|{kv['max_abs_dH']:.3f}|, "
              f"E {kv['E_at_max']}) | clears={e['_clears']}")

    print("\n== P2: planted noise ==")
    dec = decompose(_planted_noise(), n_ducks=16)
    rep = duck_sensitivity(dec, shots=shots, r_preps=r_preps, seed=seed)
    print(f"verdict={rep['verdict']['verdict']} "
          f"past_entropy={rep['past_entropy']['measured']:.3f}")
    print(f"reason: {rep['verdict']['reason']}")

    print("\n== P4: planted value carrier (one-duck value spike) ==")
    dec = decompose(_plant_value_carrier_series(), n_ducks=16)
    rep = duck_sensitivity(dec, shots=shots, r_preps=r_preps, seed=seed)
    v = rep["verdict"]
    print(f"verdict={v['verdict']} duck={v['duck']} carriers={v['carriers']}")
    print(f"reason: {v['reason']}")
    print(f"past_entropy={rep['past_entropy']['measured']:.3f} "
          f"floor_H={rep['floor']['floor_H']:.3f}")
    kt7 = rep["per_duck"][7]["temporal"]
    print(f"spike duck 7: KT max|{kt7['max_abs_dH']:.3f}| "
          f"KV max|{rep['per_duck'][7]['value']['max_abs_dH']:.3f}|")

    print("\n== P5: phase-shuffled carrier ==")
    shuffled = np.random.default_rng(13).permutation(_plant_carrier_series())
    dec = decompose(shuffled, n_ducks=16)
    rep = duck_sensitivity(dec, shots=shots, r_preps=r_preps, seed=seed)
    print(f"verdict={rep['verdict']['verdict']} "
          f"past_entropy={rep['past_entropy']['measured']:.3f}")
    print(f"reason: {rep['verdict']['reason']}")


if __name__ == "__main__":
    main()
