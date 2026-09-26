"""Duck-sensitivity receipts — which duck carries the regime?

Phase 2 of moth-waveform (design: cell work/moth-waveform-phase2.md,
2026-09-27). Phase 1 hears WHETHER the board sings; Phase 2 names WHICH duck
makes it sing, with a per-duck receipt that fails loudly if the instrument
goes deaf.

The two kick operators, never conflated:
    K_T (temporal): t_i → t_i + δ·Δ_i   — moves the duck in time (the thesis)
    K_V (value):    y_i → y_i + δ·σ_y   — moves the duck's height (the control)
If a duck answers both alike, the ear is hearing data perturbation, not time
geometry — the receipt says so (CARRIER-DUAL) instead of flattering the thesis.

Three instruments per (duck, δ, operator):
    ΔH  signed entropy shift of the measured WH spectrum (bits of 6). Sign is
        read: a kicked carrier usually RAISES entropy (scrambles phase); a
        duck whose kick LOWERS entropy was fighting the regime.
    E   echo overlap: measured swap test |⟨ψ_before|ψ_after⟩|² — how much of
        the state vector the kick moved. Resolves motion far below the
        entropy floor, so at these shot counts nearly every real kick
        confirms — it is reported per duck and kept as a veto, never the
        discriminator (live receipt 2026-09-27: a sigma_E from 2 self-swaps
        collapsed to 0.0 and made an E-threshold gate fully vacuous).
    r   reach: which WH bins moved beyond the measured per-bin noise floor.

Two bars separate a carrier from the board (both measured in-run), with
divided labor — a generous detector, then a specific tie-breaker:
    floor_H = k·σ_H        — did the kick do ANYTHING beyond shot noise?
    cohort bar             — did THIS duck do more than the others?
        median + COHORT_K·MAD of the per-duck max|ΔH| under the same
        operator, same run. Live receipt 2026-09-27: at 1000 shots every
        duck's kick is genuinely audible, so a floor-only bar crowned all
        16 ducks MULTI-CARRIER on the planted carrier ("ducks [0..15] all
        clear floor 0.07 bits"). The bar may over-admit; it only detects.
    tie test               — has the board resolved ONE duck?
        top-two carrier responses within TIE_K·σ_H of each other →
        MULTI-CARRIER (the set carries it); a gap beyond that → the single
        duck is named. Classical-screen receipt: at a 3·MAD-only bar the
        verdict hinged on a 0.001-bit coincidence; detector + tie-breaker
        makes each test do one job.

The floor is measured in-run, never assumed: R fresh preparations of the
unchanged state give σ_H and per-bin σ_p; R self-swaps give σ_E (fewer than
3 is noise-on-noise — the failure above); the full kick pipeline runs once
with δ=0 (the no-op) and must stay under the NOOP_K·σ_H health gate. The
health gate is wider than the kick floor deliberately: σ_H from 6 preps
carries ~5 dof and a 2σ health check false-refused a healthy instrument live
(noop_dH −0.121 vs floor 0.065 — also flipped P4 between suite and
standalone runs). A gate that passes without a live floor measurement passes
vacuously (fleet-murmur scar).

The reach vector has the shape of an otoc-echo tap profile (duck ↔ kick_site,
curvature bin ↔ tap depth), so a Phase 4 hardware run (otoc-echo-v1, 1
credit, Casey-gated) diffs cleanly against the classical influence matrix;
the receipt carries the mapping table.

ΔH references the CLASSICAL entropy of the un-kicked spectrum (deterministic,
replayable); σ_H carries the measurement noise. Receipt dicts keep native int
duck keys — json.dumps(receipt, default=str) is the caller's portability seam.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from .resonance import (
    _EAR_NOISE_MIN,
    _entropy,
    _measure_probs,
    _swap_overlap,
    wh_spectrum,
)
from .spline import SplineDecomposition

# the pre-gate is inherited, not re-derived: Phase 1 measured structure ≤3.0 /
# noise ≥3.5 of 6 bits (README calibration table); see CALIBRATED_NOISE_ENTROPY_MIN
_NOISE_GATE = _EAR_NOISE_MIN

# the kick floor factor (how far a response must clear shot noise to count)
KICK_K = 2.0
# the no-op HEALTH gate factor — wider than the kick floor on purpose (see
# module docstring: 5-dof sigma_H makes a 2σ health check a false-refuser)
NOOP_K = 4.0
# the cohort-outlier factor: a carrier's max|ΔH| must exceed
# median + COHORT_K·MAD of all ducks' responses under the same operator.
# Deliberately the SENSITIVE test: it may over-admit, because the tie test
# below discriminates (a generous detector + a specific tie-breaker beats
# one knife-edge gate trying to do both — classical screen receipt
# 2026-09-27: at 3·MAD the bar landed 0.001 under the runner-up).
COHORT_K = 2.5
# the tie width, in σ_H units: if the top two carrier responses differ by
# less than TIE_K·σ_H the board has NOT resolved a single carrier —
# MULTI-CARRIER names the set. This is the design's "within floor of each
# other" tie semantics, made noise-aware (the floor alone is a 2σ bar; the
# difference of two noisy responses carries √2·σ_H).
TIE_K = 2.0

# stable per-operator seed offsets — never hash(): str hashing is randomized
_OP_SEED = {"temporal": 0, "value": 5000}


def field_from_knots(
    tk: np.ndarray, vk: np.ndarray, n_samples: int, n_grid: int = 64
) -> np.ndarray:
    """Curvature field on a fixed grid from explicit knot geometry.

    Mirrors resonance.curvature_field() exactly when (tk, vk) are the ducks
    of decompose(series) — pinned by the P0 test so the two ears cannot drift.
    """
    t = np.linspace(0.0, 1.0, n_samples)
    cs = CubicSpline(tk, vk, bc_type="natural")
    curv = cs(t, 2)
    grid = np.linspace(0.0, 1.0, n_grid)
    field = CubicSpline(t, curv)(grid)
    return field / (np.max(np.abs(field)) + 1e-12)


def kick_duck(dec, i: int, delta: float, operator: str = "temporal"):
    """One duck kicked; the spline re-fit through the moved temporal variable.

    temporal: knot position moves by delta × local spacing (height held).
    value:    knot height moves by delta × series σ (position held).
    The knot is clamped a 5% gap short of its neighbors so the spline stays
    strictly monotone; at |δ| ≤ 0.5 the clamp never engages.
    δ=0 is a classical identity (pinned) — its only randomness is shot noise.
    """
    if operator not in ("temporal", "value"):
        raise ValueError(f"unknown operator '{operator}' (want temporal|value)")
    tk = np.array(dec.knots, dtype=float)
    vk = np.array(dec.duck_values, dtype=float)
    if not 0 <= i < len(tk):
        raise IndexError(f"duck {i} out of range 0..{len(tk) - 1}")
    if operator == "temporal":
        gaps = np.diff(tk)
        left = gaps[i - 1] if i > 0 else gaps[0]
        right = gaps[i] if i < len(gaps) else gaps[-1]
        spacing = 0.5 * (left + right)
        lo = tk[i - 1] + 0.05 * left if i > 0 else 0.0
        hi = tk[i + 1] - 0.05 * right if i < len(tk) - 1 else 1.0
        tk[i] = min(max(tk[i] + delta * spacing, lo), hi)
    else:
        vk[i] = vk[i] + delta * float(np.std(dec.signal))
    cs = CubicSpline(tk, vk, bc_type="natural")
    return SplineDecomposition(
        t=dec.t,
        signal=dec.signal,
        curvature=cs(dec.t, 2),
        knots=tk,
        duck_values=vk,
    )


def _hear_record(spec0, spec1, shots: int, seed: int, sigma_p: np.ndarray,
                 k: float) -> Dict[str, Any]:
    """All three instruments on one kicked state (ΔH vs the classical ref)."""
    h1 = _entropy(_measure_probs(spec1, shots, seed))
    d_h = h1 - _entropy(spec0)
    e = _swap_overlap(spec0, spec1, shots, seed + 1)
    moved = np.abs(spec1 - spec0) > k * sigma_p
    reach = {int(b): float(spec1[b] - spec0[b]) for b in np.where(moved)[0]}
    return {"dH": float(d_h), "E": float(e), "reach": reach}


def duck_sensitivity(
    dec,
    shots: int = 2000,
    deltas: Sequence[float] = (-0.5, -0.25, 0.25, 0.5),
    r_preps: int = 8,
    k: float = KICK_K,
    seed: int = 42,
    operators: Sequence[str] = ("temporal", "value"),
) -> Dict[str, Any]:
    """Per-duck sensitivity receipts: which duck carries the regime?

    Measures the floor first (R fresh preparations + R self-swaps + a
    full-pipeline δ=0 no-op); refuses itself if the instrument leaks. Every
    verdict names its reason. Returns a FINDING/v1-shaped receipt dict.
    """
    n_samples = int(dec.signal.size)
    field0 = field_from_knots(dec.knots, dec.duck_values, n_samples)
    spec0 = wh_spectrum(field0)

    # -- the floor, measured this run, in full, before any branch --
    preps = [_measure_probs(spec0, shots, seed + j) for j in range(r_preps)]
    hists = np.array([_entropy(p) for p in preps])
    h0 = float(np.mean(hists))
    sigma_h = float(np.std(hists, ddof=1)) if r_preps > 1 else 0.0
    sigma_p = (np.std(np.array(preps), axis=0, ddof=1)
               if r_preps > 1 else np.zeros_like(spec0))
    floor_h = k * sigma_h

    # echo floor: σ_E from R identical-state self-swaps (>=3 or it is
    # noise-on-noise: with 2 the estimate collapsed to 0.0 live and the echo
    # gate went fully vacuous)
    e_self = [_swap_overlap(spec0, spec0, shots, seed + 500 + j)
              for j in range(max(r_preps, 3))]
    sigma_e = float(np.std(e_self, ddof=1)) if len(e_self) > 1 else 0.0

    # no-op kick: the FULL pipeline at δ=0 (ΔH vs the classical ref)
    noop = kick_duck(dec, 0, 0.0, "temporal")
    noop_spec = wh_spectrum(field_from_knots(noop.knots, noop.duck_values, n_samples))
    noop_dh = _entropy(_measure_probs(noop_spec, shots, seed + 99)) - _entropy(spec0)

    floor = {
        "sigma_H": sigma_h, "sigma_E": sigma_e, "k": k, "floor_H": floor_h,
        "floor_E": k * sigma_e, "noop_dH": noop_dh,
        "noop_health_k": NOOP_K, "r_preps": r_preps, "e_preps": len(e_self),
    }
    common = dict(
        past_entropy={"measured": h0, "threshold": _NOISE_GATE},
        floor=floor,
        records={}, per_duck={}, n_ducks=len(dec.knots), shots=shots,
        seed=seed, deltas=list(deltas), operators=list(operators),
    )

    # -- pre-gate: nothing to attribute on an unstructured board --
    if h0 >= _NOISE_GATE:
        common["verdict"] = {
            "verdict": "REFUSED", "duck": None, "carriers": [],
            "runner_up_margin_bits": None,
            "reason": (f"curvature spectrum entropy {h0:.2f} bits ≥ "
                       f"{_NOISE_GATE} — no structure to attribute"),
        }
        return _receipt(**common)

    # -- no-op health gate (the receipt carries the no-op value either way) --
    if abs(noop_dh) > NOOP_K * sigma_h:
        common["verdict"] = {
            "verdict": "REFUSED", "duck": None, "carriers": [],
            "runner_up_margin_bits": None,
            "reason": (f"no-op kick produced signal {noop_dh:+.3f} bits > "
                       f"health gate {NOOP_K * sigma_h:.3f} — instrument broken"),
        }
        return _receipt(**common)

    # -- kick every duck under every operator --
    records: Dict[str, Dict[int, Dict[float, Dict[str, Any]]]] = {op: {} for op in operators}
    for op in operators:
        for i in range(len(dec.knots)):
            per_delta: Dict[float, Dict[str, Any]] = {}
            for delta in deltas:
                rec_seed = seed + 1000 * (i + 1) + _OP_SEED[op] + int(delta * 100)
                kicked = kick_duck(dec, i, delta, op)
                spec1 = wh_spectrum(
                    field_from_knots(kicked.knots, kicked.duck_values, n_samples))
                per_delta[delta] = _hear_record(spec0, spec1, shots, rec_seed,
                                                sigma_p, k)
            records[op][i] = per_delta

    return _assemble_report(
        past_entropy_measured=h0, floor=floor, records=records,
        n_ducks=len(dec.knots), shots=shots, seed=seed, deltas=deltas,
        operators=operators,
    )


def _asymmetry(recs: Dict[float, Dict[str, Any]]) -> Optional[float]:
    """A_i = |ΔH(+D) − ΔH(−D)| / max(|ΔH(+D)|, |ΔH(−D)|) at the widest δ pair.

    A_i ≈ 0: linear responder. A_i ≈ 1: cliff duck — one direction
    re-smooths, the other scrambles. Needs both extreme deltas present;
    returns None on a one-sided sweep (named, not faked).
    """
    neg = [d for d in recs if d < 0]
    pos = [d for d in recs if d > 0]
    if not neg or not pos:
        return None
    dp, dn = max(pos), max(neg, key=abs)
    a, b = recs[dp]["dH"], recs[dn]["dH"]
    denom = max(abs(a), abs(b))
    if denom == 0.0:
        return None
    return float(abs(a - b) / denom)


def _assemble_report(
    past_entropy_measured: float,
    floor: Dict[str, Any],
    records: Dict[str, Dict[int, Dict[float, Dict[str, Any]]]],
    n_ducks: int,
    shots: int,
    seed: int,
    deltas: Optional[Sequence[float]] = None,
    operators: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Verdict logic, isolated from measurement so the refusal branches are
    pin-testable without the simulator (P3b)."""
    floor_h = floor["floor_H"]
    floor_e = floor.get("floor_E") or 0.0
    ops = operators if operators is not None else list(records.keys())

    def refused(reason: str) -> Dict[str, Any]:
        return _receipt(
            verdict={"verdict": "REFUSED", "duck": None, "carriers": [],
                     "runner_up_margin_bits": None, "reason": reason},
            past_entropy={"measured": past_entropy_measured,
                          "threshold": _NOISE_GATE},
            floor=floor, records={}, per_duck={}, n_ducks=n_ducks,
            shots=shots, seed=seed, deltas=list(deltas or []),
            operators=list(ops),
        )

    if past_entropy_measured >= _NOISE_GATE:
        return refused(f"curvature spectrum entropy "
                       f"{past_entropy_measured:.2f} bits ≥ {_NOISE_GATE} — "
                       "no structure to attribute")
    if abs(floor["noop_dH"]) > NOOP_K * floor["sigma_H"]:
        return refused(f"no-op kick produced signal {floor['noop_dH']:+.3f} "
                       f"bits > health gate {NOOP_K * floor['sigma_H']:.3f} "
                       "— instrument broken")

    # -- per-duck summaries: max |ΔH| per operator, echo flag + asymmetry --
    per_duck: Dict[int, Dict[str, Any]] = {}
    for i in range(n_ducks):
        entry: Dict[str, Any] = {}
        clears: Dict[str, bool] = {}
        for op in ops:
            recs = records.get(op, {}).get(i, {})
            if not recs:
                entry[op] = {"max_abs_dH": 0.0, "dH_at_max": 0.0,
                             "E_at_max": None, "echo_confirms": False,
                             "reach_bins": 0, "asymmetry_A": None}
                clears[op] = False
                continue
            dmax = max(recs, key=lambda d: abs(recs[d]["dH"]))
            rec = recs[dmax]
            echo = rec["E"] < 1.0 - floor_e
            entry[op] = {"max_abs_dH": abs(rec["dH"]), "dH_at_max": rec["dH"],
                         "E_at_max": rec["E"], "echo_confirms": bool(echo),
                         "reach_bins": len(rec.get("reach", {})),
                         "asymmetry_A": _asymmetry(recs)}
            clears[op] = False  # filled after the cohort bars are known
        entry["_clears"] = clears
        per_duck[i] = entry

    # -- cohort bars (the DETECTOR): a carrier must out-kick the board, not
    #    just the noise. Generous on purpose — the tie test discriminates. --
    cohort_bars: Dict[str, float] = {}
    for op in ops:
        resp = np.array([per_duck[i][op]["max_abs_dH"] for i in per_duck])
        med = float(np.median(resp))
        mad = float(np.median(np.abs(resp - med)))
        # a degenerate cohort (MAD 0) still owes separation beyond shot noise
        bar = med + COHORT_K * mad if mad > 0 else med + floor_h
        cohort_bars[op] = float(max(bar, floor_h))
    for i in per_duck:
        for op in ops:
            e = per_duck[i][op]
            per_duck[i]["_clears"][op] = bool(
                e["max_abs_dH"] > floor_h
                and e["max_abs_dH"] > cohort_bars[op]
                and e["echo_confirms"]
            )

    carriers = [i for i in per_duck if any(per_duck[i]["_clears"].values())]

    def best_response(i: int) -> float:
        return max(per_duck[i]["temporal"]["max_abs_dH"],
                   per_duck[i]["value"]["max_abs_dH"]) if n_ducks else 0.0

    def dominant_op(i: int) -> str:
        m_t = per_duck[i]["temporal"]["max_abs_dH"]
        m_v = per_duck[i]["value"]["max_abs_dH"]
        if abs(m_t - m_v) <= floor_h:  # both answers alike → the honest tie
            return "dual"
        return "temporal" if m_t > m_v else "value"

    if not carriers:
        verdict = {"verdict": "DISTRIBUTED", "duck": None, "carriers": [],
                   "runner_up_margin_bits": None,
                   "reason": (f"no duck out-kicked its cohort (bars "
                              f"{ {op: round(b, 3) for op, b in cohort_bars.items()} }, "
                              f"floor {floor_h:.3f} bits) — the regime is "
                              "spread across the board")}
    else:
        # -- the TIE TEST (the discriminator): has the board resolved ONE
        #    duck, or is the top two within noise of each other? --
        order = sorted(carriers, key=best_response, reverse=True)
        top_i = order[0]
        top = best_response(top_i)
        second = best_response(order[1]) if len(order) > 1 else max(
            (best_response(j) for j in per_duck if j != top_i), default=0.0)
        tie_width = TIE_K * floor["sigma_H"]
        runner = max((best_response(j) for j in per_duck if j != top_i),
                     default=0.0)
        margin = top - runner
        sec_note = ""
        if len(order) > 1:
            sec_note = (f" secondary outliers {order[1:]} ride the bar but "
                        f"trail the top by {top - second:.2f} bits")
        if top - second <= tie_width and len(order) > 1:
            names = ", ".join(str(i) for i in sorted(order))
            margins = {str(i): round(best_response(i), 3) for i in sorted(order)}
            verdict = {"verdict": "MULTI-CARRIER", "duck": None,
                       "carriers": sorted(order),
                       "runner_up_margin_bits": None,
                       "reason": (f"ducks [{names}] all out-kick their cohort "
                                  f"and the top two sit within {tie_width:.3f} "
                                  f"bits ({TIE_K}σ_H) of each other "
                                  f"(max|ΔH| {margins}) — regime carried by a "
                                  "set, not one duck")}
        else:
            dom = dominant_op(top_i)
            if dom == "dual":
                verdict = {"verdict": "CARRIER-DUAL", "duck": top_i,
                           "carriers": sorted(order),
                           "runner_up_margin_bits": margin,
                           "reason": (f"duck {top_i} leads its cohort but its "
                                      f"temporal ({per_duck[top_i]['temporal']['dH_at_max']:+.2f})"
                                      f" and value ({per_duck[top_i]['value']['dH_at_max']:+.2f})"
                                      " responses are indistinguishable — the ear "
                                      "hears data perturbation, not time geometry"
                                      + sec_note)}
            elif dom == "value":
                verdict = {"verdict": "VALUE-CARRIER", "duck": top_i,
                           "carriers": sorted(order),
                           "runner_up_margin_bits": margin,
                           "reason": (f"duck {top_i} is carried by its VALUE (K_V "
                                      f"{per_duck[top_i]['value']['dH_at_max']:+.2f} vs "
                                      f"K_T {per_duck[top_i]['temporal']['dH_at_max']:+.2f} "
                                      f"bits, floor {floor_h:.2f}) — a data outlier, "
                                      "not a temporal variable" + sec_note)}
            else:
                verdict = {"verdict": "CARRIER", "duck": top_i,
                           "carriers": sorted(order),
                           "runner_up_margin_bits": margin,
                           "reason": (f"duck {top_i} carries the regime: K_T "
                                      f"{per_duck[top_i]['temporal']['dH_at_max']:+.2f} "
                                      f"bits (K_V {per_duck[top_i]['value']['dH_at_max']:+.2f},"
                                      f" floor {floor_h:.2f}, cohort bar "
                                      f"{cohort_bars['temporal']:.2f}, runner-up "
                                      f"{runner:.2f}, tie width {tie_width:.2f})"
                                      " — the temporal variable is load-bearing"
                                      + sec_note)}

    return _receipt(
        verdict=verdict,
        past_entropy={"measured": past_entropy_measured,
                      "threshold": _NOISE_GATE},
        floor=floor, records=records, per_duck=per_duck,
        n_ducks=n_ducks, shots=shots, seed=seed,
        deltas=list(deltas or []), operators=list(ops),
        cohort_bars=cohort_bars,
    )


# Calibration provenance: these constants quote the planted-protocol
# calibration run (tests/calibrate_sensitivity.py, seeds 42, 1000 shots,
# r_preps 6) with a named safety fraction — the Phase 1 pattern (structure
# ≤3.0 / noise ≥3.5 came from the same discipline).
# Measured 2026-09-27, notch plant depth 1.2: P1 named duck 7 CARRIER with
# margin 0.317 bits (0.981 over runner 0.664); P2 refused at 4.20 bits;
# P5 refused at 3.76 bits; P4 left the spike duck unnamed (KT 0.122 / KV
# 0.124 vs temporal bar 0.66). Margin bar keeps 25% of the measured gap;
# the noise band sits 0.25 under the gate, inside the 0.44-bit
# structured-vs-shuffled separation this plant actually shows.
CALIBRATED_CARRIER_MARGIN = 0.079
_CALIBRATION_SAFETY = 0.25  # keep this fraction of the measured gap as margin
CALIBRATED_NOISE_ENTROPY_MIN = _NOISE_GATE - 0.25


def _receipt(**kw) -> Dict[str, Any]:
    """FINDING/v1-shaped receipt: every number the verdict rests on."""
    return {
        "subject": "moth_waveform.duck_sensitivity",
        "verdict": kw["verdict"],
        "past_entropy": kw["past_entropy"],
        "floor": kw["floor"],
        "per_duck": kw.get("per_duck", {}),
        "cohort_bars": kw.get("cohort_bars", {}),
        "params": {"n_ducks": kw["n_ducks"], "shots": kw["shots"],
                   "seed": kw["seed"], "deltas": kw["deltas"],
                   "operators": kw["operators"]},
        "otoc_bridge": {
            "note": ("the reach vector is the classical cousin of an "
                     "otoc-echo tap profile; a Phase 4 hardware run "
                     "(otoc-echo-v1, 1 credit, Casey-gated) diffs against it"),
            "duck_i": "kick_site",
            "curvature_bin_t": "tap depth t",
            "delta_sweep": "kick amplitude",
            "asymmetry_A": "disorder sensitivity",
            "one_job_reads_all_taps": True,
        },
        "calibration": ("resonance.py Phase 1 table (README): structure ≤3.0, "
                        "noise ≥3.5 of 6 bits; carrier margin from "
                        "tests/calibrate_sensitivity.py"),
    }
