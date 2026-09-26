# moth-waveform

**Quantum resonance hearing for spline-tension waveform extrapolation.**
*Temporal variables as first-class citizens of distinction — not reference points.*

## The idea, precisely

A sampled series treats time as an index: t is where a value sits. A tension
spline treats time as geometry: the knots (the ducks, in drafting-shop slang)
*are* the temporal variables, and moving one duck reshapes the curve's local
temporal neighborhood. The spline's second derivative — the curvature field —
is where the data forces hard bends against the board and where it runs free.
That field is a **waveform in its own right**: pluck the interpolant and its
spectrum is the resonance of the tension on the soundboard.

moth-waveform makes that precise and puts a quantum ear on it:

1. **Decompose** a series into signal + curvature field (`spline.py`).
   Time enters as knot geometry; nothing interesting happens at sample indices.
2. **Hear the board** (`resonance.py`): the Walsh–Hadamard power spectrum of
   the curvature field is prepared as quantum probability amplitudes on the
   quantum-audio substrate and measured. Periodic regime structure
   concentrates the spectrum (low entropy); structureless data spreads it.
3. **Judge continuations by resonance** (predictive engining): each candidate
   future extends the curve; the curvature spectrum of the extended region is
   compared to the actual continuation's spectrum with a **measured swap
   test** — `overlap = 2·P(ancilla=0) − 1 = |⟨ψ|φ⟩|²`. Candidates never judge
   themselves; the board does. No resonance above threshold → **REFUSED**
   with a named reason, never a guess.

## Division of labor, named honestly

| Stage | Where it runs | Why |
|---|---|---|
| Tension spline, curvature, WH spectrum | numpy/scipy (classical) | Exact geometry; no substrate noise |
| Spectrum-as-state preparation + entropy readout | quantumaudio QPAM API + qiskit Aer | The substrate carries the spectrum as a waveform; measurement statistics ARE the receipt |
| Candidate-vs-continuation similarity | measured swap test (quantum protocol, simulable at 6 qubits) | Quantum-native judging: interference, not dot product; protocol runs unchanged on real QPUs |

Two substrate scars from quilt-doctor are load-bearing here:
- **QPAM's `encode()` answers a different question.** It offsets data by
  `(x+1)/2` before normalization — right for audio round-trips, fatal for
  spectrum concentration. We use the scheme's own `value_setting()` seam to
  prepare exact amplitudes. Named, not silent.
- **Fidelity is lossless for everything; discrimination lives in basis
  choice.** The ear reads spectrum entropy, not round-trip fidelity.

## Calibration receipt (the planted protocol)

Measured on AerSimulator, 2000 shots, seeds 42/1/13 — the meter earns trust
on planted truth before it touches fleet data:

| Planted input | Curvature spectrum entropy (of 6 bits) | Verdict |
|---|---|---|
| Periodic regime structure (period 32 + slow envelope) | **1.98** | structure heard |
| Same series, phase-shuffled | **4.54** | geometry destroyed |
| White noise through the spline | **3.98** | no tension to hear |

Margins are >2 bits — thresholds (`_EAR_STRUCTURE_MAX = 3.0`,
`_EAR_NOISE_MIN = 3.5`) sit one-sided inside the gap. An acoustic fact the
pins deliberately do NOT encode: shuffled structure strains the board *more*
than smooth spline noise (scrambled ducks force sharper bends), so the
shuffled-vs-noise ordering is documented, not pinned.

Swap test: same-regime spectra overlap **0.967**; regime-vs-noise **0.101**
(ideal 0.968 / 0.071 — measurement noise is part of the receipt).

Gate sanity: on a held-out planted regime, resonance ranking 0.486 / 0.947 /
0.953 tracks held-out MSE 69.5 / 0.29 / 0.25 — the ear is predictive, not
decorative.

## Phase 2 — which duck carries the regime? (`sensitivity.py`)

Phase 1 hears *whether* the board sings; Phase 2 names *which duck* makes it
sing. Two kick operators, never conflated: **K_T** moves a duck *in time*
(`t_i → t_i + δ·Δ_i` — the thesis) and **K_V** moves its *value*
(`y_i → y_i + δ·σ_y` — the control). If a duck answers both alike, the
receipt says CARRIER-DUAL instead of flattering the thesis.

Three instruments per (duck, δ): signed entropy shift **ΔH**, measured echo
overlap **E** (swap test — a veto and a report, not the discriminator), and
the **reach** vector of WH bins that moved. The reach vector has the shape
of an `otoc-echo-v1` tap profile (duck ↔ kick_site, curvature bin ↔ tap
depth), so the Phase 4 hardware run diffs cleanly against the classical
influence matrix — the mapping table ships in every receipt.

**Verdict discipline — every bar measured in-run, every test doing one job:**
the shot floor `k·σ_H` asks "did the kick do anything" (a floor-only bar
once crowned all 16 ducks MULTI-CARRIER — at 1000 shots every kick is
audible); the cohort bar `median + 2.5·MAD` asks "did this duck do more
than the others" (the sensitive detector); the tie test asks "has the board
resolved ONE duck" (top-two within `2·σ_H` → MULTI-CARRIER, else the duck
is named). The no-op health gate runs at `4·σ_H` — a 2σ check on a 5-dof
σ̂ false-refused a healthy instrument live. The no-op value and σ_E (from ≥3
self-swaps; 2 collapsed to 0.0 and made an echo gate vacuous) ride in every
receipt.

Calibration receipt (AerSimulator, 1000 shots, seeds 42, r_preps 6 —
`tests/calibrate_sensitivity.py`, same run that sets the pins' constants):

| Planted truth | Verdict | Receipt |
|---|---|---|
| Sharp notch pinned to knot 7 of 16 (temporal carrier) | **CARRIER duck 7** | K_T +0.98 vs runner 0.66, margin 0.32 bits; knot 7 a cliff duck (A = 1.0) |
| White noise | REFUSED | 4.20 bits ≥ 3.5 — no structure to attribute |
| One-duck value spike | CARRIER duck 2, spike unnamed | the spike duck stays 5× under the cohort bar — the design's VALUE-CARRIER hope was **falsified by calibration** and the pin pins the falsified reality |
| Phase-shuffled carrier | REFUSED | 3.76 bits — geometry destroyed, kicks never ran |

Three carrier plants died by classical screen before the notch survived —
plateau+spike (attribution smeared onto the plateau's right edge: knot 8
out-kicked knot 7), twin bumps (0.11-bit margin over four ducks),
dip-dominated boards (blew the pre-gate or made knot 8 the top responder).
Planted what survives, named what died.

## Usage

```python
from moth_waveform import decompose, candidate_futures, judge_extrapolation

dec = decompose(past_series)                  # ducks → curvature field
cands = candidate_futures(dec, horizon=32)    # natural | drift | revert
verdict = judge_extrapolation(dec, cands, actual_continuation, shots=2000)

# verdict = {"verdict": "CONFIRMED"|"REFUSED", "winner": name|None,
#            "resonances": {name: overlap}, "past_entropy": bits,
#            "reason": "a sentence a skeptic can check"}
```

## Receipts doctrine

Every refusal names its reason. Every threshold names its calibration. The
planted protocol must REFUTE planted noise before any fleet series is heard.
This repo exists because claims without receipts are the disease — see
pong-quilt (honesty pins), hermit (quilt-WAL), quilt-doctor (moth-ledger
trial balance), quality-gate-stream (REFUSAL receipts), fleet-murmur
(transport modes).

## Roadmap (predictive engining, in order)

1. **Phase 1 (on main):** ear + swap-test judge + REFUSAL receipts on
   planted protocols.
2. **Phase 2 (this build):** duck-sensitivity receipts — move one duck and
   hear how the spectrum responds; which duck carries the regime
   (`sensitivity.py`, `tests/test_sensitivity.py`).
3. **Phase 3:** fleet series walk-in: jev commit cadence, tidepool memory
   ocean pressure, PLATO room churn — every hearing ships a FINDING/v1
   receipt (quilt-doctor format).
4. **Phase 4:** QPU run — same protocol on real hardware; the receipt then
   means something no simulator can fake. One `otoc-echo-v1` job reads all
   taps at once (Casey-gated, 1 credit).

## Tests

14 FAIL-first pins. Phase 1 (`tests/test_resonance.py`, 6): the curvature
field is first-class; planted periodic concentrates; planted noise refutes;
phase-shuffle destroys; the gate beats its drift baseline on planted regimes;
pure noise gets a named REFUSAL. Phase 2 (`tests/test_sensitivity.py`, 8):
the sensitivity ear is tied to the Phase 1 ear (P0); the planted carrier is
named with margin (P1); noise never yields a carrier (P2); the floor is
measured in-run, σ_E from ≥3 self-swaps, and a no-op leak refuses itself
(P3a/b — the vacuous-pass trap); the value spike is not misheard as a
temporal carrier (P4, a falsification pin); the phase-shuffled carrier is
refused before any kick runs (P5).
`python3 -m pytest tests/ -q`.

---

*They spline the ducks. The board tells you where it's forced. The ear hears
whether your continuation is singing with the tension or against it.*
