"""Tension-spline decomposition of a time series.

The temporal-variables move: a sample series is indexed by time (time as
reference point). A tension spline re-parameterizes the same curve in
knot/control-point space — the *ducks*. Moving one duck reshapes the curve's
local temporal geometry; the second derivative of the spline (the curvature
field) is where the data forces hard bends vs free runs.

moth-waveform treats that curvature field as a first-class waveform: it is
the "sound" of the tension on the board, and it — not the raw samples — is
what gets quantum-heard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class SplineDecomposition:
    """A series decomposed into signal, tension, and curvature fields."""

    t: np.ndarray            # normalized time grid [0, 1]
    signal: np.ndarray       # the series on that grid
    curvature: np.ndarray    # second derivative — the tension field (the sound)
    knots: np.ndarray        # duck positions (the temporal variables)
    duck_values: np.ndarray  # values at the ducks

    @property
    def tension_energy(self) -> np.ndarray:
        """Local bending energy density (curvature²) — where the board strains."""
        return self.curvature ** 2


def decompose(
    series: Sequence[float],
    n_ducks: Optional[int] = None,
) -> SplineDecomposition:
    """Fit a natural (minimum-curvature) tension spline through the series.

    The knots ARE the temporal variables of distinction: prediction and
    resonance operate on the geometry they induce, not on sample indices.
    """
    y = np.asarray(series, dtype=float)
    if y.ndim != 1 or y.size < 4:
        raise ValueError("series must be 1-D with at least 4 points")
    n = y.size
    if n_ducks is None:
        n_ducks = min(n, max(8, n // 4))
    idx = np.linspace(0, n - 1, n_ducks).astype(int)
    idx = np.unique(idx)
    t = np.linspace(0.0, 1.0, n)
    tk = t[idx]
    cs = CubicSpline(tk, y[idx], bc_type="natural")
    return SplineDecomposition(
        t=t,
        signal=y,
        curvature=cs(t, 2),
        knots=tk,
        duck_values=y[idx],
    )


def candidate_futures(
    dec: SplineDecomposition,
    horizon: int,
) -> dict:
    """Extrapolation candidates, each a claim about the temporal geometry.

    Returns a dict name -> np.ndarray of length `horizon`. The resonance gate
    (resonance.py) judges which candidate's curvature spectrum the real
    continuation resonates with — the candidates never judge themselves.
    """
    n = dec.signal.size
    t = dec.t
    y = dec.signal
    future_t = 1.0 + (np.arange(1, horizon + 1) / n)

    # natural extension: keep fitting through the last ducks, extend the spline
    idx = np.unique(np.linspace(0, n - 1, len(dec.knots)).astype(int))
    cs = CubicSpline(t[idx], y[idx], bc_type="natural")
    natural = cs(future_t)

    # linear drift: slope of the differenced tail (JEPA-cadence baseline)
    diffs = np.diff(y[-max(4, n // 8):])
    drift = np.cumsum(np.full(horizon, diffs.mean())) + y[-1]

    # mean revert: pull toward the duck-window mean
    mean = float(np.mean(y))
    revert = y[-1] + (mean - y[-1]) * (1.0 - np.exp(-np.arange(1, horizon + 1) / max(2.0, horizon / 4.0)))

    return {"natural": natural, "drift": drift, "revert": revert}
