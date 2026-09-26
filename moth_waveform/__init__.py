"""moth-waveform — quantum resonance hearing for spline-tension extrapolation."""

from .spline import SplineDecomposition, candidate_futures, decompose
from .resonance import curvature_entropy, judge_extrapolation, resonance_gate

__all__ = [
    "SplineDecomposition",
    "decompose",
    "candidate_futures",
    "curvature_entropy",
    "resonance_gate",
    "judge_extrapolation",
]
__version__ = "0.1.0"
