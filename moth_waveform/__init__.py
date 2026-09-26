"""moth-waveform — quantum resonance hearing for spline-tension extrapolation."""

from .spline import SplineDecomposition, candidate_futures, decompose
from .resonance import curvature_entropy, judge_extrapolation, resonance_gate
from .sensitivity import duck_sensitivity, kick_duck

__all__ = [
    "SplineDecomposition",
    "decompose",
    "candidate_futures",
    "curvature_entropy",
    "resonance_gate",
    "judge_extrapolation",
    "duck_sensitivity",
    "kick_duck",
]
__version__ = "0.2.0"
