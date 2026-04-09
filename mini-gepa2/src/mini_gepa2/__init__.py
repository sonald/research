"""Teaching-oriented GEPA implementation."""

from .adapter import Adapter, Candidate, EvaluationBatch, ReflectiveRecord
from .api import MiniGEPAConfig, optimize
from .state import MiniGEPAState

__all__ = [
    "Adapter",
    "Candidate",
    "EvaluationBatch",
    "MiniGEPAConfig",
    "MiniGEPAState",
    "ReflectiveRecord",
    "optimize",
]
