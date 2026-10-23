"""Lattice path generation and result types."""
from . import parameters
from .structures import CandidatePath, LatticePlanningResult, PolynomialCoefficients

__all__ = ["CandidatePath", "LatticePlanningProcess", "LatticePlanningResult",
           "PolynomialCoefficients", "parameters"]


def __getattr__(name):
    if name == "LatticePlanningProcess":
        from ..section_planning import LatticePlanningProcess
        return LatticePlanningProcess
    raise AttributeError(name)
