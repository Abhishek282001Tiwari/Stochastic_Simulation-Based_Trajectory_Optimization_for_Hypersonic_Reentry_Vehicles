"""Hypersonic Reentry Vehicle Trajectory Optimization Package.

This package provides tools for stochastic simulation-based trajectory optimization
of hypersonic reentry vehicles with uncertainty quantification.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .dynamics import VehicleDynamics
from .atmosphere import AtmosphereModel
from .optimization import TrajectoryOptimizer
from .uncertainty import UncertaintyQuantifier
from .visualization import PlotManager
from .control import ControlSystem

__all__ = [
    "VehicleDynamics",
    "AtmosphereModel", 
    "TrajectoryOptimizer",
    "UncertaintyQuantifier",
    "PlotManager",
    "ControlSystem",
]