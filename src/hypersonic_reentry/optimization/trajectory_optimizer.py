"""Base trajectory optimizer for hypersonic reentry vehicles.

This module provides the base class for trajectory optimization including
objective function definition, constraint handling, and optimization result
management.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
import time

from ..dynamics.vehicle_dynamics import VehicleDynamics, VehicleState
from ..utils.constants import DEG_TO_RAD


@dataclass
class OptimizationConstraint:
    """Definition of an optimization constraint."""
    
    name: str
    constraint_type: str  # 'equality', 'inequality'
    target_value: float
    tolerance: float = 1e-6
    weight: float = 1.0
    description: str = ""


@dataclass  
class OptimizationObjective:
    """Definition of optimization objective."""
    
    name: str
    objective_type: str  # 'minimize', 'maximize'
    weight: float = 1.0
    description: str = ""


@dataclass
class OptimizationResult:
    """Results from trajectory optimization."""
    
    # Optimization status
    success: bool = False
    message: str = ""
    num_iterations: int = 0
    computation_time: float = 0.0
    
    # Optimal solution
    optimal_controls: Optional[Dict[str, np.ndarray]] = None
    optimal_trajectory: Optional[Dict[str, np.ndarray]] = None
    optimal_performance: Optional[Dict[str, float]] = None
    
    # Objective function and constraints
    final_objective_value: float = np.inf
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    
    # Optimization history
    objective_history: List[float] = field(default_factory=list)
    constraint_history: List[Dict[str, float]] = field(default_factory=list)


class TrajectoryOptimizer(ABC):
    """Abstract base class for trajectory optimization algorithms.
    
    Provides common interface for different optimization methods including
    gradient-based, evolutionary, and hybrid approaches.
    """
    
    def __init__(self, 
                 vehicle_dynamics: VehicleDynamics,
                 objectives: List[OptimizationObjective],
                 constraints: List[OptimizationConstraint],
                 control_bounds: Dict[str, Tuple[float, float]]):
        """Initialize trajectory optimizer.
        
        Args:
            vehicle_dynamics: Vehicle dynamics model
            objectives: List of optimization objectives
            constraints: List of optimization constraints
            control_bounds: Bounds for control variables
        """
        self.logger = logging.getLogger(__name__)
        self.vehicle_dynamics = vehicle_dynamics
        self.objectives = objectives
        self.constraints = constraints
        self.control_bounds = control_bounds
        
        # Optimization settings
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.constraint_tolerance = 1e-4
        
        # Results storage
        self.optimization_history = []
        
        self.logger.info(f"Initialized trajectory optimizer with {len(objectives)} objectives and {len(constraints)} constraints")
    
    @abstractmethod
    def optimize(self, 
                initial_state: VehicleState,
                time_span: Tuple[float, float],
                initial_guess: Optional[Dict[str, np.ndarray]] = None) -> OptimizationResult:
        """Run trajectory optimization.
        
        Args:
            initial_state: Initial vehicle state
            time_span: Time span for optimization
            initial_guess: Initial guess for control variables
            
        Returns:
            OptimizationResult containing optimal solution
        """
        pass
    
    def evaluate_objective(self, 
                          controls: Dict[str, np.ndarray],
                          initial_state: VehicleState,
                          time_span: Tuple[float, float]) -> float:
        """Evaluate objective function for given control history.
        
        Args:
            controls: Control variable history
            initial_state: Initial vehicle state
            time_span: Time span for simulation
            
        Returns:
            Objective function value
        """
        try:
            # Simulate trajectory with given controls
            trajectory = self.vehicle_dynamics.integrate_trajectory(
                initial_state, time_span, controls
            )
            
            # Calculate performance metrics
            performance = self.vehicle_dynamics.calculate_performance_metrics(trajectory)
            
            # Evaluate objectives
            total_objective = 0.0
            
            for objective in self.objectives:
                if objective.name in performance:
                    value = performance[objective.name]
                    
                    if objective.objective_type == 'minimize':
                        total_objective += objective.weight * value
                    elif objective.objective_type == 'maximize':
                        total_objective -= objective.weight * value
            
            return total_objective
        
        except Exception as e:
            self.logger.warning(f"Objective evaluation failed: {str(e)}")
            return np.inf
    
    def evaluate_constraints(self, 
                           controls: Dict[str, np.ndarray],
                           initial_state: VehicleState,
                           time_span: Tuple[float, float]) -> Dict[str, float]:
        """Evaluate constraint violations for given control history.
        
        Args:
            controls: Control variable history
            initial_state: Initial vehicle state  
            time_span: Time span for simulation
            
        Returns:
            Dictionary of constraint violations
        """
        violations = {}
        
        try:
            # Simulate trajectory
            trajectory = self.vehicle_dynamics.integrate_trajectory(
                initial_state, time_span, controls
            )
            
            # Calculate performance metrics
            performance = self.vehicle_dynamics.calculate_performance_metrics(trajectory)
            
            # Evaluate constraints
            for constraint in self.constraints:
                if constraint.name in performance:
                    actual_value = performance[constraint.name]
                    target_value = constraint.target_value
                    
                    if constraint.constraint_type == 'equality':
                        violations[constraint.name] = abs(actual_value - target_value)
                    elif constraint.constraint_type == 'inequality':
                        # Inequality constraint: actual_value <= target_value
                        violations[constraint.name] = max(0, actual_value - target_value)
                
                # Path constraints (evaluated along trajectory)
                elif constraint.name in trajectory:
                    trajectory_values = trajectory[constraint.name]
                    
                    if constraint.constraint_type == 'path_max':
                        max_violation = np.max(trajectory_values) - constraint.target_value
                        violations[constraint.name] = max(0, max_violation)
                    elif constraint.constraint_type == 'path_min':
                        min_violation = constraint.target_value - np.min(trajectory_values)  
                        violations[constraint.name] = max(0, min_violation)
        
        except Exception as e:
            self.logger.warning(f"Constraint evaluation failed: {str(e)}")
            # Return large violations for failed cases
            for constraint in self.constraints:
                violations[constraint.name] = 1e6
        
        return violations
    
    def generate_initial_guess(self, 
                              initial_state: VehicleState,
                              time_span: Tuple[float, float],
                              num_control_points: int = 50) -> Dict[str, np.ndarray]:
        """Generate initial guess for control variables.
        
        Args:
            initial_state: Initial vehicle state
            time_span: Time span for optimization
            num_control_points: Number of control points
            
        Returns:
            Dictionary containing initial control guess
        """
        t_start, t_end = time_span
        time_points = np.linspace(t_start, t_end, num_control_points)
        
        # Simple initial guess based on nominal reentry profile
        controls = {}
        
        # Bank angle: start at 0, increase to manage lift
        bank_angle_profile = np.zeros(num_control_points)
        # Gradually increase bank angle for range extension
        for i in range(num_control_points):
            progress = i / (num_control_points - 1)
            bank_angle_profile[i] = 30.0 * DEG_TO_RAD * np.sin(np.pi * progress)
        
        controls['bank_angle'] = bank_angle_profile
        
        # Angle of attack: small positive value for lift generation
        angle_of_attack_profile = np.full(num_control_points, 5.0 * DEG_TO_RAD)
        controls['angle_of_attack'] = angle_of_attack_profile
        
        return controls
    
    def apply_control_bounds(self, controls: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply bounds to control variables.
        
        Args:
            controls: Control variable arrays
            
        Returns:
            Bounded control variables
        """
        bounded_controls = {}
        
        for control_name, control_values in controls.items():
            if control_name in self.control_bounds:
                lower, upper = self.control_bounds[control_name]
                bounded_controls[control_name] = np.clip(control_values, lower, upper)
            else:
                bounded_controls[control_name] = control_values.copy()
        
        return bounded_controls
    
    def check_convergence(self, 
                         objective_history: List[float],
                         constraint_violations: List[Dict[str, float]]) -> bool:
        """Check if optimization has converged.
        
        Args:
            objective_history: History of objective function values
            constraint_violations: History of constraint violations
            
        Returns:
            True if converged
        """
        if len(objective_history) < 10:
            return False
        
        # Check objective function convergence
        recent_objectives = objective_history[-5:]
        objective_change = abs(max(recent_objectives) - min(recent_objectives))
        relative_change = objective_change / abs(recent_objectives[-1]) if abs(recent_objectives[-1]) > 1e-10 else objective_change
        
        if relative_change > self.tolerance:
            return False
        
        # Check constraint satisfaction
        if len(constraint_violations) > 0:
            recent_violations = constraint_violations[-1]
            max_violation = max(recent_violations.values()) if recent_violations else 0.0
            
            if max_violation > self.constraint_tolerance:
                return False
        
        return True
    
    def penalty_function(self, 
                        objective_value: float,
                        constraint_violations: Dict[str, float],
                        penalty_parameter: float = 1e3) -> float:
        """Apply penalty function for constraint violations.
        
        Args:
            objective_value: Original objective function value
            constraint_violations: Dictionary of constraint violations  
            penalty_parameter: Penalty parameter
            
        Returns:
            Penalized objective value
        """
        penalty = 0.0
        
        for constraint_name, violation in constraint_violations.items():
            # Find constraint weight
            weight = 1.0
            for constraint in self.constraints:
                if constraint.name == constraint_name:
                    weight = constraint.weight
                    break
            
            penalty += weight * penalty_parameter * violation**2
        
        return objective_value + penalty
    
    def validate_solution(self, 
                         result: OptimizationResult,
                         initial_state: VehicleState,
                         time_span: Tuple[float, float]) -> bool:
        """Validate optimization result.
        
        Args:
            result: Optimization result to validate
            initial_state: Initial vehicle state
            time_span: Time span for validation
            
        Returns:
            True if solution is valid
        """
        if not result.success or result.optimal_controls is None:
            return False
        
        try:
            # Re-simulate with optimal controls
            trajectory = self.vehicle_dynamics.integrate_trajectory(
                initial_state, time_span, result.optimal_controls
            )
            
            # Check trajectory validity
            if any(np.any(np.isnan(values)) for values in trajectory.values()):
                self.logger.error("Trajectory contains NaN values")
                return False
            
            # Check altitude bounds (vehicle should not go underground)
            if np.any(trajectory['altitude'] < -1000):
                self.logger.error("Vehicle altitude goes below ground level")
                return False
            
            # Check velocity bounds (should remain positive)
            if np.any(trajectory['velocity'] < 0):
                self.logger.error("Vehicle velocity becomes negative")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Solution validation failed: {str(e)}")
            return False
    
    def create_control_parameterization(self, 
                                       time_span: Tuple[float, float],
                                       num_segments: int = 20,
                                       parameterization: str = 'piecewise_constant') -> Tuple[np.ndarray, Callable]:
        """Create control parameterization for optimization.
        
        Args:
            time_span: Time span for optimization
            num_segments: Number of control segments
            parameterization: Type of parameterization
            
        Returns:
            Tuple of (time_points, interpolation_function)
        """
        t_start, t_end = time_span
        
        if parameterization == 'piecewise_constant':
            # Piecewise constant controls
            time_points = np.linspace(t_start, t_end, num_segments + 1)
            
            def interpolate_controls(control_values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
                """Interpolate piecewise constant controls."""
                interpolated = np.zeros_like(query_times)
                
                for i, t in enumerate(query_times):
                    # Find which segment this time belongs to
                    segment_idx = np.searchsorted(time_points[1:], t)
                    segment_idx = min(segment_idx, len(control_values) - 1)
                    interpolated[i] = control_values[segment_idx]
                
                return interpolated
            
            return time_points, interpolate_controls
        
        elif parameterization == 'linear':
            # Piecewise linear controls
            time_points = np.linspace(t_start, t_end, num_segments)
            
            def interpolate_controls(control_values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
                """Interpolate piecewise linear controls."""
                return np.interp(query_times, time_points, control_values)
            
            return time_points, interpolate_controls
        
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
    
    def log_optimization_progress(self, 
                                 iteration: int,
                                 objective_value: float,
                                 constraint_violations: Dict[str, float],
                                 step_size: Optional[float] = None) -> None:
        """Log optimization progress.
        
        Args:
            iteration: Current iteration number
            objective_value: Current objective function value
            constraint_violations: Current constraint violations
            step_size: Optimization step size (if applicable)
        """
        max_violation = max(constraint_violations.values()) if constraint_violations else 0.0
        
        log_message = f"Iteration {iteration}: Objective = {objective_value:.6f}, Max Violation = {max_violation:.6f}"
        
        if step_size is not None:
            log_message += f", Step Size = {step_size:.6f}"
        
        self.logger.info(log_message)
        
        # Store history
        self.optimization_history.append({
            'iteration': iteration,
            'objective': objective_value,
            'violations': constraint_violations.copy(),
            'step_size': step_size
        })