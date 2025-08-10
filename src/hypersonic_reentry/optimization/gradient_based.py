"""Gradient-based trajectory optimization algorithms.

This module implements gradient-based optimization methods including:
- Sequential Quadratic Programming (SQP)
- Interior Point Methods
- Finite difference gradient computation
- Automatic differentiation support
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize, NonlinearConstraint
import time

from .trajectory_optimizer import TrajectoryOptimizer, OptimizationResult
from ..dynamics.vehicle_dynamics import VehicleState


class GradientBasedOptimizer(TrajectoryOptimizer):
    """Gradient-based trajectory optimization using SQP and interior point methods.
    
    Implements sophisticated gradient-based optimization with:
    - Finite difference gradient computation
    - Constraint linearization
    - Line search and trust region methods
    - Convergence acceleration techniques
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize gradient-based optimizer."""
        super().__init__(*args, **kwargs)
        
        # Gradient computation settings
        self.finite_difference_step = 1e-6
        self.gradient_method = 'forward'  # 'forward', 'central', 'complex_step'
        
        # Optimization algorithm settings
        self.algorithm = 'SLSQP'  # 'SLSQP', 'trust-constr'
        self.line_search_method = 'strong_wolfe'
        self.hessian_approximation = 'BFGS'
        
        # Convergence criteria
        self.gradient_tolerance = 1e-6
        self.step_tolerance = 1e-8
        
        self.logger.info("Initialized gradient-based optimizer")
    
    def optimize(self, 
                initial_state: VehicleState,
                time_span: Tuple[float, float],
                initial_guess: Optional[Dict[str, np.ndarray]] = None) -> OptimizationResult:
        """Run gradient-based trajectory optimization.
        
        Args:
            initial_state: Initial vehicle state
            time_span: Time span for optimization
            initial_guess: Initial guess for control variables
            
        Returns:
            OptimizationResult containing optimal solution
        """
        self.logger.info("Starting gradient-based trajectory optimization")
        start_time = time.time()
        
        # Generate initial guess if not provided
        if initial_guess is None:
            initial_guess = self.generate_initial_guess(initial_state, time_span)
        
        # Set up control parameterization
        num_control_points = len(list(initial_guess.values())[0])
        time_points, interpolation_func = self.create_control_parameterization(
            time_span, num_control_points
        )
        
        # Convert control dictionaries to optimization variable vector
        x0 = self._controls_to_vector(initial_guess)
        
        # Set up bounds
        bounds = self._create_bounds_vector(initial_guess)
        
        # Set up objective function
        def objective_function(x):
            controls = self._vector_to_controls(x, initial_guess.keys())
            return self.evaluate_objective(controls, initial_state, time_span)
        
        # Set up constraints
        constraint_functions = []
        if self.constraints:
            def constraint_function(x):
                controls = self._vector_to_controls(x, initial_guess.keys())
                violations = self.evaluate_constraints(controls, initial_state, time_span)
                return list(violations.values())
            
            constraint_functions.append(constraint_function)
        
        # Run optimization
        result = OptimizationResult()
        
        try:
            if self.algorithm == 'SLSQP':
                scipy_result = self._run_slsqp_optimization(
                    objective_function, x0, bounds, constraint_functions
                )
            elif self.algorithm == 'trust-constr':
                scipy_result = self._run_trust_constr_optimization(
                    objective_function, x0, bounds, constraint_functions
                )
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            # Process results
            result.success = scipy_result.success
            result.message = scipy_result.message
            result.num_iterations = scipy_result.nit
            result.final_objective_value = scipy_result.fun
            
            # Extract optimal controls
            if result.success:
                optimal_x = scipy_result.x
                result.optimal_controls = self._vector_to_controls(optimal_x, initial_guess.keys())
                
                # Apply bounds to ensure feasibility
                result.optimal_controls = self.apply_control_bounds(result.optimal_controls)
                
                # Simulate optimal trajectory
                result.optimal_trajectory = self.vehicle_dynamics.integrate_trajectory(
                    initial_state, time_span, result.optimal_controls
                )
                
                # Calculate performance metrics
                result.optimal_performance = self.vehicle_dynamics.calculate_performance_metrics(
                    result.optimal_trajectory
                )
                
                # Evaluate final constraint violations
                result.constraint_violations = self.evaluate_constraints(
                    result.optimal_controls, initial_state, time_span
                )
        
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            result.success = False
            result.message = f"Optimization error: {str(e)}"
        
        result.computation_time = time.time() - start_time
        
        # Validate solution
        if result.success:
            is_valid = self.validate_solution(result, initial_state, time_span)
            if not is_valid:
                result.success = False
                result.message = "Solution validation failed"
        
        self.logger.info(f"Gradient-based optimization completed in {result.computation_time:.2f} seconds")
        
        return result
    
    def _run_slsqp_optimization(self, 
                               objective_func,
                               x0: np.ndarray,
                               bounds: List[Tuple[float, float]],
                               constraint_funcs: List) -> object:
        """Run SLSQP optimization."""
        # Set up constraints for SLSQP
        constraints = []
        
        for constraint_func in constraint_funcs:
            # SLSQP expects constraint functions that return >= 0 for feasible points
            def slsqp_constraint(x, func=constraint_func):
                violations = func(x)
                # Convert violations (>= 0 for violated) to SLSQP format (<= 0 for feasible)
                return [-v for v in violations]
            
            constraints.append({'type': 'ineq', 'fun': slsqp_constraint})
        
        # Run SLSQP
        return minimize(
            objective_func,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'eps': self.finite_difference_step,
                'disp': True
            }
        )
    
    def _run_trust_constr_optimization(self, 
                                     objective_func,
                                     x0: np.ndarray,
                                     bounds: List[Tuple[float, float]],
                                     constraint_funcs: List) -> object:
        """Run trust-constr optimization."""
        from scipy.optimize import Bounds
        
        # Convert bounds
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        bounds_obj = Bounds(lower_bounds, upper_bounds)
        
        # Set up constraints
        constraints = []
        for constraint_func in constraint_funcs:
            # Trust-constr expects constraint functions
            def trust_constraint(x, func=constraint_func):
                violations = func(x)
                return np.array(violations)
            
            # Constraint: violations <= 0 (feasible when violations are zero or negative)
            constraints.append(NonlinearConstraint(
                trust_constraint, 
                -np.inf, 
                np.zeros(len(self.constraints))
            ))
        
        # Run trust-constr
        return minimize(
            objective_func,
            x0,
            method='trust-constr',
            bounds=bounds_obj,
            constraints=constraints,
            options={
                'maxiter': self.max_iterations,
                'gtol': self.gradient_tolerance,
                'xtol': self.step_tolerance,
                'verbose': 1
            }
        )
    
    def compute_gradient(self, 
                        objective_func,
                        x: np.ndarray,
                        step_size: Optional[float] = None) -> np.ndarray:
        """Compute gradient using finite differences.
        
        Args:
            objective_func: Objective function
            x: Current point
            step_size: Finite difference step size
            
        Returns:
            Gradient vector
        """
        if step_size is None:
            step_size = self.finite_difference_step
        
        n = len(x)
        gradient = np.zeros(n)
        
        if self.gradient_method == 'forward':
            # Forward difference
            f0 = objective_func(x)
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += step_size
                f_plus = objective_func(x_plus)
                gradient[i] = (f_plus - f0) / step_size
        
        elif self.gradient_method == 'central':
            # Central difference
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += step_size
                x_minus[i] -= step_size
                
                f_plus = objective_func(x_plus)
                f_minus = objective_func(x_minus)
                gradient[i] = (f_plus - f_minus) / (2 * step_size)
        
        elif self.gradient_method == 'complex_step':
            # Complex step differentiation (if supported)
            for i in range(n):
                x_complex = x.astype(complex)
                x_complex[i] += 1j * step_size
                f_complex = objective_func(x_complex)
                gradient[i] = np.imag(f_complex) / step_size
        
        return gradient
    
    def compute_hessian_approximation(self, 
                                    gradient_current: np.ndarray,
                                    gradient_previous: np.ndarray,
                                    step: np.ndarray,
                                    hessian_previous: np.ndarray) -> np.ndarray:
        """Compute Hessian approximation using BFGS update.
        
        Args:
            gradient_current: Current gradient
            gradient_previous: Previous gradient
            step: Optimization step taken
            hessian_previous: Previous Hessian approximation
            
        Returns:
            Updated Hessian approximation
        """
        y = gradient_current - gradient_previous
        s = step
        
        # BFGS update
        rho = 1.0 / np.dot(y, s)
        
        if rho > 0:  # Check for positive definiteness
            I = np.eye(len(gradient_current))
            H_new = (I - rho * np.outer(s, y)) @ hessian_previous @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        else:
            # Fall back to previous Hessian if update would not be positive definite
            H_new = hessian_previous
        
        return H_new
    
    def line_search(self, 
                   objective_func,
                   x: np.ndarray,
                   search_direction: np.ndarray,
                   gradient: np.ndarray,
                   alpha_max: float = 1.0) -> float:
        """Perform line search to find optimal step size.
        
        Args:
            objective_func: Objective function
            x: Current point
            search_direction: Search direction
            gradient: Current gradient
            alpha_max: Maximum step size
            
        Returns:
            Optimal step size
        """
        # Armijo line search
        c1 = 1e-4  # Armijo constant
        rho = 0.5  # Backtracking factor
        
        f0 = objective_func(x)
        grad_dot_dir = np.dot(gradient, search_direction)
        
        alpha = alpha_max
        
        for _ in range(20):  # Maximum line search iterations
            x_new = x + alpha * search_direction
            f_new = objective_func(x_new)
            
            # Armijo condition
            if f_new <= f0 + c1 * alpha * grad_dot_dir:
                return alpha
            
            alpha *= rho
        
        return alpha  # Return last alpha if no improvement found
    
    def _controls_to_vector(self, controls: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert control dictionaries to optimization vector."""
        return np.concatenate([controls[key] for key in sorted(controls.keys())])
    
    def _vector_to_controls(self, x: np.ndarray, control_names: List[str]) -> Dict[str, np.ndarray]:
        """Convert optimization vector to control dictionaries."""
        controls = {}
        start_idx = 0
        
        # Assume equal length arrays for all controls
        control_length = len(x) // len(control_names)
        
        for name in sorted(control_names):
            end_idx = start_idx + control_length
            controls[name] = x[start_idx:end_idx]
            start_idx = end_idx
        
        return controls
    
    def _create_bounds_vector(self, initial_guess: Dict[str, np.ndarray]) -> List[Tuple[float, float]]:
        """Create bounds vector for optimization variables."""
        bounds = []
        
        for control_name in sorted(initial_guess.keys()):
            if control_name in self.control_bounds:
                lower, upper = self.control_bounds[control_name]
                control_length = len(initial_guess[control_name])
                bounds.extend([(lower, upper)] * control_length)
            else:
                # No bounds specified
                control_length = len(initial_guess[control_name])
                bounds.extend([(None, None)] * control_length)
        
        return bounds