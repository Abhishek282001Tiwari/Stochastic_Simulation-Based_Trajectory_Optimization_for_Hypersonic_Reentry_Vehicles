"""Uncertainty quantification framework for hypersonic reentry vehicles.

This module provides a comprehensive framework for quantifying uncertainties
in trajectory simulations including parameter uncertainties, model uncertainties,
and their propagation through the system dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import pickle
import time

from .monte_carlo import MonteCarloSampler
from .polynomial_chaos import PolynomialChaosExpansion
from .sensitivity_analysis import SensitivityAnalyzer
from ..dynamics.vehicle_dynamics import VehicleDynamics, VehicleState


@dataclass
class UncertainParameter:
    """Definition of an uncertain parameter for UQ analysis."""
    
    name: str
    distribution_type: str  # 'normal', 'uniform', 'lognormal', 'beta'
    parameters: Dict[str, float]  # Distribution parameters
    bounds: Optional[Tuple[float, float]] = None  # Parameter bounds
    description: str = ""


@dataclass
class UQResults:
    """Results from uncertainty quantification analysis."""
    
    # Statistical moments
    mean_values: Dict[str, float] = field(default_factory=dict)
    std_deviations: Dict[str, float] = field(default_factory=dict)
    variances: Dict[str, float] = field(default_factory=dict)
    
    # Percentiles and confidence intervals
    percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    
    # Sensitivity analysis results
    sobol_indices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    correlation_coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Raw data
    input_samples: Optional[np.ndarray] = None
    output_samples: Optional[Dict[str, np.ndarray]] = None
    
    # Metadata
    num_samples: int = 0
    computation_time: float = 0.0
    method_used: str = ""


class UncertaintyQuantifier:
    """Main class for uncertainty quantification in hypersonic reentry simulation.
    
    Provides methods for:
    - Parameter uncertainty definition and sampling
    - Monte Carlo simulation with uncertainty propagation
    - Statistical analysis of results
    - Sensitivity analysis
    - Polynomial chaos expansion (if applicable)
    """
    
    def __init__(self, 
                 vehicle_dynamics: VehicleDynamics,
                 uncertain_parameters: List[UncertainParameter],
                 random_seed: Optional[int] = None):
        """Initialize uncertainty quantification framework.
        
        Args:
            vehicle_dynamics: Vehicle dynamics model
            uncertain_parameters: List of uncertain parameters
            random_seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.vehicle_dynamics = vehicle_dynamics
        self.uncertain_parameters = uncertain_parameters
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            self.random_seed = random_seed
        else:
            self.random_seed = np.random.randint(0, 2**31)
            np.random.seed(self.random_seed)
        
        # Initialize UQ methods
        self.monte_carlo = MonteCarloSampler(uncertain_parameters, random_seed)
        self.polynomial_chaos = PolynomialChaosExpansion(uncertain_parameters)
        self.sensitivity_analyzer = SensitivityAnalyzer(uncertain_parameters)
        
        # Storage for results
        self.results_cache = {}
        
        self.logger.info(f"Initialized UQ with {len(uncertain_parameters)} uncertain parameters")
    
    def run_monte_carlo_analysis(self, 
                                initial_state: VehicleState,
                                time_span: Tuple[float, float],
                                num_samples: int = 1000,
                                output_quantities: Optional[List[str]] = None,
                                parallel: bool = True,
                                save_all_trajectories: bool = False) -> UQResults:
        """Run Monte Carlo uncertainty propagation analysis.
        
        Args:
            initial_state: Nominal initial vehicle state
            time_span: Time span for trajectory integration
            num_samples: Number of Monte Carlo samples
            output_quantities: List of output quantities to analyze
            parallel: Whether to use parallel processing
            save_all_trajectories: Whether to save all trajectory data
            
        Returns:
            UQResults object containing statistical analysis
        """
        self.logger.info(f"Starting Monte Carlo analysis with {num_samples} samples")
        start_time = time.time()
        
        # Default output quantities
        if output_quantities is None:
            output_quantities = [
                'final_altitude', 'final_velocity', 'flight_time',
                'downrange', 'max_mach_number', 'max_dynamic_pressure', 
                'max_heat_rate', 'total_heat_load'
            ]
        
        # Generate parameter samples
        parameter_samples = self.monte_carlo.generate_samples(num_samples)
        
        # Run simulations
        if parallel:
            results = self._run_parallel_simulations(
                parameter_samples, initial_state, time_span, output_quantities
            )
        else:
            results = self._run_sequential_simulations(
                parameter_samples, initial_state, time_span, output_quantities
            )
        
        # Extract output data
        output_data = {}
        trajectories = []
        
        for i, result in enumerate(results):
            if result is not None:
                performance_metrics, trajectory = result
                
                # Store performance metrics
                for qty in output_quantities:
                    if qty in performance_metrics:
                        if qty not in output_data:
                            output_data[qty] = []
                        output_data[qty].append(performance_metrics[qty])
                
                # Store trajectory if requested
                if save_all_trajectories:
                    trajectories.append(trajectory)
        
        # Convert to numpy arrays
        for qty in output_data:
            output_data[qty] = np.array(output_data[qty])
        
        # Statistical analysis
        uq_results = self._compute_statistics(
            parameter_samples, output_data, output_quantities
        )
        
        # Sensitivity analysis
        sensitivity_results = self.sensitivity_analyzer.compute_sobol_indices(
            parameter_samples, output_data
        )
        uq_results.sobol_indices = sensitivity_results
        
        # Store metadata
        uq_results.num_samples = len(results)
        uq_results.computation_time = time.time() - start_time
        uq_results.method_used = "Monte Carlo"
        uq_results.input_samples = parameter_samples
        uq_results.output_samples = output_data
        
        self.logger.info(f"Monte Carlo analysis completed in {uq_results.computation_time:.2f} seconds")
        
        return uq_results
    
    def _run_parallel_simulations(self, 
                                 parameter_samples: np.ndarray,
                                 initial_state: VehicleState,
                                 time_span: Tuple[float, float],
                                 output_quantities: List[str]) -> List:
        """Run simulations in parallel using multiprocessing."""
        
        def simulate_single_case(params):
            """Simulate single parameter case."""
            try:
                # Update uncertain parameters in models
                modified_dynamics = self._update_dynamics_parameters(params)
                
                # Run trajectory simulation
                trajectory = modified_dynamics.integrate_trajectory(
                    initial_state, time_span
                )
                
                # Calculate performance metrics
                performance_metrics = modified_dynamics.calculate_performance_metrics(trajectory)
                
                return performance_metrics, trajectory
            
            except Exception as e:
                self.logger.warning(f"Simulation failed: {str(e)}")
                return None
        
        # Run simulations in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(simulate_single_case, params) 
                      for params in parameter_samples]
            results = [future.result() for future in futures]
        
        return results
    
    def _run_sequential_simulations(self, 
                                   parameter_samples: np.ndarray,
                                   initial_state: VehicleState,
                                   time_span: Tuple[float, float],
                                   output_quantities: List[str]) -> List:
        """Run simulations sequentially."""
        results = []
        
        for i, params in enumerate(parameter_samples):
            try:
                # Update uncertain parameters in models
                modified_dynamics = self._update_dynamics_parameters(params)
                
                # Run trajectory simulation  
                trajectory = modified_dynamics.integrate_trajectory(
                    initial_state, time_span
                )
                
                # Calculate performance metrics
                performance_metrics = modified_dynamics.calculate_performance_metrics(trajectory)
                
                results.append((performance_metrics, trajectory))
                
                # Progress reporting
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(parameter_samples)} simulations")
            
            except Exception as e:
                self.logger.warning(f"Simulation {i} failed: {str(e)}")
                results.append(None)
        
        return results
    
    def _update_dynamics_parameters(self, parameter_values: np.ndarray) -> VehicleDynamics:
        """Update vehicle dynamics with uncertain parameter values."""
        # Create copy of dynamics model
        modified_dynamics = self.vehicle_dynamics
        
        # Update parameters based on uncertain parameter definitions
        for i, param in enumerate(self.uncertain_parameters):
            param_value = parameter_values[i]
            
            # Update vehicle parameters
            if param.name == 'mass':
                modified_dynamics.mass = param_value
            elif param.name == 'drag_coefficient':
                modified_dynamics.drag_coefficient = param_value
            elif param.name == 'lift_coefficient':
                modified_dynamics.lift_coefficient = param_value
            elif param.name == 'reference_area':
                modified_dynamics.reference_area = param_value
            # Add more parameter updates as needed
        
        return modified_dynamics
    
    def _compute_statistics(self, 
                           input_samples: np.ndarray,
                           output_data: Dict[str, np.ndarray],
                           output_quantities: List[str]) -> UQResults:
        """Compute statistical measures from simulation results."""
        results = UQResults()
        
        # Percentile levels for analysis
        percentile_levels = [1, 5, 25, 50, 75, 95, 99]
        confidence_levels = [90, 95, 99]
        
        for qty in output_quantities:
            if qty in output_data:
                data = output_data[qty]
                
                # Basic statistics
                results.mean_values[qty] = np.mean(data)
                results.std_deviations[qty] = np.std(data)
                results.variances[qty] = np.var(data)
                
                # Percentiles
                percentiles = np.percentile(data, percentile_levels)
                results.percentiles[qty] = {
                    f"p{level}": value 
                    for level, value in zip(percentile_levels, percentiles)
                }
                
                # Confidence intervals
                results.confidence_intervals[qty] = {}
                for conf_level in confidence_levels:
                    alpha = (100 - conf_level) / 2
                    lower = np.percentile(data, alpha)
                    upper = np.percentile(data, 100 - alpha)
                    results.confidence_intervals[qty][f"{conf_level}%"] = (lower, upper)
        
        return results
    
    def run_sensitivity_analysis(self, 
                                initial_state: VehicleState,
                                time_span: Tuple[float, float],
                                output_quantities: Optional[List[str]] = None,
                                method: str = 'sobol') -> Dict[str, Any]:
        """Run parameter sensitivity analysis.
        
        Args:
            initial_state: Initial vehicle state
            time_span: Time span for simulation
            output_quantities: Output quantities to analyze
            method: Sensitivity analysis method ('sobol', 'morris', 'correlation')
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        self.logger.info(f"Running sensitivity analysis using {method} method")
        
        if output_quantities is None:
            output_quantities = ['final_altitude', 'final_velocity', 'downrange']
        
        if method == 'sobol':
            return self._run_sobol_sensitivity(
                initial_state, time_span, output_quantities
            )
        elif method == 'morris':
            return self._run_morris_sensitivity(
                initial_state, time_span, output_quantities
            )
        elif method == 'correlation':
            return self._run_correlation_sensitivity(
                initial_state, time_span, output_quantities
            )
        else:
            raise ValueError(f"Unknown sensitivity analysis method: {method}")
    
    def _run_sobol_sensitivity(self, 
                              initial_state: VehicleState,
                              time_span: Tuple[float, float],
                              output_quantities: List[str]) -> Dict[str, Any]:
        """Run Sobol sensitivity analysis."""
        # Generate Sobol samples
        num_base_samples = 500  # Adjust based on computational budget
        
        # Generate samples for Sobol analysis
        samples_A, samples_B = self.sensitivity_analyzer.generate_sobol_samples(
            num_base_samples
        )
        
        # Run simulations for both sample sets
        results_A = self._evaluate_samples(samples_A, initial_state, time_span)
        results_B = self._evaluate_samples(samples_B, initial_state, time_span)
        
        # Calculate Sobol indices
        sobol_indices = {}
        for qty in output_quantities:
            if qty in results_A and qty in results_B:
                indices = self.sensitivity_analyzer.calculate_sobol_indices(
                    samples_A, samples_B, results_A[qty], results_B[qty]
                )
                sobol_indices[qty] = indices
        
        return {
            'method': 'sobol',
            'indices': sobol_indices,
            'samples_A': samples_A,
            'samples_B': samples_B,
            'results_A': results_A,
            'results_B': results_B
        }
    
    def _evaluate_samples(self, 
                         samples: np.ndarray,
                         initial_state: VehicleState,
                         time_span: Tuple[float, float]) -> Dict[str, np.ndarray]:
        """Evaluate model for given parameter samples."""
        results = {}
        
        for i, params in enumerate(samples):
            try:
                # Update dynamics with parameter values
                modified_dynamics = self._update_dynamics_parameters(params)
                
                # Run simulation
                trajectory = modified_dynamics.integrate_trajectory(
                    initial_state, time_span
                )
                
                # Calculate performance metrics
                performance_metrics = modified_dynamics.calculate_performance_metrics(trajectory)
                
                # Store results
                for key, value in performance_metrics.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
            
            except Exception as e:
                self.logger.warning(f"Sample {i} evaluation failed: {str(e)}")
                # Add NaN values for failed simulations
                for key in results:
                    results[key].append(np.nan)
        
        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results
    
    def run_polynomial_chaos_expansion(self, 
                                     initial_state: VehicleState,
                                     time_span: Tuple[float, float],
                                     polynomial_order: int = 3,
                                     output_quantities: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run polynomial chaos expansion for uncertainty quantification.
        
        Args:
            initial_state: Initial vehicle state
            time_span: Time span for simulation
            polynomial_order: Order of polynomial expansion
            output_quantities: Output quantities to analyze
            
        Returns:
            Dictionary containing PCE results
        """
        self.logger.info(f"Running polynomial chaos expansion (order {polynomial_order})")
        
        if output_quantities is None:
            output_quantities = ['final_altitude', 'final_velocity', 'downrange']
        
        # Generate samples for PCE
        samples = self.polynomial_chaos.generate_collocation_points(polynomial_order)
        
        # Evaluate model at collocation points
        model_evaluations = self._evaluate_samples(samples, initial_state, time_span)
        
        # Build polynomial chaos expansion
        pce_results = {}
        for qty in output_quantities:
            if qty in model_evaluations:
                pce_coeffs, pce_stats = self.polynomial_chaos.build_expansion(
                    samples, model_evaluations[qty], polynomial_order
                )
                
                pce_results[qty] = {
                    'coefficients': pce_coeffs,
                    'statistics': pce_stats,
                    'polynomial_order': polynomial_order
                }
        
        return {
            'method': 'polynomial_chaos',
            'polynomial_order': polynomial_order,
            'results': pce_results,
            'samples': samples,
            'evaluations': model_evaluations
        }
    
    def validate_surrogate_model(self, 
                                surrogate_results: Dict[str, Any],
                                initial_state: VehicleState,
                                time_span: Tuple[float, float],
                                num_validation_samples: int = 100) -> Dict[str, float]:
        """Validate surrogate model accuracy against direct simulations.
        
        Args:
            surrogate_results: Results from surrogate model (PCE)
            initial_state: Initial vehicle state
            time_span: Time span for simulation
            num_validation_samples: Number of validation samples
            
        Returns:
            Dictionary containing validation metrics
        """
        self.logger.info("Validating surrogate model accuracy")
        
        # Generate validation samples
        validation_samples = self.monte_carlo.generate_samples(num_validation_samples)
        
        # Direct model evaluations
        direct_results = self._evaluate_samples(validation_samples, initial_state, time_span)
        
        # Surrogate model predictions
        surrogate_predictions = {}
        if surrogate_results['method'] == 'polynomial_chaos':
            for qty, pce_data in surrogate_results['results'].items():
                if qty in direct_results:
                    predictions = self.polynomial_chaos.evaluate_expansion(
                        pce_data['coefficients'], validation_samples
                    )
                    surrogate_predictions[qty] = predictions
        
        # Calculate validation metrics
        validation_metrics = {}
        for qty in surrogate_predictions:
            direct_vals = direct_results[qty]
            surrogate_vals = surrogate_predictions[qty]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(direct_vals) | np.isnan(surrogate_vals))
            direct_vals = direct_vals[valid_mask]
            surrogate_vals = surrogate_vals[valid_mask]
            
            if len(direct_vals) > 0:
                # R-squared coefficient
                ss_res = np.sum((direct_vals - surrogate_vals)**2)
                ss_tot = np.sum((direct_vals - np.mean(direct_vals))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Root mean square error
                rmse = np.sqrt(np.mean((direct_vals - surrogate_vals)**2))
                
                # Relative error
                relative_error = rmse / np.mean(np.abs(direct_vals))
                
                validation_metrics[qty] = {
                    'r_squared': r_squared,
                    'rmse': rmse,
                    'relative_error': relative_error,
                    'num_samples': len(direct_vals)
                }
        
        return validation_metrics