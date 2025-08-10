"""Comprehensive results generation system for hypersonic reentry analysis.

This module provides automated generation of simulation results including:
- Large-scale Monte Carlo simulations
- Optimization scenario comparisons
- Sensitivity analysis studies
- Performance benchmarking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import yaml
import h5py
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass, asdict
import pickle
import time

from ..dynamics import VehicleDynamics, VehicleState
from ..atmosphere import USStandard1976
from ..uncertainty import UncertaintyQuantifier, UncertainParameter
from ..optimization import GradientBasedOptimizer, OptimizationObjective, OptimizationConstraint
from ..utils.constants import DEG_TO_RAD, RAD_TO_DEG


@dataclass
class SimulationScenario:
    """Definition of a simulation scenario."""
    name: str
    description: str
    initial_conditions: Dict[str, float]
    vehicle_parameters: Dict[str, float]
    uncertain_parameters: List[Dict[str, Any]]
    optimization_settings: Optional[Dict[str, Any]] = None
    simulation_settings: Dict[str, float] = None


@dataclass
class ResultsMetadata:
    """Metadata for simulation results."""
    scenario_name: str
    generation_time: str
    num_samples: int
    computation_time: float
    git_commit: Optional[str] = None
    system_info: Dict[str, str] = None


class ResultsGenerator:
    """Comprehensive results generation system.
    
    Provides automated generation of large-scale simulation results including
    Monte Carlo studies, optimization comparisons, and sensitivity analysis.
    """
    
    def __init__(self, 
                 output_directory: str = "results",
                 parallel_workers: Optional[int] = None,
                 save_format: str = "hdf5"):
        """Initialize results generator.
        
        Args:
            output_directory: Directory for saving results
            parallel_workers: Number of parallel workers (default: CPU count)
            save_format: Data format for results ('hdf5', 'pickle', 'json')
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Parallel processing setup
        self.parallel_workers = parallel_workers or mp.cpu_count()
        self.save_format = save_format
        
        # Results storage
        self.results_database = {}
        self.metadata_database = {}
        
        self.logger.info(f"Initialized ResultsGenerator with {self.parallel_workers} workers")
    
    def generate_monte_carlo_study(self,
                                  scenario: SimulationScenario,
                                  num_samples: int = 1000,
                                  save_all_trajectories: bool = False,
                                  chunk_size: int = 100) -> Dict[str, Any]:
        """Generate comprehensive Monte Carlo simulation study.
        
        Args:
            scenario: Simulation scenario definition
            num_samples: Number of Monte Carlo samples
            save_all_trajectories: Whether to save all trajectory data
            chunk_size: Size of parallel processing chunks
            
        Returns:
            Dictionary containing complete Monte Carlo results
        """
        self.logger.info(f"Starting Monte Carlo study: {scenario.name} ({num_samples} samples)")
        start_time = time.time()
        
        # Set up simulation components
        dynamics, initial_state, uncertain_params = self._setup_scenario(scenario)
        
        # Create uncertainty quantifier
        uq = UncertaintyQuantifier(
            vehicle_dynamics=dynamics,
            uncertain_parameters=uncertain_params,
            random_seed=42
        )
        
        # Generate parameter samples
        parameter_samples = uq.monte_carlo.generate_samples(num_samples, method='lhs')
        
        # Run simulations in parallel chunks
        all_results = []
        all_trajectories = [] if save_all_trajectories else None
        
        # Split samples into chunks for parallel processing
        sample_chunks = [parameter_samples[i:i+chunk_size] 
                        for i in range(0, len(parameter_samples), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._simulate_chunk, chunk, scenario, save_all_trajectories): i
                for i, chunk in enumerate(sample_chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results, chunk_trajectories = future.result()
                    all_results.extend(chunk_results)
                    if save_all_trajectories and chunk_trajectories:
                        all_trajectories.extend(chunk_trajectories)
                    
                    completed_samples = len(all_results)
                    progress = (completed_samples / num_samples) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({completed_samples}/{num_samples})")
                    
                except Exception as e:
                    self.logger.error(f"Chunk {chunk_idx} failed: {str(e)}")
        
        # Process results
        mc_results = self._process_monte_carlo_results(
            all_results, all_trajectories, parameter_samples[:len(all_results)]
        )
        
        # Add metadata
        computation_time = time.time() - start_time
        metadata = ResultsMetadata(
            scenario_name=scenario.name,
            generation_time=datetime.now().isoformat(),
            num_samples=len(all_results),
            computation_time=computation_time,
            system_info=self._get_system_info()
        )
        
        # Save results
        self._save_results(f"mc_{scenario.name}", mc_results, metadata)
        
        self.logger.info(f"Monte Carlo study completed in {computation_time:.1f}s")
        return mc_results
    
    def generate_optimization_comparison(self,
                                       base_scenario: SimulationScenario,
                                       reentry_angles: List[float] = [-1.0, -3.0, -5.0, -10.0],
                                       optimization_methods: List[str] = ['gradient']) -> Dict[str, Any]:
        """Generate optimization comparison for different reentry conditions.
        
        Args:
            base_scenario: Base scenario to modify
            reentry_angles: List of flight path angles (degrees) to test
            optimization_methods: Optimization methods to compare
            
        Returns:
            Dictionary containing optimization comparison results
        """
        self.logger.info("Starting optimization comparison study")
        start_time = time.time()
        
        comparison_results = {
            'scenarios': [],
            'optimization_results': {},
            'performance_comparison': {},
            'convergence_analysis': {}
        }
        
        # Generate scenarios for different reentry angles
        scenarios = []
        for angle in reentry_angles:
            scenario = self._create_reentry_scenario(base_scenario, angle)
            scenarios.append(scenario)
            comparison_results['scenarios'].append({
                'name': scenario.name,
                'flight_path_angle_deg': angle,
                'description': scenario.description
            })
        
        # Run optimization for each scenario and method
        for method in optimization_methods:
            comparison_results['optimization_results'][method] = {}
            
            for scenario in scenarios:
                self.logger.info(f"Optimizing {scenario.name} with {method}")
                
                # Set up optimization
                dynamics, initial_state, _ = self._setup_scenario(scenario)
                optimizer = self._create_optimizer(dynamics, method, scenario.optimization_settings)
                
                # Run optimization
                opt_result = optimizer.optimize(
                    initial_state=initial_state,
                    time_span=(0.0, scenario.simulation_settings.get('max_time', 3000.0))
                )
                
                # Store results
                result_data = {
                    'success': opt_result.success,
                    'message': opt_result.message,
                    'num_iterations': opt_result.num_iterations,
                    'computation_time': opt_result.computation_time,
                    'final_objective': opt_result.final_objective_value,
                    'constraint_violations': opt_result.constraint_violations,
                    'performance_metrics': opt_result.optimal_performance
                }
                
                comparison_results['optimization_results'][method][scenario.name] = result_data
        
        # Analyze performance comparison
        comparison_results['performance_comparison'] = self._analyze_optimization_performance(
            comparison_results['optimization_results']
        )
        
        # Save results
        computation_time = time.time() - start_time
        metadata = ResultsMetadata(
            scenario_name="optimization_comparison",
            generation_time=datetime.now().isoformat(),
            num_samples=len(scenarios) * len(optimization_methods),
            computation_time=computation_time
        )
        
        self._save_results("optimization_comparison", comparison_results, metadata)
        
        self.logger.info(f"Optimization comparison completed in {computation_time:.1f}s")
        return comparison_results
    
    def generate_sensitivity_analysis(self,
                                    scenario: SimulationScenario,
                                    analysis_method: str = 'sobol',
                                    num_base_samples: int = 500) -> Dict[str, Any]:
        """Generate comprehensive sensitivity analysis.
        
        Args:
            scenario: Simulation scenario
            analysis_method: Sensitivity analysis method ('sobol', 'morris', 'correlation')
            num_base_samples: Base number of samples for analysis
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        self.logger.info(f"Starting sensitivity analysis: {analysis_method}")
        start_time = time.time()
        
        # Set up simulation components
        dynamics, initial_state, uncertain_params = self._setup_scenario(scenario)
        
        # Create uncertainty quantifier
        uq = UncertaintyQuantifier(
            vehicle_dynamics=dynamics,
            uncertain_parameters=uncertain_params,
            random_seed=42
        )
        
        # Run sensitivity analysis
        sensitivity_results = uq.run_sensitivity_analysis(
            initial_state=initial_state,
            time_span=(0.0, scenario.simulation_settings.get('max_time', 3000.0)),
            method=analysis_method
        )
        
        # Enhanced analysis
        if analysis_method == 'sobol':
            sensitivity_results['total_effect_indices'] = self._calculate_total_effects(
                sensitivity_results
            )
            sensitivity_results['parameter_ranking'] = self._rank_parameters(
                sensitivity_results['indices']
            )
        
        # Add parameter correlation analysis
        sensitivity_results['correlation_analysis'] = self._correlation_analysis(
            sensitivity_results.get('samples_A', np.array([])),
            sensitivity_results.get('results_A', {})
        )
        
        # Save results
        computation_time = time.time() - start_time
        metadata = ResultsMetadata(
            scenario_name=f"sensitivity_{scenario.name}",
            generation_time=datetime.now().isoformat(),
            num_samples=num_base_samples * (len(uncertain_params) + 2),
            computation_time=computation_time
        )
        
        self._save_results(f"sensitivity_{scenario.name}", sensitivity_results, metadata)
        
        self.logger.info(f"Sensitivity analysis completed in {computation_time:.1f}s")
        return sensitivity_results
    
    def generate_convergence_study(self,
                                 scenario: SimulationScenario,
                                 sample_sizes: List[int] = [100, 250, 500, 1000, 2500, 5000],
                                 num_replications: int = 5) -> Dict[str, Any]:
        """Generate Monte Carlo convergence study.
        
        Args:
            scenario: Simulation scenario
            sample_sizes: List of sample sizes to test
            num_replications: Number of replications for each sample size
            
        Returns:
            Dictionary containing convergence study results
        """
        self.logger.info("Starting convergence study")
        start_time = time.time()
        
        convergence_results = {
            'sample_sizes': sample_sizes,
            'replications': num_replications,
            'convergence_data': {},
            'statistical_tests': {}
        }
        
        # Set up simulation components
        dynamics, initial_state, uncertain_params = self._setup_scenario(scenario)
        
        output_metrics = ['final_altitude', 'final_velocity', 'downrange', 'max_heat_rate']
        
        for sample_size in sample_sizes:
            self.logger.info(f"Testing sample size: {sample_size}")
            
            replication_results = []
            
            for rep in range(num_replications):
                # Create uncertainty quantifier with different seed for each replication
                uq = UncertaintyQuantifier(
                    vehicle_dynamics=dynamics,
                    uncertain_parameters=uncertain_params,
                    random_seed=42 + rep
                )
                
                # Run Monte Carlo with current sample size
                mc_result = uq.run_monte_carlo_analysis(
                    initial_state=initial_state,
                    time_span=(0.0, scenario.simulation_settings.get('max_time', 3000.0)),
                    num_samples=sample_size,
                    output_quantities=output_metrics,
                    parallel=False  # Disable nested parallelism
                )
                
                # Extract statistics
                rep_stats = {}
                for metric in output_metrics:
                    if metric in mc_result.mean_values:
                        rep_stats[f"{metric}_mean"] = mc_result.mean_values[metric]
                        rep_stats[f"{metric}_std"] = mc_result.std_deviations[metric]
                
                replication_results.append(rep_stats)
            
            convergence_results['convergence_data'][sample_size] = replication_results
        
        # Analyze convergence
        convergence_results['convergence_metrics'] = self._analyze_convergence(
            convergence_results['convergence_data'], output_metrics
        )
        
        # Save results
        computation_time = time.time() - start_time
        metadata = ResultsMetadata(
            scenario_name=f"convergence_{scenario.name}",
            generation_time=datetime.now().isoformat(),
            num_samples=sum(sample_sizes) * num_replications,
            computation_time=computation_time
        )
        
        self._save_results(f"convergence_{scenario.name}", convergence_results, metadata)
        
        self.logger.info(f"Convergence study completed in {computation_time:.1f}s")
        return convergence_results
    
    def _setup_scenario(self, scenario: SimulationScenario) -> Tuple[VehicleDynamics, VehicleState, List[UncertainParameter]]:
        """Set up simulation components from scenario definition."""
        # Create atmosphere model
        atmosphere = USStandard1976(
            include_uncertainties=True,
            uncertainty_params={'density_uncertainty': 0.15}
        )
        
        # Create vehicle dynamics
        dynamics = VehicleDynamics(
            vehicle_params=scenario.vehicle_parameters,
            atmosphere_model=atmosphere
        )
        
        # Create initial state
        ic = scenario.initial_conditions
        initial_state = VehicleState(
            altitude=ic['altitude'],
            latitude=ic['latitude'] * DEG_TO_RAD,
            longitude=ic['longitude'] * DEG_TO_RAD,
            velocity=ic['velocity'],
            flight_path_angle=ic['flight_path_angle'] * DEG_TO_RAD,
            azimuth=ic['azimuth'] * DEG_TO_RAD,
            time=0.0
        )
        
        # Create uncertain parameters
        uncertain_params = []
        for param_def in scenario.uncertain_parameters:
            uncertain_params.append(UncertainParameter(**param_def))
        
        return dynamics, initial_state, uncertain_params
    
    def _simulate_chunk(self, 
                       parameter_chunk: np.ndarray,
                       scenario: SimulationScenario,
                       save_trajectories: bool = False) -> Tuple[List[Dict], Optional[List[Dict]]]:
        """Simulate a chunk of parameter samples."""
        dynamics, initial_state, _ = self._setup_scenario(scenario)
        
        chunk_results = []
        chunk_trajectories = [] if save_trajectories else None
        
        time_span = (0.0, scenario.simulation_settings.get('max_time', 3000.0))
        
        for params in parameter_chunk:
            try:
                # Update dynamics with parameter values
                modified_dynamics = self._update_dynamics_parameters(dynamics, params, scenario)
                
                # Run simulation
                trajectory = modified_dynamics.integrate_trajectory(
                    initial_state, time_span, 
                    time_step=scenario.simulation_settings.get('time_step', 0.1)
                )
                
                # Calculate performance metrics
                performance = modified_dynamics.calculate_performance_metrics(trajectory)
                chunk_results.append(performance)
                
                if save_trajectories:
                    chunk_trajectories.append(trajectory)
                    
            except Exception as e:
                # Log error and add NaN result
                self.logger.warning(f"Simulation failed: {str(e)}")
                nan_result = {key: np.nan for key in ['final_altitude', 'final_velocity', 
                                                    'downrange', 'flight_time', 'max_heat_rate']}
                chunk_results.append(nan_result)
                
                if save_trajectories:
                    chunk_trajectories.append({})
        
        return chunk_results, chunk_trajectories
    
    def _process_monte_carlo_results(self, 
                                   results: List[Dict],
                                   trajectories: Optional[List[Dict]],
                                   parameter_samples: np.ndarray) -> Dict[str, Any]:
        """Process Monte Carlo simulation results into comprehensive analysis."""
        # Convert results to DataFrame for analysis
        df_results = pd.DataFrame(results)
        
        # Remove failed simulations (NaN values)
        valid_mask = ~df_results.isnull().any(axis=1)
        df_clean = df_results[valid_mask]
        valid_samples = parameter_samples[valid_mask]
        
        processed_results = {
            'raw_data': {
                'performance_metrics': df_clean.to_dict('records'),
                'parameter_samples': valid_samples.tolist(),
                'num_valid_samples': len(df_clean),
                'num_failed_samples': len(results) - len(df_clean)
            },
            'statistical_summary': {},
            'uncertainty_bounds': {},
            'probability_distributions': {},
            'risk_metrics': {}
        }
        
        # Statistical summary
        for column in df_clean.columns:
            processed_results['statistical_summary'][column] = {
                'mean': float(df_clean[column].mean()),
                'std': float(df_clean[column].std()),
                'min': float(df_clean[column].min()),
                'max': float(df_clean[column].max()),
                'median': float(df_clean[column].median()),
                'skewness': float(df_clean[column].skew()),
                'kurtosis': float(df_clean[column].kurtosis())
            }
        
        # Confidence intervals
        confidence_levels = [68, 90, 95, 99]
        for column in df_clean.columns:
            processed_results['uncertainty_bounds'][column] = {}
            for conf_level in confidence_levels:
                alpha = (100 - conf_level) / 2
                lower = np.percentile(df_clean[column], alpha)
                upper = np.percentile(df_clean[column], 100 - alpha)
                processed_results['uncertainty_bounds'][column][f"{conf_level}%"] = {
                    'lower': float(lower),
                    'upper': float(upper)
                }
        
        # Probability distributions (histograms)
        for column in df_clean.columns:
            hist, bin_edges = np.histogram(df_clean[column], bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            processed_results['probability_distributions'][column] = {
                'bin_centers': bin_centers.tolist(),
                'probability_density': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        
        # Risk metrics
        processed_results['risk_metrics'] = self._calculate_risk_metrics(df_clean)
        
        # Add trajectory statistics if available
        if trajectories:
            processed_results['trajectory_statistics'] = self._analyze_trajectory_ensemble(trajectories)
        
        return processed_results
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk assessment metrics."""
        risk_metrics = {}
        
        # Define failure thresholds
        thresholds = {
            'final_altitude': {'min': 25000, 'max': 35000},  # Acceptable altitude range (m)
            'max_heat_rate': {'max': 5e6},  # Maximum allowable heat rate (W/m^2)
            'downrange': {'min': 1000000, 'max': 3000000}  # Target range limits (m)
        }
        
        for metric, limits in thresholds.items():
            if metric in df.columns:
                data = df[metric]
                metric_risk = {}
                
                # Probability of exceeding upper limit
                if 'max' in limits:
                    exceed_prob = (data > limits['max']).mean()
                    metric_risk['probability_exceed_max'] = float(exceed_prob)
                
                # Probability of being below lower limit
                if 'min' in limits:
                    below_prob = (data < limits['min']).mean()
                    metric_risk['probability_below_min'] = float(below_prob)
                
                # Overall success probability
                success_mask = True
                if 'max' in limits:
                    success_mask &= (data <= limits['max'])
                if 'min' in limits:
                    success_mask &= (data >= limits['min'])
                
                metric_risk['success_probability'] = float(success_mask.mean())
                risk_metrics[metric] = metric_risk
        
        return risk_metrics
    
    def _create_reentry_scenario(self, base_scenario: SimulationScenario, flight_path_angle: float) -> SimulationScenario:
        """Create reentry scenario with modified flight path angle."""
        new_ic = base_scenario.initial_conditions.copy()
        new_ic['flight_path_angle'] = flight_path_angle
        
        return SimulationScenario(
            name=f"{base_scenario.name}_fpa_{flight_path_angle:.1f}deg",
            description=f"Reentry with {flight_path_angle:.1f}° flight path angle",
            initial_conditions=new_ic,
            vehicle_parameters=base_scenario.vehicle_parameters.copy(),
            uncertain_parameters=base_scenario.uncertain_parameters.copy(),
            optimization_settings=base_scenario.optimization_settings.copy() if base_scenario.optimization_settings else None,
            simulation_settings=base_scenario.simulation_settings.copy() if base_scenario.simulation_settings else {}
        )
    
    def _create_optimizer(self, dynamics: VehicleDynamics, method: str, settings: Dict) -> GradientBasedOptimizer:
        """Create optimizer with specified method and settings."""
        # Define standard objectives and constraints
        objectives = [
            OptimizationObjective(
                name="downrange",
                objective_type="maximize",
                weight=1.0
            )
        ]
        
        constraints = [
            OptimizationConstraint(
                name="final_altitude",
                constraint_type="equality",
                target_value=30000.0,
                tolerance=1000.0
            ),
            OptimizationConstraint(
                name="max_heat_rate",
                constraint_type="path_max",
                target_value=5e6
            )
        ]
        
        control_bounds = {
            "bank_angle": (-60.0 * DEG_TO_RAD, 60.0 * DEG_TO_RAD),
            "angle_of_attack": (0.0 * DEG_TO_RAD, 40.0 * DEG_TO_RAD)
        }
        
        optimizer = GradientBasedOptimizer(
            vehicle_dynamics=dynamics,
            objectives=objectives,
            constraints=constraints,
            control_bounds=control_bounds
        )
        
        # Apply settings
        if settings:
            optimizer.max_iterations = settings.get('max_iterations', 100)
            optimizer.tolerance = settings.get('tolerance', 1e-6)
            optimizer.algorithm = settings.get('algorithm', 'SLSQP')
        
        return optimizer
    
    def _save_results(self, name: str, results: Dict[str, Any], metadata: ResultsMetadata):
        """Save results in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.output_dir / f"{name}_{timestamp}"
        
        # Save metadata
        metadata_path = base_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        # Save results based on format
        if self.save_format == 'hdf5':
            self._save_hdf5(base_path.with_suffix('.h5'), results)
        elif self.save_format == 'pickle':
            with open(base_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(results, f)
        elif self.save_format == 'json':
            with open(base_path.with_suffix('.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved: {base_path}")
    
    def _save_hdf5(self, filepath: Path, results: Dict[str, Any]):
        """Save results in HDF5 format."""
        with h5py.File(filepath, 'w') as f:
            self._dict_to_hdf5(f, results)
    
    def _dict_to_hdf5(self, h5group, data_dict):
        """Recursively save dictionary to HDF5."""
        for key, value in data_dict.items():
            if isinstance(value, dict):
                subgroup = h5group.create_group(key)
                self._dict_to_hdf5(subgroup, value)
            elif isinstance(value, (list, np.ndarray)):
                h5group.create_dataset(key, data=np.array(value))
            elif isinstance(value, (int, float, str)):
                h5group.attrs[key] = value
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for metadata."""
        import platform
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': str(mp.cpu_count()),
            'architecture': platform.architecture()[0]
        }
    
    def _update_dynamics_parameters(self, dynamics: VehicleDynamics, params: np.ndarray, scenario: SimulationScenario) -> VehicleDynamics:
        """Update dynamics with uncertain parameter values."""
        # This is a simplified implementation - in practice, you would map
        # parameter indices to specific vehicle/atmosphere parameters
        modified_dynamics = dynamics  # In practice, create a copy
        
        # Example parameter mapping (customize based on scenario.uncertain_parameters)
        if len(params) >= 4:
            modified_dynamics.mass = params[0]
            modified_dynamics.drag_coefficient = params[1]
            modified_dynamics.lift_coefficient = params[2]
            modified_dynamics.reference_area = params[3]
        
        return modified_dynamics
    
    def _analyze_optimization_performance(self, opt_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze optimization performance across methods and scenarios."""
        analysis = {
            'success_rates': {},
            'convergence_comparison': {},
            'performance_comparison': {},
            'efficiency_metrics': {}
        }
        
        for method, method_results in opt_results.items():
            # Success rate
            successes = sum(1 for result in method_results.values() if result['success'])
            analysis['success_rates'][method] = successes / len(method_results)
            
            # Average iterations and time
            iterations = [r['num_iterations'] for r in method_results.values() if r['success']]
            times = [r['computation_time'] for r in method_results.values() if r['success']]
            
            if iterations:
                analysis['convergence_comparison'][method] = {
                    'avg_iterations': np.mean(iterations),
                    'avg_time': np.mean(times),
                    'std_iterations': np.std(iterations),
                    'std_time': np.std(times)
                }
            
            # Performance metrics comparison
            objectives = [r['final_objective'] for r in method_results.values() if r['success']]
            if objectives:
                analysis['performance_comparison'][method] = {
                    'best_objective': np.min(objectives),
                    'avg_objective': np.mean(objectives),
                    'worst_objective': np.max(objectives)
                }
        
        return analysis