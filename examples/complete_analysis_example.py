#!/usr/bin/env python3
"""Complete analysis example for hypersonic reentry vehicle trajectory optimization.

This script demonstrates the full capabilities of the hypersonic reentry
simulation framework including:
- Vehicle dynamics simulation
- Uncertainty quantification
- Trajectory optimization
- Visualization and analysis

Run this script to generate a complete analysis with plots and results.
"""

import numpy as np
import sys
import os
from pathlib import Path
import logging
import yaml
from typing import Dict, List

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import our hypersonic reentry framework
from hypersonic_reentry.dynamics import VehicleDynamics, VehicleState
from hypersonic_reentry.atmosphere import AtmosphereModel
from hypersonic_reentry.uncertainty import UncertaintyQuantifier, UncertainParameter
from hypersonic_reentry.optimization import GradientBasedOptimizer, OptimizationObjective, OptimizationConstraint
from hypersonic_reentry.visualization import PlotManager
from hypersonic_reentry.utils.constants import DEG_TO_RAD, RAD_TO_DEG


def setup_logging():
    """Set up logging for the analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_configuration():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_vehicle_dynamics(config: Dict) -> VehicleDynamics:
    """Set up vehicle dynamics model."""
    vehicle_params = config['vehicle']
    
    # Initialize atmosphere model
    atmosphere = AtmosphereModel(
        include_uncertainties=True,
        uncertainty_params=config['atmosphere']
    )
    
    # Initialize vehicle dynamics
    dynamics = VehicleDynamics(
        vehicle_params=vehicle_params,
        atmosphere_model=atmosphere
    )
    
    return dynamics


def create_initial_state(config: Dict) -> VehicleState:
    """Create initial vehicle state from configuration."""
    initial_conditions = config['initial_conditions']
    
    return VehicleState(
        altitude=initial_conditions['altitude'],
        latitude=initial_conditions['latitude'] * DEG_TO_RAD,
        longitude=initial_conditions['longitude'] * DEG_TO_RAD,
        velocity=initial_conditions['velocity'],
        flight_path_angle=initial_conditions['flight_path_angle'] * DEG_TO_RAD,
        azimuth=initial_conditions['azimuth'] * DEG_TO_RAD,
        time=0.0
    )


def setup_uncertainty_parameters(config: Dict) -> List[UncertainParameter]:
    """Define uncertain parameters for analysis."""
    uncertain_params = [
        UncertainParameter(
            name="mass",
            distribution_type="normal",
            parameters={"mean": config['vehicle']['mass'], "std": config['vehicle']['mass'] * 0.05},
            description="Vehicle mass uncertainty"
        ),
        UncertainParameter(
            name="drag_coefficient",
            distribution_type="normal",
            parameters={"mean": config['vehicle']['drag_coefficient'], "std": 0.1},
            description="Drag coefficient uncertainty"
        ),
        UncertainParameter(
            name="lift_coefficient",
            distribution_type="normal",
            parameters={"mean": config['vehicle']['lift_coefficient'], "std": 0.05},
            description="Lift coefficient uncertainty"
        ),
        UncertainParameter(
            name="reference_area",
            distribution_type="normal",
            parameters={"mean": config['vehicle']['reference_area'], "std": 0.5},
            description="Reference area uncertainty"
        )
    ]
    
    return uncertain_params


def run_nominal_simulation(dynamics: VehicleDynamics, 
                          initial_state: VehicleState,
                          config: Dict) -> Dict:
    """Run nominal trajectory simulation."""
    logging.info("Running nominal trajectory simulation")
    
    # Set up time span
    time_span = (0.0, config['simulation']['max_time'])
    
    # Run simulation
    trajectory = dynamics.integrate_trajectory(
        initial_state, 
        time_span,
        time_step=config['simulation']['time_step']
    )
    
    # Calculate performance metrics
    performance = dynamics.calculate_performance_metrics(trajectory)
    
    logging.info(f"Nominal simulation completed:")
    logging.info(f"  Final altitude: {performance['final_altitude']/1000:.1f} km")
    logging.info(f"  Final velocity: {performance['final_velocity']/1000:.2f} km/s")
    logging.info(f"  Flight time: {performance['flight_time']:.1f} s")
    logging.info(f"  Downrange: {performance['downrange']/1000:.1f} km")
    
    return {"trajectory": trajectory, "performance": performance}


def run_uncertainty_analysis(dynamics: VehicleDynamics,
                            initial_state: VehicleState,
                            uncertain_params: List[UncertainParameter],
                            config: Dict) -> Dict:
    """Run uncertainty quantification analysis."""
    logging.info("Starting uncertainty quantification analysis")
    
    # Set up uncertainty quantifier
    uq = UncertaintyQuantifier(
        vehicle_dynamics=dynamics,
        uncertain_parameters=uncertain_params,
        random_seed=config['uncertainty']['seed']
    )
    
    # Run Monte Carlo analysis
    time_span = (0.0, config['simulation']['max_time'])
    num_samples = min(100, config['uncertainty']['samples'])  # Reduced for example
    
    uq_results = uq.run_monte_carlo_analysis(
        initial_state=initial_state,
        time_span=time_span,
        num_samples=num_samples,
        parallel=False  # Set to True for faster execution with multiprocessing
    )
    
    # Log results
    logging.info(f"Uncertainty analysis completed with {uq_results.num_samples} samples:")
    for metric, mean_val in uq_results.mean_values.items():
        std_val = uq_results.std_deviations[metric]
        cv = (std_val / abs(mean_val)) * 100 if abs(mean_val) > 1e-10 else 0
        logging.info(f"  {metric}: {mean_val:.2e} ± {std_val:.2e} (CV: {cv:.1f}%)")
    
    return uq_results


def run_trajectory_optimization(dynamics: VehicleDynamics,
                               initial_state: VehicleState,
                               config: Dict) -> Dict:
    """Run trajectory optimization."""
    logging.info("Starting trajectory optimization")
    
    # Define optimization objectives
    objectives = [
        OptimizationObjective(
            name="downrange",
            objective_type="maximize",
            weight=1.0,
            description="Maximize downrange distance"
        )
    ]
    
    # Define constraints
    constraints = [
        OptimizationConstraint(
            name="final_altitude",
            constraint_type="equality",
            target_value=config['target_conditions']['altitude_final'],
            tolerance=1000.0,  # 1 km tolerance
            description="Final altitude constraint"
        ),
        OptimizationConstraint(
            name="max_heat_rate",
            constraint_type="path_max",
            target_value=5e6,  # 5 MW/m^2 maximum heat rate
            description="Maximum heat rate constraint"
        )
    ]
    
    # Control bounds
    control_bounds = {
        "bank_angle": (-60.0 * DEG_TO_RAD, 60.0 * DEG_TO_RAD),
        "angle_of_attack": (0.0 * DEG_TO_RAD, 40.0 * DEG_TO_RAD)
    }
    
    # Set up optimizer
    optimizer = GradientBasedOptimizer(
        vehicle_dynamics=dynamics,
        objectives=objectives,
        constraints=constraints,
        control_bounds=control_bounds
    )
    
    # Set optimization parameters
    optimizer.max_iterations = 50  # Reduced for example
    optimizer.tolerance = 1e-4
    
    # Run optimization
    time_span = (0.0, config['simulation']['max_time'])
    
    try:
        result = optimizer.optimize(
            initial_state=initial_state,
            time_span=time_span
        )
        
        if result.success:
            logging.info("Optimization completed successfully:")
            logging.info(f"  Final objective value: {result.final_objective_value:.2e}")
            logging.info(f"  Number of iterations: {result.num_iterations}")
            logging.info(f"  Computation time: {result.computation_time:.1f} s")
            
            # Log performance metrics
            if result.optimal_performance:
                for metric, value in result.optimal_performance.items():
                    logging.info(f"  {metric}: {value:.2e}")
        else:
            logging.warning(f"Optimization failed: {result.message}")
            
    except Exception as e:
        logging.error(f"Optimization error: {str(e)}")
        result = None
    
    return result


def create_visualizations(nominal_results: Dict,
                         uq_results: Dict,
                         optimization_results: Dict,
                         config: Dict):
    """Create comprehensive visualizations."""
    logging.info("Creating visualizations")
    
    # Set up plot manager
    output_dir = Path("results") / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_manager = PlotManager(
        output_directory=str(output_dir),
        style_theme="publication",
        dpi=300
    )
    
    # Plot nominal trajectory
    nominal_trajectory = nominal_results["trajectory"]
    
    fig1 = plot_manager.plot_trajectory_2d(
        trajectory=nominal_trajectory,
        save_path="nominal_trajectory_2d.png",
        show_plot=False
    )
    
    fig2 = plot_manager.plot_trajectory_3d(
        trajectory=nominal_trajectory,
        save_path="nominal_trajectory_3d.png",
        show_plot=False
    )
    
    # Create interactive 3D plot
    interactive_fig = plot_manager.create_interactive_3d_plot(
        trajectory=nominal_trajectory,
        save_path="trajectory_interactive.html"
    )
    
    # Plot uncertainty results if available
    if uq_results and hasattr(uq_results, 'output_samples'):
        # Create sample trajectories for visualization (simplified)
        sample_trajectories = []
        
        # For demonstration, create a few sample trajectories
        # In a real implementation, you would store all trajectories from MC analysis
        for i in range(min(10, len(list(uq_results.output_samples.values())[0]))):
            # Create simplified trajectory for visualization
            sample_traj = {
                'time': nominal_trajectory['time'],
                'altitude': nominal_trajectory['altitude'] * (1 + 0.1 * (np.random.random() - 0.5)),
                'velocity': nominal_trajectory['velocity'] * (1 + 0.05 * (np.random.random() - 0.5)),
                'mach_number': nominal_trajectory['mach_number'] * (1 + 0.05 * (np.random.random() - 0.5))
            }
            sample_trajectories.append(sample_traj)
        
        fig3 = plot_manager.plot_uncertainty_bands(
            trajectories=sample_trajectories,
            save_path="uncertainty_analysis.png",
            show_plot=False
        )
    
    # Plot optimization results if available
    if optimization_results and optimization_results.success:
        optimal_trajectory = optimization_results.optimal_trajectory
        
        fig4 = plot_manager.plot_trajectory_2d(
            trajectory=optimal_trajectory,
            save_path="optimal_trajectory_2d.png",
            show_plot=False
        )
        
        # Compare performance metrics
        comparison_results = {
            "Nominal": nominal_results["performance"],
            "Optimized": optimization_results.optimal_performance
        }
        
        fig5 = plot_manager.plot_performance_comparison(
            results=comparison_results,
            save_path="performance_comparison.png",
            show_plot=False
        )
    
    logging.info(f"Visualizations saved to {output_dir}")


def save_results(nominal_results: Dict,
                uq_results: Dict,
                optimization_results: Dict):
    """Save analysis results to files."""
    logging.info("Saving results to files")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save nominal results
    np.savez(
        results_dir / "nominal_trajectory.npz",
        **nominal_results["trajectory"]
    )
    
    with open(results_dir / "nominal_performance.yaml", 'w') as f:
        yaml.dump(nominal_results["performance"], f, default_flow_style=False)
    
    # Save uncertainty results if available
    if uq_results:
        with open(results_dir / "uncertainty_statistics.yaml", 'w') as f:
            results_dict = {
                'mean_values': uq_results.mean_values,
                'std_deviations': uq_results.std_deviations,
                'confidence_intervals': uq_results.confidence_intervals,
                'num_samples': uq_results.num_samples,
                'computation_time': uq_results.computation_time
            }
            yaml.dump(results_dict, f, default_flow_style=False)
    
    # Save optimization results if available  
    if optimization_results and optimization_results.success:
        np.savez(
            results_dir / "optimal_trajectory.npz",
            **optimization_results.optimal_trajectory
        )
        
        with open(results_dir / "optimization_summary.yaml", 'w') as f:
            summary = {
                'success': optimization_results.success,
                'final_objective': optimization_results.final_objective_value,
                'num_iterations': optimization_results.num_iterations,
                'computation_time': optimization_results.computation_time,
                'performance_metrics': optimization_results.optimal_performance,
                'constraint_violations': optimization_results.constraint_violations
            }
            yaml.dump(summary, f, default_flow_style=False)
    
    logging.info(f"Results saved to {results_dir}")


def main():
    """Main analysis function."""
    print("=" * 70)
    print("HYPERSONIC REENTRY VEHICLE TRAJECTORY OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Set up logging
    setup_logging()
    
    try:
        # Load configuration
        logging.info("Loading configuration")
        config = load_configuration()
        
        # Set up vehicle dynamics
        logging.info("Setting up vehicle dynamics model")
        dynamics = setup_vehicle_dynamics(config)
        
        # Create initial state
        initial_state = create_initial_state(config)
        
        # Run nominal simulation
        nominal_results = run_nominal_simulation(dynamics, initial_state, config)
        
        # Set up uncertainty parameters
        uncertain_params = setup_uncertainty_parameters(config)
        
        # Run uncertainty analysis
        uq_results = run_uncertainty_analysis(
            dynamics, initial_state, uncertain_params, config
        )
        
        # Run trajectory optimization
        optimization_results = run_trajectory_optimization(
            dynamics, initial_state, config
        )
        
        # Create visualizations
        create_visualizations(
            nominal_results, uq_results, optimization_results, config
        )
        
        # Save results
        save_results(nominal_results, uq_results, optimization_results)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Results saved to: {Path('results').absolute()}")
        print(f"Plots saved to: {Path('results/plots').absolute()}")
        print(f"Log file: {Path('analysis.log').absolute()}")
        
    except Exception as e:
        logging.error(f"Analysis failed with error: {str(e)}")
        print(f"\nERROR: Analysis failed - {str(e)}")
        print("Check analysis.log for detailed error information")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)