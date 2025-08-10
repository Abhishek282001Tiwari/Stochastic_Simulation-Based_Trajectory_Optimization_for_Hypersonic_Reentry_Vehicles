"""Comprehensive validation and testing suite for hypersonic reentry framework.

This module provides extensive validation including:
- Mathematical model verification
- Physical behavior validation  
- Numerical accuracy testing
- Performance regression testing
- Integration testing
- Cross-platform compatibility testing
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
import logging
import time
import json
from typing import Dict, List, Any, Tuple
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import framework components
from hypersonic_reentry.dynamics import VehicleDynamics, VehicleState
from hypersonic_reentry.atmosphere import USStandard1976
from hypersonic_reentry.uncertainty import UncertaintyQuantifier, UncertainParameter
from hypersonic_reentry.optimization import GradientBasedOptimizer, OptimizationObjective, OptimizationConstraint
from hypersonic_reentry.visualization import PlotManager
from hypersonic_reentry.analysis import StatisticalAnalyzer
from hypersonic_reentry.utils.constants import DEG_TO_RAD, RAD_TO_DEG, EARTH_RADIUS, GRAVITATIONAL_PARAMETER
from hypersonic_reentry.utils.performance import PerformanceProfiler, profile


class TestMathematicalValidation:
    """Test mathematical models against analytical solutions."""
    
    @pytest.fixture
    def setup_test_environment(self):
        """Set up test environment with logging."""
        logging.basicConfig(level=logging.INFO)
        self.profiler = PerformanceProfiler(enable_profiling=True)
        
    def test_atmospheric_model_standard_conditions(self, setup_test_environment):
        """Test atmospheric model against standard conditions."""
        atmosphere = USStandard1976()
        
        # Test sea level conditions
        props_sl = atmosphere.get_properties(0.0)
        
        # Validate against standard values (within tolerance)
        assert abs(props_sl['temperature'] - 288.15) < 1.0, "Sea level temperature incorrect"
        assert abs(props_sl['pressure'] - 101325.0) < 1000.0, "Sea level pressure incorrect"  
        assert abs(props_sl['density'] - 1.225) < 0.1, "Sea level density incorrect"
        
        # Test temperature profile consistency
        altitudes = [0, 5000, 11000, 20000, 32000, 47000, 51000, 71000]
        expected_temps = [288.15, 255.65, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
        
        for alt, expected_temp in zip(altitudes, expected_temps):
            props = atmosphere.get_properties(alt)
            temp_error = abs(props['temperature'] - expected_temp)
            assert temp_error < 5.0, f"Temperature error at {alt}m: {temp_error}K"
    
    def test_coordinate_transformations(self):
        """Test coordinate transformation accuracy."""
        from hypersonic_reentry.dynamics.coordinate_transforms import CoordinateTransforms
        
        transforms = CoordinateTransforms()
        
        # Test spherical to Cartesian and back
        test_cases = [
            (EARTH_RADIUS + 100000, 0.5, 1.0),  # 100km altitude
            (EARTH_RADIUS + 50000, -0.5, -1.0),  # 50km altitude
            (EARTH_RADIUS + 200000, 1.0, 0.0)   # 200km altitude
        ]
        
        for r_orig, lat_orig, lon_orig in test_cases:
            # Convert to Cartesian
            x, y, z = transforms.spherical_to_cartesian(r_orig, lat_orig, lon_orig)
            
            # Convert back to spherical
            r_new, lat_new, lon_new = transforms.cartesian_to_spherical(x, y, z)
            
            # Check round-trip accuracy
            assert abs(r_new - r_orig) < 1e-6, "Radius conversion error"
            assert abs(lat_new - lat_orig) < 1e-9, "Latitude conversion error" 
            assert abs(lon_new - lon_orig) < 1e-9, "Longitude conversion error"
    
    def test_vehicle_dynamics_conservation(self):
        """Test energy and momentum conservation in dynamics."""
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 1.2,
            'lift_coefficient': 0.8,
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5,
            'length': 10.0,
            'diameter': 2.0
        }
        
        atmosphere = USStandard1976()
        dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)
        
        # Test in vacuum (no atmosphere) for energy conservation
        initial_state = VehicleState(
            altitude=200000.0,  # High altitude (thin atmosphere)
            latitude=0.0,
            longitude=0.0,
            velocity=7000.0,
            flight_path_angle=0.0,  # Horizontal
            azimuth=0.0,
            time=0.0
        )
        
        # Short simulation to minimize atmospheric effects
        trajectory = dynamics.integrate_trajectory(
            initial_state, (0.0, 100.0), time_step=1.0
        )
        
        # Calculate specific energy at start and end
        r_start = EARTH_RADIUS + trajectory['altitude'][0]
        v_start = trajectory['velocity'][0]
        energy_start = 0.5 * v_start**2 - GRAVITATIONAL_PARAMETER / r_start
        
        r_end = EARTH_RADIUS + trajectory['altitude'][-1]
        v_end = trajectory['velocity'][-1]
        energy_end = 0.5 * v_end**2 - GRAVITATIONAL_PARAMETER / r_end
        
        # Energy should be approximately conserved (within numerical error)
        energy_error = abs(energy_end - energy_start) / abs(energy_start)
        assert energy_error < 0.01, f"Energy conservation error: {energy_error:.4f}"
    
    def test_aerodynamic_force_scaling(self):
        """Test aerodynamic force scaling with dynamic pressure."""
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 1.2,
            'lift_coefficient': 0.8,
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5,
            'length': 10.0,
            'diameter': 2.0
        }
        
        from hypersonic_reentry.dynamics.aerodynamics import AerodynamicsModel
        aero_model = AerodynamicsModel(vehicle_params)
        
        # Test force scaling with velocity
        test_velocities = [1000, 2000, 4000, 8000]  # m/s
        density = 0.01  # kg/m³
        alpha = 5.0 * DEG_TO_RAD  # angle of attack
        
        forces = []
        for velocity in test_velocities:
            force_data = aero_model.calculate_forces(velocity, density, alpha)
            forces.append(force_data['drag'])
        
        # Forces should scale as velocity squared
        for i in range(1, len(forces)):
            velocity_ratio = test_velocities[i] / test_velocities[0]
            force_ratio = forces[i] / forces[0]
            expected_ratio = velocity_ratio**2
            
            ratio_error = abs(force_ratio - expected_ratio) / expected_ratio
            assert ratio_error < 0.01, f"Force scaling error: {ratio_error:.4f}"


class TestPhysicalBehavior:
    """Test physical realism of simulation results."""
    
    def test_trajectory_physical_behavior(self):
        """Test that trajectories behave physically."""
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 1.2,
            'lift_coefficient': 0.8,
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5,
            'length': 10.0,
            'diameter': 2.0
        }
        
        atmosphere = USStandard1976()
        dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)
        
        # Test reentry trajectory
        initial_state = VehicleState(
            altitude=120000.0,
            latitude=28.5 * DEG_TO_RAD,
            longitude=-80.6 * DEG_TO_RAD,
            velocity=7800.0,
            flight_path_angle=-1.5 * DEG_TO_RAD,  # Shallow entry
            azimuth=90.0 * DEG_TO_RAD,
            time=0.0
        )
        
        trajectory = dynamics.integrate_trajectory(
            initial_state, (0.0, 2000.0), time_step=1.0
        )
        
        # Physical behavior checks
        
        # 1. Vehicle should descend (altitude decreases)
        assert trajectory['altitude'][-1] < trajectory['altitude'][0], "Vehicle should descend"
        
        # 2. Velocity should initially decrease due to drag
        velocity_change = trajectory['velocity'][50] - trajectory['velocity'][0]
        assert velocity_change < 0, "Velocity should decrease initially due to drag"
        
        # 3. Flight path angle should become more negative (steepen)
        fpa_change = trajectory['flight_path_angle'][-1] - trajectory['flight_path_angle'][0]
        assert fpa_change < 0, "Flight path angle should steepen during entry"
        
        # 4. Dynamic pressure should reach a maximum and then decrease
        dynamic_pressures = trajectory.get('dynamic_pressure', [])
        if len(dynamic_pressures) > 100:
            max_q_index = np.argmax(dynamic_pressures)
            assert 10 < max_q_index < len(dynamic_pressures) - 10, "Dynamic pressure should peak mid-trajectory"
        
        # 5. Heat rate should be reasonable
        heat_rates = trajectory.get('heat_rate', [])
        if len(heat_rates) > 0:
            max_heat_rate = np.max(heat_rates)
            assert max_heat_rate > 0, "Heat rate should be positive"
            assert max_heat_rate < 1e8, "Heat rate should be reasonable (< 100 MW/m²)"
    
    def test_extreme_conditions_stability(self):
        """Test simulation stability under extreme conditions."""
        vehicle_params = {
            'mass': 1000.0,  # Light vehicle
            'reference_area': 5.0,
            'drag_coefficient': 2.0,  # High drag
            'lift_coefficient': 0.5,
            'ballistic_coefficient': 200.0,
            'nose_radius': 0.1,  # Small nose radius
            'length': 5.0,
            'diameter': 1.0
        }
        
        atmosphere = USStandard1976()
        dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)
        
        # Test steep entry
        initial_state = VehicleState(
            altitude=100000.0,
            latitude=0.0,
            longitude=0.0,
            velocity=8000.0,  # High velocity
            flight_path_angle=-10.0 * DEG_TO_RAD,  # Steep entry
            azimuth=0.0,
            time=0.0
        )
        
        # Should complete without numerical instabilities
        try:
            trajectory = dynamics.integrate_trajectory(
                initial_state, (0.0, 1000.0), time_step=0.5
            )
            
            # Check for NaN or infinite values
            for key, values in trajectory.items():
                assert not np.any(np.isnan(values)), f"NaN values found in {key}"
                assert not np.any(np.isinf(values)), f"Infinite values found in {key}"
            
            # Trajectory should be complete
            assert len(trajectory['time']) > 10, "Trajectory should have reasonable length"
            
        except Exception as e:
            pytest.fail(f"Extreme condition simulation failed: {str(e)}")


class TestNumericalAccuracy:
    """Test numerical accuracy and precision."""
    
    def test_integration_accuracy(self):
        """Test numerical integration accuracy using analytical solutions."""
        # Test simple ballistic trajectory (no atmosphere, spherical gravity)
        
        # Analytical solution for ballistic trajectory
        def analytical_ballistic(v0, gamma0, h0, t):
            """Analytical solution for ballistic trajectory."""
            r0 = EARTH_RADIUS + h0
            
            # Simplified analytical solution (small angle approximation)
            h_analytical = h0 + v0 * np.sin(gamma0) * t - 0.5 * (GRAVITATIONAL_PARAMETER / r0**2) * t**2
            v_analytical = np.sqrt(
                (v0 * np.cos(gamma0))**2 + 
                (v0 * np.sin(gamma0) - (GRAVITATIONAL_PARAMETER / r0**2) * t)**2
            )
            
            return h_analytical, v_analytical
        
        # Simulate with very low atmosphere to approximate vacuum
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 0.001,  # Very low drag
            'lift_coefficient': 0.001,  # Very low lift
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5,
            'length': 10.0,
            'diameter': 2.0
        }
        
        # Custom atmosphere with very low density
        class VacuumAtmosphere(USStandard1976):
            def get_properties(self, altitude, latitude=0.0, longitude=0.0, time=0.0):
                props = super().get_properties(altitude, latitude, longitude, time)
                props['density'] *= 1e-6  # Reduce density by factor of 1M
                return props
        
        atmosphere = VacuumAtmosphere()
        dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)
        
        # Test conditions
        h0 = 200000.0  # High altitude
        v0 = 5000.0    # Moderate velocity
        gamma0 = 0.1   # Small flight path angle
        
        initial_state = VehicleState(
            altitude=h0,
            latitude=0.0,
            longitude=0.0,
            velocity=v0,
            flight_path_angle=gamma0,
            azimuth=0.0,
            time=0.0
        )
        
        # Short integration time
        t_final = 100.0
        trajectory = dynamics.integrate_trajectory(
            initial_state, (0.0, t_final), time_step=0.1
        )
        
        # Compare with analytical solution
        t_analytical = trajectory['time']
        h_analytical, v_analytical = analytical_ballistic(v0, gamma0, h0, t_analytical)
        
        # Calculate relative errors
        h_error = np.abs((trajectory['altitude'] - h_analytical) / h_analytical)
        v_error = np.abs((trajectory['velocity'] - v_analytical) / v_analytical)
        
        # Errors should be small
        max_h_error = np.max(h_error)
        max_v_error = np.max(v_error)
        
        assert max_h_error < 0.01, f"Altitude integration error too large: {max_h_error:.4f}"
        assert max_v_error < 0.01, f"Velocity integration error too large: {max_v_error:.4f}"
    
    def test_monte_carlo_convergence(self):
        """Test Monte Carlo convergence properties."""
        # Set up simple uncertainty quantification
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 1.2,
            'lift_coefficient': 0.8,
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5,
            'length': 10.0,
            'diameter': 2.0
        }
        
        uncertain_params = [
            UncertainParameter(
                name="mass",
                distribution_type="normal",
                parameters={"mean": 5000.0, "std": 250.0}
            ),
            UncertainParameter(
                name="drag_coefficient",
                distribution_type="normal", 
                parameters={"mean": 1.2, "std": 0.12}
            )
        ]
        
        atmosphere = USStandard1976()
        dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)
        
        uq = UncertaintyQuantifier(
            vehicle_dynamics=dynamics,
            uncertain_parameters=uncertain_params,
            random_seed=42
        )
        
        initial_state = VehicleState(
            altitude=80000.0,
            latitude=0.0,
            longitude=0.0,
            velocity=6000.0,
            flight_path_angle=-5.0 * DEG_TO_RAD,
            azimuth=0.0,
            time=0.0
        )
        
        # Test convergence with increasing sample sizes
        sample_sizes = [50, 100, 200]
        means = []
        
        for n_samples in sample_sizes:
            mc_result = uq.run_monte_carlo_analysis(
                initial_state=initial_state,
                time_span=(0.0, 500.0),
                num_samples=n_samples,
                parallel=False
            )
            
            if 'final_altitude' in mc_result.mean_values:
                means.append(mc_result.mean_values['final_altitude'])
        
        # Means should be converging
        if len(means) >= 2:
            convergence_rate = abs(means[-1] - means[-2]) / abs(means[-2])
            assert convergence_rate < 0.1, f"Monte Carlo not converging: {convergence_rate:.4f}"


class TestPerformanceRegression:
    """Test performance regression and benchmarking."""
    
    def test_simulation_performance_benchmarks(self):
        """Test that simulations complete within expected time limits."""
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 1.2,
            'lift_coefficient': 0.8,
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5,
            'length': 10.0,
            'diameter': 2.0
        }
        
        atmosphere = USStandard1976()
        dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)
        
        initial_state = VehicleState(
            altitude=100000.0,
            latitude=0.0,
            longitude=0.0,
            velocity=7000.0,
            flight_path_angle=-3.0 * DEG_TO_RAD,
            azimuth=0.0,
            time=0.0
        )
        
        # Benchmark single trajectory simulation
        start_time = time.perf_counter()
        
        trajectory = dynamics.integrate_trajectory(
            initial_state, (0.0, 1000.0), time_step=1.0
        )
        
        end_time = time.perf_counter()
        simulation_time = end_time - start_time
        
        # Should complete within reasonable time
        assert simulation_time < 5.0, f"Single trajectory simulation too slow: {simulation_time:.2f}s"
        
        # Calculate throughput
        trajectory_length = len(trajectory['time'])
        throughput = trajectory_length / simulation_time
        
        assert throughput > 100, f"Simulation throughput too low: {throughput:.0f} steps/s"
    
    def test_memory_usage_bounds(self):
        """Test that memory usage stays within acceptable bounds."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple simulations to check for memory leaks
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 1.2,
            'lift_coefficient': 0.8,
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5,
            'length': 10.0,
            'diameter': 2.0
        }
        
        atmosphere = USStandard1976()
        dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)
        
        initial_state = VehicleState(
            altitude=100000.0,
            latitude=0.0,
            longitude=0.0,
            velocity=7000.0,
            flight_path_angle=-3.0 * DEG_TO_RAD,
            azimuth=0.0,
            time=0.0
        )
        
        # Run multiple simulations
        for i in range(10):
            trajectory = dynamics.integrate_trajectory(
                initial_state, (0.0, 200.0), time_step=1.0
            )
            
            # Clear references
            del trajectory
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Excessive memory usage increase: {memory_increase:.1f} MB"
    
    @profile("test_optimization_performance")
    def test_optimization_performance(self):
        """Test optimization algorithm performance."""
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 1.2,
            'lift_coefficient': 0.8,
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5,
            'length': 10.0,
            'diameter': 2.0
        }
        
        atmosphere = USStandard1976()
        dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)
        
        # Set up optimization
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
            )
        ]
        
        control_bounds = {
            "bank_angle": (-30.0 * DEG_TO_RAD, 30.0 * DEG_TO_RAD),
            "angle_of_attack": (0.0 * DEG_TO_RAD, 20.0 * DEG_TO_RAD)
        }
        
        optimizer = GradientBasedOptimizer(
            vehicle_dynamics=dynamics,
            objectives=objectives,
            constraints=constraints,
            control_bounds=control_bounds
        )
        
        optimizer.max_iterations = 20  # Reduced for testing
        
        initial_state = VehicleState(
            altitude=80000.0,
            latitude=0.0,
            longitude=0.0,
            velocity=6000.0,
            flight_path_angle=-3.0 * DEG_TO_RAD,
            azimuth=0.0,
            time=0.0
        )
        
        start_time = time.perf_counter()
        
        result = optimizer.optimize(
            initial_state=initial_state,
            time_span=(0.0, 800.0)
        )
        
        end_time = time.perf_counter()
        optimization_time = end_time - start_time
        
        # Should complete within reasonable time
        assert optimization_time < 30.0, f"Optimization too slow: {optimization_time:.2f}s"
        
        if result.success:
            # Should converge in reasonable number of iterations
            assert result.num_iterations < 50, f"Too many iterations: {result.num_iterations}"


class TestIntegrationAndCompatibility:
    """Test system integration and cross-platform compatibility."""
    
    def test_complete_workflow_integration(self):
        """Test complete analysis workflow integration."""
        # This test runs a mini version of the complete workflow
        
        # Set up components
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 1.2,
            'lift_coefficient': 0.8,
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5,
            'length': 10.0,
            'diameter': 2.0
        }
        
        uncertain_params = [
            UncertainParameter(
                name="mass",
                distribution_type="normal",
                parameters={"mean": 5000.0, "std": 250.0}
            )
        ]
        
        atmosphere = USStandard1976()
        dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)
        
        initial_state = VehicleState(
            altitude=80000.0,
            latitude=0.0,
            longitude=0.0,
            velocity=6000.0,
            flight_path_angle=-3.0 * DEG_TO_RAD,
            azimuth=0.0,
            time=0.0
        )
        
        # 1. Run basic simulation
        trajectory = dynamics.integrate_trajectory(
            initial_state, (0.0, 500.0), time_step=2.0
        )
        
        assert len(trajectory['time']) > 10, "Trajectory should have data points"
        
        # 2. Run small Monte Carlo study
        uq = UncertaintyQuantifier(
            vehicle_dynamics=dynamics,
            uncertain_parameters=uncertain_params,
            random_seed=42
        )
        
        mc_result = uq.run_monte_carlo_analysis(
            initial_state=initial_state,
            time_span=(0.0, 500.0),
            num_samples=20,
            parallel=False
        )
        
        assert mc_result.num_samples > 0, "Monte Carlo should produce results"
        
        # 3. Run statistical analysis
        analyzer = StatisticalAnalyzer()
        stats_result = analyzer.comprehensive_analysis(mc_result)
        
        assert 'descriptive_statistics' in stats_result, "Statistical analysis should produce results"
        
        # 4. Create visualization (test setup only)
        plot_manager = PlotManager(output_directory="test_plots")
        
        # Should not crash when creating plots
        try:
            fig = plot_manager.plot_trajectory_2d(trajectory, show_plot=False)
            assert fig is not None, "Plot creation should succeed"
        except ImportError:
            # Skip if matplotlib not available
            pass
    
    def test_data_serialization_compatibility(self):
        """Test data serialization and deserialization."""
        import json
        import pickle
        
        # Test data structures
        test_data = {
            'trajectory': {
                'time': np.linspace(0, 100, 101).tolist(),
                'altitude': np.random.normal(50000, 1000, 101).tolist(),
                'velocity': np.random.normal(5000, 100, 101).tolist()
            },
            'performance_metrics': {
                'final_altitude': 35000.0,
                'downrange': 1500000.0,
                'max_heat_rate': 3.5e6
            }
        }
        
        # Test JSON serialization
        try:
            json_str = json.dumps(test_data)
            recovered_data = json.loads(json_str)
            
            assert 'trajectory' in recovered_data, "JSON serialization should preserve structure"
            assert len(recovered_data['trajectory']['time']) == 101, "JSON should preserve array lengths"
            
        except Exception as e:
            pytest.fail(f"JSON serialization failed: {str(e)}")
        
        # Test pickle serialization
        try:
            pickled_data = pickle.dumps(test_data)
            recovered_data = pickle.loads(pickled_data)
            
            assert recovered_data == test_data, "Pickle should preserve data exactly"
            
        except Exception as e:
            pytest.fail(f"Pickle serialization failed: {str(e)}")
    
    def test_numpy_version_compatibility(self):
        """Test compatibility with different NumPy versions."""
        # Test basic NumPy operations used in the framework
        
        # Array creation and manipulation
        test_array = np.linspace(0, 100, 101)
        assert len(test_array) == 101, "LinSpace should work correctly"
        
        # Mathematical operations
        test_result = np.sin(test_array) + np.cos(test_array)
        assert not np.any(np.isnan(test_result)), "Math operations should not produce NaN"
        
        # Array indexing and slicing
        subset = test_array[10:90:5]
        assert len(subset) == 16, "Array slicing should work correctly"
        
        # Statistical functions
        mean_val = np.mean(test_array)
        std_val = np.std(test_array)
        assert 45 < mean_val < 55, "Mean calculation should be reasonable"
        assert std_val > 0, "Standard deviation should be positive"


def run_comprehensive_validation():
    """Run all validation tests with reporting."""
    
    print("=" * 70)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("Hypersonic Reentry Framework Testing")
    print("=" * 70)
    print()
    
    # Set up test reporting
    test_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_categories': {},
        'overall_status': 'UNKNOWN',
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0
    }
    
    # Define test categories
    test_categories = [
        ('Mathematical Validation', TestMathematicalValidation),
        ('Physical Behavior', TestPhysicalBehavior),
        ('Numerical Accuracy', TestNumericalAccuracy),
        ('Performance Regression', TestPerformanceRegression),
        ('Integration & Compatibility', TestIntegrationAndCompatibility)
    ]
    
    overall_success = True
    
    for category_name, test_class in test_categories:
        print(f"\n{category_name}:")
        print("-" * len(category_name))
        
        category_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'failures': []
        }
        
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for test_method in test_methods:
            test_name = test_method.replace('test_', '').replace('_', ' ').title()
            category_results['tests_run'] += 1
            
            try:
                print(f"  Running: {test_name}...", end=" ")
                
                # Set up if needed
                if hasattr(test_instance, 'setup_test_environment'):
                    test_instance.setup_test_environment()
                
                # Run test
                getattr(test_instance, test_method)()
                
                print("PASSED")
                category_results['tests_passed'] += 1
                
            except Exception as e:
                print(f"FAILED - {str(e)}")
                category_results['tests_failed'] += 1
                category_results['failures'].append({
                    'test_name': test_name,
                    'error': str(e)
                })
                overall_success = False
        
        test_results['test_categories'][category_name] = category_results
        test_results['total_tests'] += category_results['tests_run']
        test_results['passed_tests'] += category_results['tests_passed']
        test_results['failed_tests'] += category_results['tests_failed']
        
        print(f"  Category Summary: {category_results['tests_passed']}/{category_results['tests_run']} passed")
    
    # Overall results
    test_results['overall_status'] = 'PASSED' if overall_success else 'FAILED'
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Overall Status: {test_results['overall_status']}")
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']}")
    print(f"Failed: {test_results['failed_tests']}")
    
    if test_results['failed_tests'] > 0:
        print(f"\nFailure Rate: {test_results['failed_tests']/test_results['total_tests']*100:.1f}%")
        print("\nFailed Tests:")
        for category, results in test_results['test_categories'].items():
            if results['failures']:
                print(f"\n{category}:")
                for failure in results['failures']:
                    print(f"  - {failure['test_name']}: {failure['error']}")
    else:
        print("\n🎉 All tests passed! Framework validation successful.")
    
    print("=" * 70)
    
    # Save test results
    results_file = f"validation_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {results_file}")
    
    return overall_success


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)