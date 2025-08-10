"""Basic functionality tests for hypersonic reentry framework."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hypersonic_reentry.dynamics.vehicle_dynamics import VehicleDynamics, VehicleState
from hypersonic_reentry.atmosphere.us_standard_1976 import USStandard1976
from hypersonic_reentry.utils.constants import DEG_TO_RAD


class TestVehicleDynamics:
    """Test vehicle dynamics functionality."""
    
    def test_vehicle_state_creation(self):
        """Test VehicleState creation."""
        state = VehicleState(
            altitude=120000.0,
            latitude=28.5 * DEG_TO_RAD,
            longitude=-80.6 * DEG_TO_RAD,
            velocity=7800.0,
            flight_path_angle=-1.5 * DEG_TO_RAD,
            azimuth=90.0 * DEG_TO_RAD
        )
        
        assert state.altitude == 120000.0
        assert abs(state.latitude - 28.5 * DEG_TO_RAD) < 1e-10
        assert state.velocity == 7800.0
    
    def test_vehicle_dynamics_creation(self):
        """Test VehicleDynamics creation."""
        vehicle_params = {
            'mass': 5000.0,
            'reference_area': 15.0,
            'drag_coefficient': 1.2,
            'lift_coefficient': 0.8,
            'ballistic_coefficient': 400.0,
            'nose_radius': 0.5
        }
        
        dynamics = VehicleDynamics(vehicle_params)
        
        assert dynamics.mass == 5000.0
        assert dynamics.reference_area == 15.0
        assert dynamics.drag_coefficient == 1.2


class TestAtmosphere:
    """Test atmosphere model functionality."""
    
    def test_us_standard_atmosphere(self):
        """Test US Standard Atmosphere 1976."""
        atmosphere = USStandard1976()
        
        # Test sea level conditions
        props_sl = atmosphere.get_properties(0.0)
        assert abs(props_sl['temperature'] - 288.15) < 1.0  # Sea level temperature
        assert abs(props_sl['pressure'] - 101325.0) < 1000.0  # Sea level pressure
        
        # Test high altitude
        props_high = atmosphere.get_properties(50000.0)
        assert props_high['temperature'] < props_sl['temperature']  # Colder at altitude
        assert props_high['pressure'] < props_sl['pressure']  # Lower pressure
        assert props_high['density'] < props_sl['density']  # Lower density


class TestIntegration:
    """Test integrated functionality."""
    
    def test_basic_trajectory_simulation(self):
        """Test basic trajectory integration."""
        # Set up vehicle
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
        
        # Initial state
        initial_state = VehicleState(
            altitude=80000.0,  # Lower altitude for faster simulation
            latitude=28.5 * DEG_TO_RAD,
            longitude=-80.6 * DEG_TO_RAD,
            velocity=6000.0,  # Lower velocity
            flight_path_angle=-5.0 * DEG_TO_RAD,
            azimuth=90.0 * DEG_TO_RAD
        )
        
        # Short simulation
        time_span = (0.0, 100.0)  # 100 seconds
        
        try:
            trajectory = dynamics.integrate_trajectory(
                initial_state, time_span, time_step=1.0
            )
            
            # Basic checks
            assert 'time' in trajectory
            assert 'altitude' in trajectory
            assert 'velocity' in trajectory
            assert len(trajectory['time']) > 10  # Should have some data points
            
            # Vehicle should descend
            assert trajectory['altitude'][-1] < trajectory['altitude'][0]
            
            # Calculate performance metrics
            performance = dynamics.calculate_performance_metrics(trajectory)
            
            assert 'final_altitude' in performance
            assert 'final_velocity' in performance
            assert 'flight_time' in performance
            
            print(f"Test trajectory completed:")
            print(f"  Initial altitude: {initial_state.altitude/1000:.1f} km")
            print(f"  Final altitude: {performance['final_altitude']/1000:.1f} km")
            print(f"  Flight time: {performance['flight_time']:.1f} s")
            
        except Exception as e:
            pytest.fail(f"Basic trajectory simulation failed: {str(e)}")


if __name__ == "__main__":
    # Run basic tests
    test_dynamics = TestVehicleDynamics()
    test_atmosphere = TestAtmosphere()
    test_integration = TestIntegration()
    
    print("Running basic functionality tests...")
    
    try:
        print("\n1. Testing VehicleState creation...")
        test_dynamics.test_vehicle_state_creation()
        print("   ✓ Passed")
        
        print("\n2. Testing VehicleDynamics creation...")
        test_dynamics.test_vehicle_dynamics_creation()
        print("   ✓ Passed")
        
        print("\n3. Testing US Standard Atmosphere...")
        test_atmosphere.test_us_standard_atmosphere()
        print("   ✓ Passed")
        
        print("\n4. Testing basic trajectory simulation...")
        test_integration.test_basic_trajectory_simulation()
        print("   ✓ Passed")
        
        print("\n" + "="*50)
        print("All basic tests passed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)