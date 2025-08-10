"""Vehicle dynamics model for hypersonic reentry vehicles.

This module implements the 3-DOF point mass equations of motion for a hypersonic
reentry vehicle operating in the Earth's atmosphere with gravitational and
aerodynamic forces.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

from ..utils.constants import EARTH_RADIUS, GRAVITATIONAL_PARAMETER
from .coordinate_transforms import CoordinateTransforms
from .aerodynamics import AerodynamicsModel
from ..atmosphere import AtmosphereModel


@dataclass
class VehicleState:
    """Vehicle state vector containing position, velocity, and orientation."""
    
    # Position components (m)
    altitude: float
    latitude: float  # radians
    longitude: float  # radians
    
    # Velocity components (m/s)
    velocity: float
    flight_path_angle: float  # radians
    azimuth: float  # radians
    
    # Control inputs (radians)
    bank_angle: float = 0.0
    angle_of_attack: float = 0.0
    
    # Time (seconds)
    time: float = 0.0


class VehicleDynamics:
    """3-DOF point mass dynamics model for hypersonic reentry vehicles.
    
    This class implements the differential equations for vehicle motion including:
    - Gravitational forces with Earth's oblateness effects
    - Aerodynamic drag and lift forces
    - Atmospheric density variations
    - Coordinate system transformations
    """
    
    def __init__(self, 
                 vehicle_params: Dict[str, float],
                 atmosphere_model: Optional[AtmosphereModel] = None,
                 aerodynamics_model: Optional[AerodynamicsModel] = None):
        """Initialize vehicle dynamics model.
        
        Args:
            vehicle_params: Dictionary containing vehicle mass, reference area,
                          drag coefficient, lift coefficient, etc.
            atmosphere_model: Atmospheric model for density/temperature calculations
            aerodynamics_model: Aerodynamics model for force/moment calculations
        """
        self.logger = logging.getLogger(__name__)
        
        # Vehicle parameters
        self.mass = vehicle_params['mass']  # kg
        self.reference_area = vehicle_params['reference_area']  # m^2
        self.drag_coefficient = vehicle_params['drag_coefficient']
        self.lift_coefficient = vehicle_params['lift_coefficient']
        self.ballistic_coefficient = vehicle_params['ballistic_coefficient']  # kg/m^2
        
        # Initialize models
        self.atmosphere = atmosphere_model or AtmosphereModel()
        self.aerodynamics = aerodynamics_model or AerodynamicsModel(vehicle_params)
        self.coordinate_transforms = CoordinateTransforms()
        
        self.logger.info(f"Initialized vehicle dynamics with mass={self.mass} kg")
    
    def equations_of_motion(self, 
                          state: VehicleState, 
                          control_inputs: Optional[Dict[str, float]] = None) -> VehicleState:
        """Compute derivatives of state variables (equations of motion).
        
        Args:
            state: Current vehicle state
            control_inputs: Optional control inputs (bank angle, angle of attack)
            
        Returns:
            VehicleState object containing time derivatives of state variables
        """
        # Extract state variables for clarity
        h = state.altitude  # altitude above Earth surface (m)
        lat = state.latitude  # latitude (rad)
        lon = state.longitude  # longitude (rad)
        V = state.velocity  # velocity magnitude (m/s)
        gamma = state.flight_path_angle  # flight path angle (rad)
        psi = state.azimuth  # azimuth angle (rad)
        
        # Control inputs
        sigma = state.bank_angle if control_inputs is None else control_inputs.get('bank_angle', 0.0)
        alpha = state.angle_of_attack if control_inputs is None else control_inputs.get('angle_of_attack', 0.0)
        
        # Calculate geocentric radius and gravitational acceleration
        r = EARTH_RADIUS + h  # geocentric radius (m)
        g = GRAVITATIONAL_PARAMETER / (r**2)  # gravitational acceleration (m/s^2)
        
        # Get atmospheric properties
        atm_props = self.atmosphere.get_properties(h, lat, lon)
        rho = atm_props['density']  # atmospheric density (kg/m^3)
        
        # Calculate aerodynamic forces
        aero_forces = self.aerodynamics.calculate_forces(V, rho, alpha, state)
        D = aero_forces['drag']  # drag force (N)
        L = aero_forces['lift']  # lift force (N)
        
        # Specific forces (force per unit mass)
        drag_accel = D / self.mass  # drag acceleration (m/s^2)
        lift_accel = L / self.mass  # lift acceleration (m/s^2)
        
        # Earth rotation effects
        omega_earth = 7.2921159e-5  # Earth's rotation rate (rad/s)
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        
        # Equations of motion in spherical coordinates
        # Altitude rate
        h_dot = V * np.sin(gamma)
        
        # Latitude rate
        lat_dot = (V * np.cos(gamma) * np.cos(psi)) / r
        
        # Longitude rate
        lon_dot = (V * np.cos(gamma) * np.sin(psi)) / (r * cos_lat)
        
        # Velocity magnitude rate
        V_dot = -drag_accel - g * np.sin(gamma) + omega_earth**2 * r * cos_lat * (
            sin_lat * np.cos(gamma) - cos_lat * np.sin(gamma) * np.cos(psi)
        )
        
        # Flight path angle rate
        gamma_dot = (lift_accel * np.cos(sigma) / V) - (g / V) * np.cos(gamma) + (V / r) * np.cos(gamma) + (
            2 * omega_earth * cos_lat * np.sin(psi) + 
            omega_earth**2 * r * cos_lat * (cos_lat * np.cos(gamma) + sin_lat * np.sin(gamma) * np.cos(psi)) / V
        )
        
        # Azimuth angle rate
        psi_dot = (lift_accel * np.sin(sigma)) / (V * np.cos(gamma)) + (V / r) * np.cos(gamma) * np.sin(psi) * np.tan(lat) - (
            2 * omega_earth * (sin_lat - cos_lat * np.cos(psi) * np.tan(gamma)) + 
            omega_earth**2 * r * cos_lat * sin_lat * np.sin(psi) / (V * np.cos(gamma))
        )
        
        # Return state derivatives
        return VehicleState(
            altitude=h_dot,
            latitude=lat_dot,
            longitude=lon_dot,
            velocity=V_dot,
            flight_path_angle=gamma_dot,
            azimuth=psi_dot,
            bank_angle=0.0,  # Control input, not a state derivative
            angle_of_attack=0.0,  # Control input, not a state derivative
            time=1.0  # dt/dt = 1
        )
    
    def integrate_trajectory(self, 
                           initial_state: VehicleState,
                           time_span: Tuple[float, float],
                           control_history: Optional[Dict[str, np.ndarray]] = None,
                           time_step: float = 0.1) -> Dict[str, np.ndarray]:
        """Integrate vehicle trajectory over specified time span.
        
        Args:
            initial_state: Initial vehicle state
            time_span: (start_time, end_time) in seconds
            control_history: Dictionary of control input time histories
            time_step: Integration time step in seconds
            
        Returns:
            Dictionary containing trajectory data arrays
        """
        from scipy.integrate import solve_ivp
        
        t_start, t_end = time_span
        t_eval = np.arange(t_start, t_end + time_step, time_step)
        
        # Convert initial state to vector
        y0 = np.array([
            initial_state.altitude,
            initial_state.latitude,
            initial_state.longitude,
            initial_state.velocity,
            initial_state.flight_path_angle,
            initial_state.azimuth
        ])
        
        def dynamics_rhs(t, y):
            """Right-hand side of differential equations for numerical integration."""
            state = VehicleState(
                altitude=y[0],
                latitude=y[1],
                longitude=y[2],
                velocity=y[3],
                flight_path_angle=y[4],
                azimuth=y[5],
                time=t
            )
            
            # Get control inputs at current time
            controls = {}
            if control_history:
                for key, values in control_history.items():
                    # Interpolate control values at current time
                    controls[key] = np.interp(t, t_eval, values[:len(t_eval)])
            
            # Calculate derivatives
            state_dot = self.equations_of_motion(state, controls)
            
            return np.array([
                state_dot.altitude,
                state_dot.latitude,
                state_dot.longitude,
                state_dot.velocity,
                state_dot.flight_path_angle,
                state_dot.azimuth
            ])
        
        # Solve differential equations
        solution = solve_ivp(
            dynamics_rhs,
            time_span,
            y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        if not solution.success:
            self.logger.error(f"Integration failed: {solution.message}")
            raise RuntimeError(f"Trajectory integration failed: {solution.message}")
        
        # Package results
        trajectory = {
            'time': solution.t,
            'altitude': solution.y[0],
            'latitude': solution.y[1],
            'longitude': solution.y[2],
            'velocity': solution.y[3],
            'flight_path_angle': solution.y[4],
            'azimuth': solution.y[5]
        }
        
        # Calculate derived quantities
        trajectory['mach_number'] = self._calculate_mach_number(
            trajectory['velocity'], 
            trajectory['altitude']
        )
        trajectory['dynamic_pressure'] = self._calculate_dynamic_pressure(
            trajectory['velocity'], 
            trajectory['altitude']
        )
        trajectory['heat_rate'] = self._calculate_heat_rate(
            trajectory['velocity'], 
            trajectory['altitude']
        )
        
        self.logger.info(f"Successfully integrated trajectory over {t_end - t_start} seconds")
        
        return trajectory
    
    def _calculate_mach_number(self, velocity: np.ndarray, altitude: np.ndarray) -> np.ndarray:
        """Calculate Mach number along trajectory."""
        mach_numbers = np.zeros_like(velocity)
        
        for i, (V, h) in enumerate(zip(velocity, altitude)):
            atm_props = self.atmosphere.get_properties(h, 0.0, 0.0)
            speed_of_sound = np.sqrt(1.4 * 287.0 * atm_props['temperature'])  # m/s
            mach_numbers[i] = V / speed_of_sound
        
        return mach_numbers
    
    def _calculate_dynamic_pressure(self, velocity: np.ndarray, altitude: np.ndarray) -> np.ndarray:
        """Calculate dynamic pressure along trajectory."""
        dynamic_pressures = np.zeros_like(velocity)
        
        for i, (V, h) in enumerate(zip(velocity, altitude)):
            atm_props = self.atmosphere.get_properties(h, 0.0, 0.0)
            dynamic_pressures[i] = 0.5 * atm_props['density'] * V**2  # Pa
        
        return dynamic_pressures
    
    def _calculate_heat_rate(self, velocity: np.ndarray, altitude: np.ndarray) -> np.ndarray:
        """Calculate heat transfer rate using Fay-Riddell stagnation point heating."""
        heat_rates = np.zeros_like(velocity)
        
        # Fay-Riddell correlation constants
        C_h = 1.7415e-4  # W/(m^2 * Pa^0.5 * (m/s)^3.15)
        nose_radius = 0.5  # m, vehicle nose radius
        
        for i, (V, h) in enumerate(zip(velocity, altitude)):
            atm_props = self.atmosphere.get_properties(h, 0.0, 0.0)
            rho = atm_props['density']
            
            # Stagnation point heat transfer rate (W/m^2)
            heat_rates[i] = C_h * np.sqrt(rho / nose_radius) * V**3.15
        
        return heat_rates
    
    def calculate_performance_metrics(self, trajectory: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate trajectory performance metrics.
        
        Args:
            trajectory: Dictionary containing trajectory data
            
        Returns:
            Dictionary of performance metrics
        """
        # Final conditions
        final_altitude = trajectory['altitude'][-1]
        final_velocity = trajectory['velocity'][-1]
        flight_time = trajectory['time'][-1] - trajectory['time'][0]
        
        # Range calculations
        lat_rad = trajectory['latitude']
        lon_rad = trajectory['longitude']
        
        # Calculate downrange and crossrange distances
        r_earth = EARTH_RADIUS + trajectory['altitude']
        
        # Use spherical geometry for range calculations
        downrange = r_earth[-1] * np.arccos(
            np.sin(lat_rad[0]) * np.sin(lat_rad[-1]) + 
            np.cos(lat_rad[0]) * np.cos(lat_rad[-1]) * np.cos(lon_rad[-1] - lon_rad[0])
        )
        
        # Peak values
        max_mach = np.max(trajectory['mach_number'])
        max_dynamic_pressure = np.max(trajectory['dynamic_pressure'])
        max_heat_rate = np.max(trajectory['heat_rate'])
        
        # Integrated values
        total_heat_load = np.trapz(trajectory['heat_rate'], trajectory['time'])
        
        return {
            'final_altitude': final_altitude,
            'final_velocity': final_velocity,
            'flight_time': flight_time,
            'downrange': downrange,
            'max_mach_number': max_mach,
            'max_dynamic_pressure': max_dynamic_pressure,
            'max_heat_rate': max_heat_rate,
            'total_heat_load': total_heat_load
        }