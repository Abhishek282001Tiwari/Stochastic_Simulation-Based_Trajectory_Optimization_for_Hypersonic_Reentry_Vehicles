"""Aerodynamics model for hypersonic reentry vehicles.

This module implements aerodynamic force and moment calculations including:
- Drag and lift coefficient models
- Heat transfer calculations
- Pressure distribution modeling
- Hypersonic flow effects
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from ..utils.constants import SPECIFIC_HEAT_RATIO


@dataclass
class AerodynamicCoefficients:
    """Aerodynamic coefficients for drag, lift, and moments."""
    
    drag: float
    lift: float
    moment_pitch: float = 0.0
    moment_yaw: float = 0.0
    moment_roll: float = 0.0


class AerodynamicsModel:
    """Aerodynamics model for hypersonic reentry vehicles.
    
    Implements various methods for calculating aerodynamic forces and moments
    including simplified coefficient models and more detailed hypersonic correlations.
    """
    
    def __init__(self, vehicle_params: Dict[str, float]):
        """Initialize aerodynamics model.
        
        Args:
            vehicle_params: Dictionary containing vehicle geometric and aerodynamic parameters
        """
        self.logger = logging.getLogger(__name__)
        
        # Vehicle parameters
        self.reference_area = vehicle_params['reference_area']  # m^2
        self.drag_coefficient = vehicle_params['drag_coefficient']
        self.lift_coefficient = vehicle_params['lift_coefficient']
        self.nose_radius = vehicle_params.get('nose_radius', 0.5)  # m
        
        # Aerodynamic model parameters
        self.use_hypersonic_correlations = vehicle_params.get('use_hypersonic_correlations', True)
        self.vehicle_length = vehicle_params.get('length', 10.0)  # m
        self.vehicle_diameter = vehicle_params.get('diameter', 2.0)  # m
        
        self.logger.info("Initialized aerodynamics model")
    
    def calculate_forces(self, 
                        velocity: float,
                        density: float, 
                        angle_of_attack: float,
                        vehicle_state: Optional[object] = None) -> Dict[str, float]:
        """Calculate aerodynamic forces acting on the vehicle.
        
        Args:
            velocity: Vehicle velocity magnitude (m/s)
            density: Atmospheric density (kg/m^3)
            angle_of_attack: Angle of attack in radians
            vehicle_state: Optional vehicle state for advanced calculations
            
        Returns:
            Dictionary containing drag and lift forces in Newtons
        """
        # Calculate dynamic pressure
        dynamic_pressure = 0.5 * density * velocity**2  # Pa
        
        # Get aerodynamic coefficients
        if self.use_hypersonic_correlations:
            coeffs = self._hypersonic_coefficients(velocity, density, angle_of_attack, vehicle_state)
        else:
            coeffs = self._constant_coefficients(angle_of_attack)
        
        # Calculate forces
        drag_force = coeffs.drag * dynamic_pressure * self.reference_area  # N
        lift_force = coeffs.lift * dynamic_pressure * self.reference_area  # N
        
        return {
            'drag': drag_force,
            'lift': lift_force,
            'dynamic_pressure': dynamic_pressure,
            'coefficients': coeffs
        }
    
    def _constant_coefficients(self, angle_of_attack: float) -> AerodynamicCoefficients:
        """Calculate constant aerodynamic coefficients.
        
        Args:
            angle_of_attack: Angle of attack in radians
            
        Returns:
            AerodynamicCoefficients object
        """
        # Simple constant coefficient model
        cd = self.drag_coefficient
        
        # Lift coefficient varies with angle of attack
        cl = self.lift_coefficient * np.sin(2 * angle_of_attack)
        
        return AerodynamicCoefficients(drag=cd, lift=cl)
    
    def _hypersonic_coefficients(self, 
                                velocity: float,
                                density: float,
                                angle_of_attack: float,
                                vehicle_state: Optional[object] = None) -> AerodynamicCoefficients:
        """Calculate hypersonic aerodynamic coefficients using correlations.
        
        Args:
            velocity: Vehicle velocity (m/s)
            density: Atmospheric density (kg/m^3)  
            angle_of_attack: Angle of attack in radians
            vehicle_state: Vehicle state for altitude-dependent calculations
            
        Returns:
            AerodynamicCoefficients object
        """
        # Calculate Mach number (approximate)
        altitude = getattr(vehicle_state, 'altitude', 50000.0) if vehicle_state else 50000.0
        temperature = self._estimate_temperature(altitude)
        speed_of_sound = np.sqrt(SPECIFIC_HEAT_RATIO * 287.0 * temperature)
        mach_number = velocity / speed_of_sound
        
        # Modified Newtonian theory for hypersonic drag
        cd_base = 2.0  # Base drag coefficient for blunt body
        cd_alpha = 0.5 * np.sin(angle_of_attack)**2  # Additional drag due to angle of attack
        cd_total = cd_base + cd_alpha
        
        # Hypersonic lift coefficient (linearized theory)
        cl_alpha = 4.0 / np.sqrt(mach_number**2 - 1) if mach_number > 1.0 else 2.0 * np.pi
        cl_total = cl_alpha * angle_of_attack
        
        # Mach number corrections for high-speed flow
        if mach_number > 5.0:
            # High Mach number corrections
            mach_factor = np.sqrt(mach_number)
            cd_total *= (1.0 + 0.1 * mach_factor)
            cl_total *= (1.0 - 0.05 * mach_factor)
        
        return AerodynamicCoefficients(drag=cd_total, lift=cl_total)
    
    def calculate_heating(self, 
                         velocity: float, 
                         density: float,
                         altitude: float) -> Dict[str, float]:
        """Calculate aerodynamic heating using established correlations.
        
        Args:
            velocity: Vehicle velocity (m/s)
            density: Atmospheric density (kg/m^3)
            altitude: Altitude (m)
            
        Returns:
            Dictionary containing heating rates and heat loads
        """
        # Fay-Riddell stagnation point heating correlation
        # q_dot = C * sqrt(rho/R_n) * V^3.15
        C_fay_riddell = 1.7415e-4  # W/(m^2 * Pa^0.5 * (m/s)^3.15)
        
        stagnation_heating = (C_fay_riddell * 
                            np.sqrt(density / self.nose_radius) * 
                            velocity**3.15)  # W/m^2
        
        # Detra-Kemp-Riddell correlation for distributed heating
        # Accounts for surface area effects
        average_heating = stagnation_heating * 0.4  # Approximate factor for average heating
        
        # Sutton-Graves correlation (alternative approach)
        # q_dot = K * sqrt(rho) * V^3 / sqrt(R_n)
        K_sutton_graves = 1.83e-4  # kg^0.5 * m^-2.5 * s^-2
        
        sutton_graves_heating = (K_sutton_graves * 
                               np.sqrt(density) * 
                               velocity**3 / 
                               np.sqrt(self.nose_radius))  # W/m^2
        
        # Calculate total heat load on reference area
        total_heat_rate = average_heating * self.reference_area  # W
        
        return {
            'stagnation_heating': stagnation_heating,
            'average_heating': average_heating,
            'sutton_graves_heating': sutton_graves_heating,
            'total_heat_rate': total_heat_rate,
            'heat_flux_density': average_heating
        }
    
    def calculate_pressure_distribution(self, 
                                      velocity: float,
                                      density: float,
                                      angle_of_attack: float) -> Dict[str, np.ndarray]:
        """Calculate pressure distribution over vehicle surface.
        
        Args:
            velocity: Vehicle velocity (m/s)
            density: Atmospheric density (kg/m^3)
            angle_of_attack: Angle of attack (radians)
            
        Returns:
            Dictionary containing pressure distribution data
        """
        # Define surface coordinate system
        n_points = 50  # Number of surface points
        theta = np.linspace(0, np.pi, n_points)  # Angle from stagnation point
        
        # Dynamic pressure
        q_inf = 0.5 * density * velocity**2
        
        # Modified Newtonian pressure distribution
        # P = P_stag * cos^2(theta) + P_inf
        pressure_coefficient = 2.0 * np.cos(theta)**2
        pressure = pressure_coefficient * q_inf  # Gauge pressure
        
        # Account for angle of attack effects
        # Shift pressure distribution based on alpha
        theta_shifted = theta - angle_of_attack
        pressure_alpha = 2.0 * np.maximum(0, np.cos(theta_shifted))**2 * q_inf
        
        return {
            'surface_angle': theta,
            'pressure_coefficient': pressure_coefficient,
            'pressure': pressure,
            'pressure_with_alpha': pressure_alpha,
            'dynamic_pressure': q_inf
        }
    
    def calculate_skin_friction(self, 
                              velocity: float,
                              density: float, 
                              temperature: float,
                              surface_area: float) -> float:
        """Calculate skin friction drag using flat plate approximation.
        
        Args:
            velocity: Vehicle velocity (m/s)
            density: Atmospheric density (kg/m^3)
            temperature: Surface temperature (K)
            surface_area: Vehicle surface area (m^2)
            
        Returns:
            Skin friction drag force (N)
        """
        # Calculate Reynolds number
        # Use Sutherland's law for viscosity
        mu_ref = 1.716e-5  # Pa*s, reference viscosity at 273K
        T_ref = 273.0  # K
        S = 110.4  # K, Sutherland constant
        
        viscosity = mu_ref * (temperature / T_ref)**1.5 * (T_ref + S) / (temperature + S)
        
        # Reynolds number based on vehicle length
        reynolds_number = density * velocity * self.vehicle_length / viscosity
        
        # Skin friction coefficient (turbulent flat plate)
        if reynolds_number > 1e6:
            cf = 0.074 / reynolds_number**0.2  # Turbulent
        else:
            cf = 1.33 / np.sqrt(reynolds_number)  # Laminar
        
        # Skin friction drag
        dynamic_pressure = 0.5 * density * velocity**2
        skin_friction_drag = cf * dynamic_pressure * surface_area
        
        return skin_friction_drag
    
    def _estimate_temperature(self, altitude: float) -> float:
        """Estimate atmospheric temperature at given altitude.
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Temperature in Kelvin
        """
        # Simple atmospheric temperature model (US Standard Atmosphere)
        if altitude <= 11000:
            # Troposphere
            T = 288.15 - 0.0065 * altitude
        elif altitude <= 20000:
            # Lower stratosphere
            T = 216.65
        elif altitude <= 32000:
            # Upper stratosphere  
            T = 216.65 + 0.001 * (altitude - 20000)
        else:
            # Simplified for higher altitudes
            T = 228.65 + 0.0028 * (altitude - 32000)
        
        return max(T, 180.0)  # Minimum temperature constraint
    
    def calculate_center_of_pressure(self, 
                                   angle_of_attack: float,
                                   mach_number: float) -> float:
        """Calculate center of pressure location.
        
        Args:
            angle_of_attack: Angle of attack (radians)
            mach_number: Mach number
            
        Returns:
            Center of pressure location from nose (fraction of vehicle length)
        """
        # Simplified center of pressure model
        # Moves aft with increasing angle of attack and Mach number
        
        cp_base = 0.6  # Base center of pressure location
        alpha_effect = 0.1 * np.abs(angle_of_attack)  # Effect of angle of attack
        mach_effect = 0.05 * (mach_number - 1.0) if mach_number > 1.0 else 0.0
        
        cp_location = cp_base + alpha_effect - mach_effect
        
        # Constrain to reasonable bounds
        return np.clip(cp_location, 0.4, 0.8)
    
    def calculate_aerodynamic_moments(self, 
                                    aerodynamic_forces: Dict[str, float],
                                    center_of_pressure: float,
                                    center_of_gravity: float = 0.5) -> Dict[str, float]:
        """Calculate aerodynamic moments about center of gravity.
        
        Args:
            aerodynamic_forces: Dictionary containing drag and lift forces
            center_of_pressure: Center of pressure location (fraction of length)
            center_of_gravity: Center of gravity location (fraction of length)
            
        Returns:
            Dictionary containing pitch, yaw, and roll moments
        """
        # Moment arm
        moment_arm = (center_of_pressure - center_of_gravity) * self.vehicle_length
        
        # Pitching moment (positive nose up)
        pitch_moment = aerodynamic_forces['lift'] * moment_arm
        
        # For symmetric vehicle, yaw and roll moments are typically small
        yaw_moment = 0.0
        roll_moment = 0.0
        
        return {
            'pitch_moment': pitch_moment,
            'yaw_moment': yaw_moment, 
            'roll_moment': roll_moment,
            'moment_arm': moment_arm
        }