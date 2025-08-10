"""Base atmospheric model for hypersonic reentry simulation.

This module provides the base class for atmospheric models used in trajectory
simulation, including density, temperature, and pressure calculations with
uncertainty quantification capabilities.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from abc import ABC, abstractmethod
import logging

from ..utils.constants import (
    SEA_LEVEL_PRESSURE, SEA_LEVEL_TEMPERATURE, SEA_LEVEL_DENSITY,
    GAS_CONSTANT, SPECIFIC_HEAT_RATIO
)


class AtmosphereModel(ABC):
    """Abstract base class for atmospheric models.
    
    Provides interface for calculating atmospheric properties including
    density, temperature, pressure, and wind conditions with optional
    uncertainty quantification.
    """
    
    def __init__(self, 
                 include_uncertainties: bool = False,
                 uncertainty_params: Optional[Dict[str, float]] = None):
        """Initialize atmospheric model.
        
        Args:
            include_uncertainties: Flag to enable uncertainty quantification
            uncertainty_params: Parameters for uncertainty models
        """
        self.logger = logging.getLogger(__name__)
        self.include_uncertainties = include_uncertainties
        self.uncertainty_params = uncertainty_params or {}
        
        # Default uncertainty parameters (standard deviations)
        self.default_uncertainties = {
            'density_uncertainty': 0.15,  # 15% standard deviation
            'temperature_uncertainty': 0.05,  # 5% standard deviation  
            'pressure_uncertainty': 0.10,  # 10% standard deviation
            'wind_uncertainty': 50.0  # 50 m/s standard deviation
        }
        
        # Merge with provided parameters
        self.uncertainties = {**self.default_uncertainties, **self.uncertainty_params}
        
        self.logger.info(f"Initialized atmosphere model with uncertainties: {include_uncertainties}")
    
    @abstractmethod
    def get_properties(self, 
                      altitude: float,
                      latitude: float = 0.0, 
                      longitude: float = 0.0,
                      time: float = 0.0) -> Dict[str, float]:
        """Get atmospheric properties at specified location and time.
        
        Args:
            altitude: Altitude above sea level in meters
            latitude: Latitude in radians  
            longitude: Longitude in radians
            time: Time in seconds
            
        Returns:
            Dictionary containing atmospheric properties
        """
        pass
    
    def get_properties_with_uncertainty(self, 
                                      altitude: float,
                                      latitude: float = 0.0,
                                      longitude: float = 0.0, 
                                      time: float = 0.0,
                                      num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Get atmospheric properties with uncertainty quantification.
        
        Args:
            altitude: Altitude above sea level in meters
            latitude: Latitude in radians
            longitude: Longitude in radians  
            time: Time in seconds
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary containing property arrays with uncertainties
        """
        if not self.include_uncertainties:
            # Return deterministic values repeated
            props = self.get_properties(altitude, latitude, longitude, time)
            return {key: np.full(num_samples, value) for key, value in props.items()}
        
        # Generate random samples for each property
        samples = {}
        
        # Get nominal properties
        nominal_props = self.get_properties(altitude, latitude, longitude, time)
        
        for prop_name, nominal_value in nominal_props.items():
            if prop_name in ['density', 'temperature', 'pressure']:
                # Lognormal distribution for positive quantities
                uncertainty_key = f"{prop_name}_uncertainty"
                if uncertainty_key in self.uncertainties:
                    sigma = self.uncertainties[uncertainty_key]
                    # Convert to lognormal parameters
                    mu = np.log(nominal_value) - 0.5 * sigma**2
                    samples[prop_name] = np.random.lognormal(mu, sigma, num_samples)
                else:
                    samples[prop_name] = np.full(num_samples, nominal_value)
            elif prop_name in ['wind_north', 'wind_east', 'wind_up']:
                # Normal distribution for wind components
                sigma = self.uncertainties.get('wind_uncertainty', 50.0)
                samples[prop_name] = np.random.normal(nominal_value, sigma, num_samples)
            else:
                # Default: no uncertainty
                samples[prop_name] = np.full(num_samples, nominal_value)
        
        return samples
    
    def calculate_derived_properties(self, 
                                   density: float,
                                   temperature: float, 
                                   pressure: float) -> Dict[str, float]:
        """Calculate derived atmospheric properties.
        
        Args:
            density: Atmospheric density (kg/m^3)
            temperature: Temperature (K)
            pressure: Pressure (Pa)
            
        Returns:
            Dictionary of derived properties
        """
        # Speed of sound
        speed_of_sound = np.sqrt(SPECIFIC_HEAT_RATIO * GAS_CONSTANT * temperature)
        
        # Dynamic viscosity using Sutherland's law
        mu_ref = 1.716e-5  # Pa*s at 273K
        T_ref = 273.0  # K
        S = 110.4  # K, Sutherland constant
        
        dynamic_viscosity = mu_ref * (temperature / T_ref)**1.5 * (T_ref + S) / (temperature + S)
        
        # Kinematic viscosity
        kinematic_viscosity = dynamic_viscosity / density
        
        # Thermal conductivity (approximate)
        thermal_conductivity = dynamic_viscosity * GAS_CONSTANT * SPECIFIC_HEAT_RATIO / (SPECIFIC_HEAT_RATIO - 1)
        
        # Scale height
        scale_height = GAS_CONSTANT * temperature / 9.81  # Using standard gravity
        
        return {
            'speed_of_sound': speed_of_sound,
            'dynamic_viscosity': dynamic_viscosity,
            'kinematic_viscosity': kinematic_viscosity,
            'thermal_conductivity': thermal_conductivity,
            'scale_height': scale_height
        }
    
    def get_wind_model(self, 
                      altitude: float,
                      latitude: float = 0.0,
                      longitude: float = 0.0,
                      time: float = 0.0) -> Dict[str, float]:
        """Get wind conditions at specified location.
        
        Args:
            altitude: Altitude in meters
            latitude: Latitude in radians
            longitude: Longitude in radians  
            time: Time in seconds
            
        Returns:
            Dictionary containing wind components (m/s)
        """
        # Simple wind model - can be overridden by subclasses
        # High altitude jet stream approximation
        
        if altitude < 10000:
            # Surface winds - minimal for reentry analysis
            wind_magnitude = 10.0  # m/s
        elif altitude < 15000:
            # Jet stream region
            wind_magnitude = 50.0 + 30.0 * np.sin(latitude * 4)  # Varies with latitude
        else:
            # High altitude - reduced winds
            wind_magnitude = 20.0 * np.exp(-(altitude - 15000) / 20000)
        
        # Predominant westerly winds at high altitudes
        wind_direction = np.pi / 2  # East direction
        
        wind_north = wind_magnitude * np.sin(wind_direction)
        wind_east = wind_magnitude * np.cos(wind_direction) 
        wind_up = 0.0  # Typically negligible vertical wind
        
        return {
            'wind_north': wind_north,
            'wind_east': wind_east,
            'wind_up': wind_up,
            'wind_magnitude': wind_magnitude,
            'wind_direction': wind_direction
        }
    
    def interpolate_altitude_profile(self, 
                                   altitudes: np.ndarray,
                                   target_altitude: float) -> Tuple[int, int, float]:
        """Find interpolation indices and weight for altitude profile.
        
        Args:
            altitudes: Array of altitude values (m)
            target_altitude: Target altitude for interpolation (m)
            
        Returns:
            Tuple of (lower_index, upper_index, interpolation_weight)
        """
        # Find bracketing altitudes
        if target_altitude <= altitudes[0]:
            return 0, 0, 0.0
        elif target_altitude >= altitudes[-1]:
            return len(altitudes)-1, len(altitudes)-1, 0.0
        
        # Find interpolation indices
        upper_idx = np.searchsorted(altitudes, target_altitude)
        lower_idx = upper_idx - 1
        
        # Calculate interpolation weight
        weight = (target_altitude - altitudes[lower_idx]) / (altitudes[upper_idx] - altitudes[lower_idx])
        
        return lower_idx, upper_idx, weight
    
    def validate_inputs(self, 
                       altitude: float,
                       latitude: float,
                       longitude: float) -> None:
        """Validate input parameters for atmospheric calculations.
        
        Args:
            altitude: Altitude in meters
            latitude: Latitude in radians
            longitude: Longitude in radians
            
        Raises:
            ValueError: If inputs are outside valid ranges
        """
        if altitude < -1000 or altitude > 1000000:
            raise ValueError(f"Altitude {altitude} m is outside valid range [-1000, 1000000] m")
        
        if not -np.pi/2 <= latitude <= np.pi/2:
            raise ValueError(f"Latitude {latitude} rad is outside valid range [-π/2, π/2]")
        
        if not -np.pi <= longitude <= np.pi:
            # Normalize longitude to [-π, π]
            longitude = np.mod(longitude + np.pi, 2*np.pi) - np.pi
        
        return longitude
    
    def get_seasonal_variation(self, 
                             latitude: float,
                             day_of_year: int) -> float:
        """Calculate seasonal temperature variation factor.
        
        Args:
            latitude: Latitude in radians
            day_of_year: Day of year (1-365)
            
        Returns:
            Temperature variation factor (multiplicative)
        """
        # Simple seasonal model
        # Maximum variation at solstices, minimum at equinoxes
        
        seasonal_amplitude = 0.1  # 10% maximum variation
        
        # Solar declination angle approximation
        declination = 23.45 * np.pi/180 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        
        # Seasonal factor based on latitude and solar angle
        seasonal_factor = 1.0 + seasonal_amplitude * np.sin(latitude) * np.sin(declination)
        
        return seasonal_factor
    
    def get_diurnal_variation(self, 
                            local_time: float,
                            altitude: float) -> float:
        """Calculate diurnal (daily) temperature variation.
        
        Args:
            local_time: Local time in hours (0-24)
            altitude: Altitude in meters
            
        Returns:
            Temperature variation factor (multiplicative)
        """
        # Diurnal variation decreases with altitude
        if altitude > 50000:
            return 1.0  # No diurnal variation at high altitudes
        
        # Maximum variation at surface, decreasing with altitude
        diurnal_amplitude = 0.05 * np.exp(-altitude / 10000)  # 5% max at surface
        
        # Peak temperature at 2 PM (14:00)
        phase_shift = 14.0  # hours
        
        diurnal_factor = 1.0 + diurnal_amplitude * np.sin(
            2 * np.pi * (local_time - phase_shift) / 24.0
        )
        
        return diurnal_factor