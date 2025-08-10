"""US Standard Atmosphere 1976 model implementation."""

import numpy as np
from typing import Dict
from .atmosphere_model import AtmosphereModel
from ..utils.constants import SEA_LEVEL_PRESSURE, SEA_LEVEL_TEMPERATURE, GAS_CONSTANT


class USStandard1976(AtmosphereModel):
    """US Standard Atmosphere 1976 model implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Atmospheric layers (altitude in meters, lapse rate in K/m)
        self.layers = [
            (0, 11000, -0.0065),      # Troposphere
            (11000, 20000, 0.0),      # Tropopause
            (20000, 32000, 0.001),    # Stratosphere 1
            (32000, 47000, 0.0028),   # Stratosphere 2
            (47000, 51000, 0.0),      # Stratopause
            (51000, 71000, -0.0028),  # Mesosphere 1
            (71000, 84852, -0.002),   # Mesosphere 2
        ]
    
    def get_properties(self, altitude: float, latitude: float = 0.0, 
                      longitude: float = 0.0, time: float = 0.0) -> Dict[str, float]:
        """Get atmospheric properties using US Standard Atmosphere 1976."""
        
        # Validate inputs
        longitude = self.validate_inputs(altitude, latitude, longitude)
        
        # Calculate base properties
        temperature = self._calculate_temperature(altitude)
        pressure = self._calculate_pressure(altitude, temperature)
        density = pressure / (GAS_CONSTANT * temperature)
        
        # Calculate derived properties
        derived = self.calculate_derived_properties(density, temperature, pressure)
        
        # Get wind conditions
        wind = self.get_wind_model(altitude, latitude, longitude, time)
        
        # Combine all properties
        properties = {
            'altitude': altitude,
            'temperature': temperature,
            'pressure': pressure,
            'density': density,
            **derived,
            **wind
        }
        
        return properties
    
    def _calculate_temperature(self, altitude: float) -> float:
        """Calculate temperature at given altitude."""
        
        if altitude < 0:
            altitude = 0
        elif altitude > 84852:
            altitude = 84852
        
        # Find appropriate layer
        for i, (h_low, h_high, lapse_rate) in enumerate(self.layers):
            if h_low <= altitude <= h_high:
                if i == 0:
                    # First layer - start from sea level
                    T_base = SEA_LEVEL_TEMPERATURE
                    h_base = 0
                else:
                    # Calculate base temperature from previous layers
                    T_base = SEA_LEVEL_TEMPERATURE
                    h_cumulative = 0
                    
                    for j in range(i):
                        h_start, h_end, lapse = self.layers[j]
                        layer_height = h_end - h_start
                        T_base += lapse * layer_height
                        h_cumulative = h_end
                    
                    h_base = h_cumulative
                
                # Calculate temperature in current layer
                temperature = T_base + lapse_rate * (altitude - h_base)
                return max(temperature, 180.0)  # Minimum temperature constraint
        
        # If altitude is above all layers, use last layer temperature
        return 180.0
    
    def _calculate_pressure(self, altitude: float, temperature: float) -> float:
        """Calculate pressure using hydrostatic equation."""
        
        if altitude <= 0:
            return SEA_LEVEL_PRESSURE
        
        pressure = SEA_LEVEL_PRESSURE
        h_current = 0
        T_current = SEA_LEVEL_TEMPERATURE
        
        for h_low, h_high, lapse_rate in self.layers:
            if altitude <= h_low:
                break
            
            # Height increment for this layer
            h_layer = min(altitude, h_high) - max(h_current, h_low)
            
            if h_layer > 0:
                if abs(lapse_rate) < 1e-10:  # Isothermal layer
                    pressure *= np.exp(-9.80665 * h_layer / (GAS_CONSTANT * T_current))
                else:  # Linear temperature gradient
                    T_ratio = (T_current + lapse_rate * h_layer) / T_current
                    pressure *= T_ratio ** (-9.80665 / (GAS_CONSTANT * lapse_rate))
                    T_current += lapse_rate * h_layer
            
            h_current = max(h_current, h_high)
            
            if altitude <= h_high:
                break
        
        return max(pressure, 1e-6)  # Minimum pressure constraint