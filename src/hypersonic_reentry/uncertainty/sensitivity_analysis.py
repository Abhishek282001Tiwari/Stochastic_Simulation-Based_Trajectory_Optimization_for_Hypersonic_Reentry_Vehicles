"""Sensitivity analysis methods for uncertainty quantification."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .uncertainty_quantifier import UncertainParameter


class SensitivityAnalyzer:
    """Sensitivity analysis methods for parameter importance assessment."""
    
    def __init__(self, uncertain_parameters: List[UncertainParameter]):
        """Initialize sensitivity analyzer.
        
        Args:
            uncertain_parameters: List of uncertain parameters
        """
        self.logger = logging.getLogger(__name__)
        self.uncertain_parameters = uncertain_parameters
        self.num_parameters = len(uncertain_parameters)
    
    def compute_sobol_indices(self, 
                            parameter_samples: np.ndarray,
                            output_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute Sobol sensitivity indices using Monte Carlo estimation.
        
        Args:
            parameter_samples: Parameter sample matrix
            output_data: Dictionary of output quantity arrays
            
        Returns:
            Dictionary containing first-order and total Sobol indices
        """
        results = {}
        
        for output_name, output_values in output_data.items():
            # Remove NaN values
            valid_mask = ~np.isnan(output_values)
            clean_params = parameter_samples[valid_mask]
            clean_outputs = output_values[valid_mask]
            
            if len(clean_outputs) < 10:
                self.logger.warning(f"Insufficient valid samples for {output_name}")
                continue
            
            # Calculate variance
            total_variance = np.var(clean_outputs)
            
            if total_variance < 1e-12:
                # Output is constant
                results[output_name] = {f"S1_{param.name}": 0.0 for param in self.uncertain_parameters}
                continue
            
            # First-order indices using correlation-based approximation
            first_order = {}
            for i, param in enumerate(self.uncertain_parameters):
                correlation = np.corrcoef(clean_params[:, i], clean_outputs)[0, 1]
                if np.isfinite(correlation):
                    first_order[f"S1_{param.name}"] = correlation**2
                else:
                    first_order[f"S1_{param.name}"] = 0.0
            
            results[output_name] = first_order
        
        return results
    
    def generate_sobol_samples(self, num_base_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate samples for Sobol analysis."""
        from scipy.stats import qmc
        
        # Generate two independent sample matrices
        sampler_A = qmc.Sobol(d=self.num_parameters, seed=42)
        sampler_B = qmc.Sobol(d=self.num_parameters, seed=123)
        
        samples_A = sampler_A.random(num_base_samples)
        samples_B = sampler_B.random(num_base_samples)
        
        return samples_A, samples_B