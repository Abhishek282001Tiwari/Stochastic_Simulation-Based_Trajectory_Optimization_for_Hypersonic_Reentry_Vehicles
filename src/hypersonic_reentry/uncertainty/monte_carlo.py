"""Monte Carlo sampling methods for uncertainty quantification.

This module provides Monte Carlo sampling techniques including:
- Latin Hypercube Sampling (LHS)
- Quasi-random sampling (Sobol sequences)  
- Standard random sampling
- Importance sampling methods
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from scipy.stats import qmc
import warnings

from .uncertainty_quantifier import UncertainParameter


class MonteCarloSampler:
    """Monte Carlo sampling methods for uncertainty quantification.
    
    Provides various sampling strategies for generating parameter samples
    from specified probability distributions.
    """
    
    def __init__(self, 
                 uncertain_parameters: List[UncertainParameter],
                 random_seed: Optional[int] = None):
        """Initialize Monte Carlo sampler.
        
        Args:
            uncertain_parameters: List of uncertain parameters
            random_seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.uncertain_parameters = uncertain_parameters
        self.num_parameters = len(uncertain_parameters)
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            self.random_seed = random_seed
        else:
            self.random_seed = None
        
        # Validate parameter definitions
        self._validate_parameters()
        
        self.logger.info(f"Initialized Monte Carlo sampler for {self.num_parameters} parameters")
    
    def _validate_parameters(self) -> None:
        """Validate uncertain parameter definitions."""
        for param in self.uncertain_parameters:
            if param.distribution_type not in ['normal', 'uniform', 'lognormal', 'beta', 'triangular']:
                raise ValueError(f"Unsupported distribution type: {param.distribution_type}")
            
            required_params = {
                'normal': ['mean', 'std'],
                'uniform': ['lower', 'upper'],
                'lognormal': ['mu', 'sigma'],
                'beta': ['alpha', 'beta'],
                'triangular': ['lower', 'mode', 'upper']
            }
            
            for req_param in required_params[param.distribution_type]:
                if req_param not in param.parameters:
                    raise ValueError(f"Missing parameter '{req_param}' for {param.distribution_type} distribution")
    
    def generate_samples(self, 
                        num_samples: int,
                        method: str = 'lhs') -> np.ndarray:
        """Generate parameter samples using specified method.
        
        Args:
            num_samples: Number of samples to generate
            method: Sampling method ('random', 'lhs', 'sobol', 'halton')
            
        Returns:
            Array of shape (num_samples, num_parameters) containing parameter samples
        """
        self.logger.info(f"Generating {num_samples} samples using {method} method")
        
        if method == 'random':
            return self._generate_random_samples(num_samples)
        elif method == 'lhs':
            return self._generate_lhs_samples(num_samples)
        elif method == 'sobol':
            return self._generate_sobol_samples(num_samples)
        elif method == 'halton':
            return self._generate_halton_samples(num_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _generate_random_samples(self, num_samples: int) -> np.ndarray:
        """Generate standard random samples."""
        samples = np.zeros((num_samples, self.num_parameters))
        
        for i, param in enumerate(self.uncertain_parameters):
            samples[:, i] = self._sample_from_distribution(param, num_samples)
        
        return samples
    
    def _generate_lhs_samples(self, num_samples: int) -> np.ndarray:
        """Generate Latin Hypercube Samples."""
        # Generate LHS samples in [0, 1]^d
        sampler = qmc.LatinHypercube(d=self.num_parameters, seed=self.random_seed)
        unit_samples = sampler.random(num_samples)
        
        # Transform to parameter distributions
        samples = np.zeros((num_samples, self.num_parameters))
        
        for i, param in enumerate(self.uncertain_parameters):
            samples[:, i] = self._transform_unit_samples(param, unit_samples[:, i])
        
        return samples
    
    def _generate_sobol_samples(self, num_samples: int) -> np.ndarray:
        """Generate Sobol sequence samples."""
        # Adjust sample size to nearest power of 2 for Sobol
        sobol_samples = int(2**np.ceil(np.log2(num_samples)))
        
        # Generate Sobol samples
        sampler = qmc.Sobol(d=self.num_parameters, seed=self.random_seed)
        unit_samples = sampler.random(sobol_samples)
        
        # Take first num_samples
        unit_samples = unit_samples[:num_samples, :]
        
        # Transform to parameter distributions
        samples = np.zeros((num_samples, self.num_parameters))
        
        for i, param in enumerate(self.uncertain_parameters):
            samples[:, i] = self._transform_unit_samples(param, unit_samples[:, i])
        
        return samples
    
    def _generate_halton_samples(self, num_samples: int) -> np.ndarray:
        """Generate Halton sequence samples."""
        # Generate Halton samples
        sampler = qmc.Halton(d=self.num_parameters, seed=self.random_seed)
        unit_samples = sampler.random(num_samples)
        
        # Transform to parameter distributions
        samples = np.zeros((num_samples, self.num_parameters))
        
        for i, param in enumerate(self.uncertain_parameters):
            samples[:, i] = self._transform_unit_samples(param, unit_samples[:, i])
        
        return samples
    
    def _sample_from_distribution(self, 
                                 param: UncertainParameter,
                                 num_samples: int) -> np.ndarray:
        """Sample from a specific distribution."""
        dist_type = param.distribution_type
        params = param.parameters
        
        if dist_type == 'normal':
            samples = np.random.normal(params['mean'], params['std'], num_samples)
        
        elif dist_type == 'uniform':
            samples = np.random.uniform(params['lower'], params['upper'], num_samples)
        
        elif dist_type == 'lognormal':
            samples = np.random.lognormal(params['mu'], params['sigma'], num_samples)
        
        elif dist_type == 'beta':
            samples = np.random.beta(params['alpha'], params['beta'], num_samples)
            # Scale to bounds if provided
            if param.bounds is not None:
                lower, upper = param.bounds
                samples = lower + (upper - lower) * samples
        
        elif dist_type == 'triangular':
            # Use scipy's triangular distribution
            mode = params['mode']
            lower = params['lower']
            upper = params['upper']
            
            # Convert to scipy parameterization
            c = (mode - lower) / (upper - lower)
            samples = stats.triang.rvs(c, loc=lower, scale=upper-lower, size=num_samples)
        
        else:
            raise ValueError(f"Unsupported distribution: {dist_type}")
        
        # Apply bounds if specified
        if param.bounds is not None and dist_type != 'beta':
            samples = np.clip(samples, param.bounds[0], param.bounds[1])
        
        return samples
    
    def _transform_unit_samples(self, 
                               param: UncertainParameter,
                               unit_samples: np.ndarray) -> np.ndarray:
        """Transform unit samples [0,1] to parameter distribution."""
        dist_type = param.distribution_type
        params = param.parameters
        
        if dist_type == 'normal':
            samples = stats.norm.ppf(unit_samples, params['mean'], params['std'])
        
        elif dist_type == 'uniform':
            samples = params['lower'] + (params['upper'] - params['lower']) * unit_samples
        
        elif dist_type == 'lognormal':
            samples = stats.lognorm.ppf(unit_samples, params['sigma'], scale=np.exp(params['mu']))
        
        elif dist_type == 'beta':
            samples = stats.beta.ppf(unit_samples, params['alpha'], params['beta'])
            # Scale to bounds if provided
            if param.bounds is not None:
                lower, upper = param.bounds
                samples = lower + (upper - lower) * samples
        
        elif dist_type == 'triangular':
            mode = params['mode']
            lower = params['lower']
            upper = params['upper']
            
            # Convert to scipy parameterization
            c = (mode - lower) / (upper - lower)
            samples = stats.triang.ppf(unit_samples, c, loc=lower, scale=upper-lower)
        
        else:
            raise ValueError(f"Unsupported distribution: {dist_type}")
        
        # Apply bounds if specified (except beta which is already handled)
        if param.bounds is not None and dist_type != 'beta':
            samples = np.clip(samples, param.bounds[0], param.bounds[1])
        
        return samples
    
    def generate_importance_samples(self, 
                                  num_samples: int,
                                  importance_region: Dict[str, Tuple[float, float]],
                                  importance_weight: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate importance-weighted samples for rare event simulation.
        
        Args:
            num_samples: Number of samples to generate
            importance_region: Dictionary mapping parameter names to (lower, upper) bounds
                              defining the importance region
            importance_weight: Weight factor for importance region
            
        Returns:
            Tuple of (samples, weights) where weights account for importance sampling
        """
        self.logger.info("Generating importance-weighted samples")
        
        samples = self._generate_lhs_samples(num_samples)
        weights = np.ones(num_samples)
        
        # Identify samples in importance region
        for param_name, (lower, upper) in importance_region.items():
            # Find parameter index
            param_idx = None
            for i, param in enumerate(self.uncertain_parameters):
                if param.name == param_name:
                    param_idx = i
                    break
            
            if param_idx is not None:
                # Check which samples are in importance region
                in_region = (samples[:, param_idx] >= lower) & (samples[:, param_idx] <= upper)
                weights[in_region] *= importance_weight
        
        return samples, weights
    
    def stratified_sampling(self, 
                           num_samples: int,
                           stratification_parameter: str,
                           num_strata: int = 10) -> np.ndarray:
        """Generate stratified samples based on one parameter.
        
        Args:
            num_samples: Total number of samples
            stratification_parameter: Name of parameter to stratify on
            num_strata: Number of strata
            
        Returns:
            Array of stratified samples
        """
        # Find stratification parameter index
        strat_idx = None
        strat_param = None
        for i, param in enumerate(self.uncertain_parameters):
            if param.name == stratification_parameter:
                strat_idx = i
                strat_param = param
                break
        
        if strat_idx is None:
            raise ValueError(f"Parameter {stratification_parameter} not found")
        
        # Calculate samples per stratum
        samples_per_stratum = num_samples // num_strata
        remaining_samples = num_samples % num_strata
        
        all_samples = []
        
        # Generate samples for each stratum
        for stratum in range(num_strata):
            # Calculate number of samples for this stratum
            stratum_samples = samples_per_stratum
            if stratum < remaining_samples:
                stratum_samples += 1
            
            if stratum_samples == 0:
                continue
            
            # Define stratum bounds
            stratum_lower = stratum / num_strata
            stratum_upper = (stratum + 1) / num_strata
            
            # Generate LHS samples for other parameters
            other_samples = qmc.LatinHypercube(
                d=self.num_parameters-1, seed=self.random_seed
            ).random(stratum_samples)
            
            # Generate stratified samples for stratification parameter
            strat_unit_samples = np.random.uniform(
                stratum_lower, stratum_upper, stratum_samples
            )
            strat_samples = self._transform_unit_samples(strat_param, strat_unit_samples)
            
            # Combine samples
            full_samples = np.zeros((stratum_samples, self.num_parameters))
            other_idx = 0
            
            for i, param in enumerate(self.uncertain_parameters):
                if i == strat_idx:
                    full_samples[:, i] = strat_samples
                else:
                    full_samples[:, i] = self._transform_unit_samples(param, other_samples[:, other_idx])
                    other_idx += 1
            
            all_samples.append(full_samples)
        
        # Combine all strata
        return np.vstack(all_samples)
    
    def adaptive_sampling(self, 
                         evaluation_function: callable,
                         initial_samples: int = 100,
                         max_samples: int = 1000,
                         convergence_threshold: float = 0.01,
                         batch_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Adaptive sampling that adds samples based on convergence criteria.
        
        Args:
            evaluation_function: Function to evaluate samples
            initial_samples: Initial number of samples
            max_samples: Maximum total samples
            convergence_threshold: Threshold for convergence detection
            batch_size: Number of samples to add per iteration
            
        Returns:
            Tuple of (samples, evaluations)
        """
        self.logger.info("Starting adaptive sampling")
        
        # Generate initial samples
        samples = self._generate_lhs_samples(initial_samples)
        evaluations = np.array([evaluation_function(sample) for sample in samples])
        
        # Adaptive sampling loop
        while len(samples) < max_samples:
            # Check convergence
            if self._check_convergence(evaluations, convergence_threshold):
                self.logger.info(f"Converged after {len(samples)} samples")
                break
            
            # Generate additional samples
            new_samples = self._generate_lhs_samples(batch_size)
            new_evaluations = np.array([evaluation_function(sample) for sample in new_samples])
            
            # Combine samples
            samples = np.vstack([samples, new_samples])
            evaluations = np.concatenate([evaluations, new_evaluations])
            
            self.logger.info(f"Added {batch_size} samples, total: {len(samples)}")
        
        return samples, evaluations
    
    def _check_convergence(self, 
                          evaluations: np.ndarray,
                          threshold: float) -> bool:
        """Check if sampling has converged based on mean stability."""
        if len(evaluations) < 50:  # Minimum samples for convergence check
            return False
        
        # Calculate running means
        n = len(evaluations)
        window = min(50, n // 4)  # Use 25% of samples or 50, whichever is smaller
        
        mean_recent = np.mean(evaluations[-window:])
        mean_previous = np.mean(evaluations[-(2*window):-window])
        
        # Check relative change
        if abs(mean_previous) > 1e-10:
            relative_change = abs(mean_recent - mean_previous) / abs(mean_previous)
            return relative_change < threshold
        else:
            return abs(mean_recent - mean_previous) < threshold