"""Polynomial Chaos Expansion for uncertainty quantification."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import special

from .uncertainty_quantifier import UncertainParameter


class PolynomialChaosExpansion:
    """Polynomial Chaos Expansion for efficient uncertainty propagation."""
    
    def __init__(self, uncertain_parameters: List[UncertainParameter]):
        """Initialize PCE.
        
        Args:
            uncertain_parameters: List of uncertain parameters
        """
        self.logger = logging.getLogger(__name__)
        self.uncertain_parameters = uncertain_parameters
        self.num_parameters = len(uncertain_parameters)
    
    def generate_collocation_points(self, polynomial_order: int) -> np.ndarray:
        """Generate collocation points for PCE."""
        # Use tensor grid of Gaussian quadrature points
        num_points_1d = polynomial_order + 1
        
        # Generate 1D quadrature points for each parameter
        points_1d = []
        for param in self.uncertain_parameters:
            if param.distribution_type == 'normal':
                # Gauss-Hermite quadrature for normal distribution
                points, _ = np.polynomial.hermite.hermgauss(num_points_1d)
                # Transform to physical space
                mean = param.parameters['mean']
                std = param.parameters['std']
                points = mean + std * np.sqrt(2) * points
            elif param.distribution_type == 'uniform':
                # Gauss-Legendre quadrature for uniform distribution
                points, _ = np.polynomial.legendre.leggauss(num_points_1d)
                # Transform to [a,b] interval
                a, b = param.parameters['lower'], param.parameters['upper']
                points = 0.5 * (b - a) * points + 0.5 * (a + b)
            else:
                # Fallback: uniform points
                a = param.parameters.get('lower', 0)
                b = param.parameters.get('upper', 1)
                points = np.linspace(a, b, num_points_1d)
            
            points_1d.append(points)
        
        # Create tensor product grid
        meshgrid = np.meshgrid(*points_1d, indexing='ij')
        collocation_points = np.column_stack([grid.ravel() for grid in meshgrid])
        
        return collocation_points
    
    def build_expansion(self, 
                       samples: np.ndarray,
                       outputs: np.ndarray,
                       polynomial_order: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Build polynomial chaos expansion."""
        # Simple linear regression approach
        # Generate polynomial basis matrix
        basis_matrix = self._generate_basis_matrix(samples, polynomial_order)
        
        # Solve least squares problem
        coefficients = np.linalg.lstsq(basis_matrix, outputs, rcond=None)[0]
        
        # Calculate statistics
        mean_pce = coefficients[0]  # First coefficient is the mean
        variance_pce = np.sum(coefficients[1:]**2)  # Sum of squared coefficients
        
        stats = {
            'mean': mean_pce,
            'variance': variance_pce,
            'std': np.sqrt(variance_pce)
        }
        
        return coefficients, stats
    
    def _generate_basis_matrix(self, samples: np.ndarray, polynomial_order: int) -> np.ndarray:
        """Generate polynomial basis matrix."""
        num_samples = len(samples)
        
        # Simple multilinear basis (tensor product of 1D polynomials)
        basis_functions = []
        
        # Constant term
        basis_functions.append(np.ones(num_samples))
        
        # Linear terms
        for i in range(self.num_parameters):
            basis_functions.append(samples[:, i])
        
        # Quadratic terms (if order >= 2)
        if polynomial_order >= 2:
            for i in range(self.num_parameters):
                basis_functions.append(samples[:, i]**2)
        
        return np.column_stack(basis_functions)
    
    def evaluate_expansion(self, coefficients: np.ndarray, test_points: np.ndarray) -> np.ndarray:
        """Evaluate PCE at test points."""
        basis_matrix = self._generate_basis_matrix(test_points, len(coefficients))
        return basis_matrix @ coefficients