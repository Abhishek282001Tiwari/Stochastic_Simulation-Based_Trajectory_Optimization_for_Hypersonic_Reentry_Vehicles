# Hypersonic Reentry Framework API Reference

## Module Overview

The Hypersonic Reentry Framework provides a comprehensive set of modules for simulating, analyzing, and optimizing hypersonic reentry trajectories with uncertainty quantification.

---

## Core Modules

### `hypersonic_reentry.dynamics`

#### `VehicleDynamics`
**Purpose**: Implements 3-DOF point mass dynamics for hypersonic vehicles

```python
from hypersonic_reentry.dynamics import VehicleDynamics

# Initialize dynamics model
dynamics = VehicleDynamics(mass=5000.0, reference_area=15.0, 
                          drag_coefficient=1.2, lift_coefficient=0.8)

# Compute time derivative of state vector
state_dot = dynamics.compute_derivatives(state, time, controls)
```

**Key Methods:**
- `compute_derivatives(state, t, controls)`: Computes state time derivatives
- `get_aerodynamic_forces(state, atmosphere)`: Returns [drag, lift, side_force]
- `coordinate_transform(state, frame_from, frame_to)`: Transform between coordinate systems

**State Vector Format:**
- Position: `[longitude, latitude, altitude]` (spherical coordinates)
- Velocity: `[velocity_magnitude, flight_path_angle, heading_angle]`

#### `VehicleState`
**Purpose**: Container for vehicle state information with validation

```python
from hypersonic_reentry.dynamics import VehicleState

state = VehicleState(
    longitude=0.0,           # rad
    latitude=0.0,            # rad  
    altitude=120000.0,       # m
    velocity=7500.0,         # m/s
    flight_path_angle=-0.05, # rad
    heading_angle=0.0        # rad
)
```

---

### `hypersonic_reentry.atmosphere`

#### `USStandard1976`
**Purpose**: US Standard Atmosphere 1976 implementation with uncertainty modeling

```python
from hypersonic_reentry.atmosphere import USStandard1976

atmosphere = USStandard1976()

# Get atmospheric properties at altitude
altitude = 50000.0  # m
density, pressure, temperature = atmosphere.get_properties(altitude)

# With uncertainty
density_uncertain = atmosphere.get_density_with_uncertainty(altitude, factor=1.15)
```

**Key Methods:**
- `get_properties(altitude)`: Returns (density, pressure, temperature)
- `get_density(altitude)`: Returns atmospheric density
- `get_temperature(altitude)`: Returns temperature
- `get_pressure(altitude)`: Returns pressure
- `get_scale_height(altitude)`: Returns atmospheric scale height

**Altitude Ranges:**
- **Layer 1**: 0-11 km (troposphere)
- **Layer 2**: 11-20 km (tropopause)
- **Layer 3**: 20-32 km (stratosphere)
- **Layer 4**: 32-47 km (stratosphere)
- **Layer 5**: 47-51 km (stratopause)
- **Layer 6**: 51-71 km (mesosphere)
- **Layer 7**: 71-84.852 km (mesosphere)

---

### `hypersonic_reentry.uncertainty`

#### `UncertaintyQuantifier`
**Purpose**: Main orchestration class for uncertainty propagation and analysis

```python
from hypersonic_reentry.uncertainty import UncertaintyQuantifier, UncertainParameter

# Define uncertain parameters
parameters = [
    UncertainParameter('mass', 5000, 250, 'normal'),
    UncertainParameter('drag_coefficient', 1.2, 0.12, 'normal'),
    UncertainParameter('atmospheric_density_factor', 1.0, 0.15, 'lognormal')
]

# Initialize UQ framework
uq = UncertaintyQuantifier(parameters)

# Run Monte Carlo analysis
results = uq.monte_carlo_analysis(
    num_samples=1000,
    simulation_function=trajectory_simulation,
    method='latin_hypercube'
)

# Perform sensitivity analysis
sensitivity_results = uq.sobol_analysis(
    simulation_function=trajectory_simulation,
    num_base_samples=1024
)
```

**Key Methods:**
- `monte_carlo_analysis(num_samples, simulation_function, **kwargs)`: MC simulation
- `polynomial_chaos_expansion(simulation_function, **kwargs)`: PCE analysis  
- `sobol_analysis(simulation_function, num_base_samples)`: Global sensitivity analysis
- `get_confidence_intervals(data, confidence_levels)`: Compute confidence bounds

#### `UncertainParameter`
**Purpose**: Defines individual uncertain parameters

```python
# Normal distribution
mass_param = UncertainParameter(
    name='vehicle_mass',
    nominal_value=5000.0,
    uncertainty=250.0,  # standard deviation
    distribution='normal',
    bounds=(4000.0, 6000.0)
)

# Log-normal distribution
density_factor = UncertainParameter(
    name='atmospheric_density_factor',
    nominal_value=1.0,
    uncertainty=0.15,
    distribution='lognormal'
)
```

**Supported Distributions:**
- `'normal'`: Gaussian distribution
- `'lognormal'`: Log-normal distribution  
- `'uniform'`: Uniform distribution
- `'beta'`: Beta distribution
- `'triangular'`: Triangular distribution

---

### `hypersonic_reentry.optimization`

#### `GradientBasedOptimizer`
**Purpose**: Sequential Quadratic Programming (SQP) trajectory optimization

```python
from hypersonic_reentry.optimization import GradientBasedOptimizer
from hypersonic_reentry.optimization import OptimizationObjective, OptimizationConstraint

# Define objective function
objective = OptimizationObjective(
    function=lambda x: -compute_downrange(x),  # Maximize downrange
    gradient=compute_downrange_gradient
)

# Define constraints
constraints = [
    OptimizationConstraint(
        function=lambda x: final_altitude_constraint(x),
        jacobian=altitude_constraint_jacobian,
        bounds=(-2000, 2000),  # ±2 km tolerance
        type='equality'
    ),
    OptimizationConstraint(
        function=lambda x: heat_rate_constraint(x),
        jacobian=heat_rate_jacobian,
        bounds=(0, 5e6),  # Max 5 MW/m²
        type='inequality'
    )
]

# Initialize optimizer
optimizer = GradientBasedOptimizer(
    objective=objective,
    constraints=constraints,
    method='SQP'
)

# Solve optimization problem
result = optimizer.optimize(
    initial_guess=initial_controls,
    bounds=control_bounds,
    options={'maxiter': 100, 'ftol': 1e-6}
)
```

**Key Methods:**
- `optimize(initial_guess, bounds, options)`: Solve optimization problem
- `check_convergence(result)`: Verify solution quality
- `compute_sensitivities(solution)`: Parameter sensitivity analysis

**Optimization Result:**
```python
result = {
    'success': True,
    'x': optimal_controls,        # Optimal control history
    'fun': optimal_objective,     # Objective function value
    'nit': num_iterations,        # Number of iterations
    'nfev': num_evaluations,      # Function evaluations
    'message': 'Convergence achieved',
    'constraints_satisfied': True
}
```

---

### `hypersonic_reentry.analysis`

#### `StatisticalAnalyzer`
**Purpose**: Comprehensive statistical analysis of simulation results

```python
from hypersonic_reentry.analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Analyze Monte Carlo results
stats = analyzer.analyze_results(monte_carlo_data)

# Fit probability distributions
distribution_fits = analyzer.fit_distributions(
    data=final_altitudes,
    distributions=['normal', 'lognormal', 'weibull']
)

# Compute confidence intervals
confidence_intervals = analyzer.compute_confidence_intervals(
    data=downrange_distances,
    confidence_levels=[0.90, 0.95, 0.99]
)

# Reliability analysis
reliability_metrics = analyzer.compute_reliability(
    data=simulation_results,
    success_criteria={
        'final_altitude': (28000, 32000),  # Target ±2 km
        'max_heat_rate': (0, 5e6),         # Below 5 MW/m²
        'downrange_error': (-100000, 100000) # Within ±100 km
    }
)
```

**Key Methods:**
- `analyze_results(data)`: Comprehensive statistical summary
- `fit_distributions(data, distributions)`: Distribution fitting with GOF tests
- `compute_confidence_intervals(data, levels)`: Bootstrap confidence intervals
- `compute_reliability(data, criteria)`: Mission success probability
- `correlation_analysis(input_data, output_data)`: Parameter correlations

#### `ResultsGenerator`
**Purpose**: Generate comprehensive analysis results and reports

```python
from hypersonic_reentry.analysis import ResultsGenerator

generator = ResultsGenerator()

# Generate complete analysis
results = generator.generate_comprehensive_results(
    monte_carlo_samples=1000,
    optimization_scenarios=['shallow', 'moderate', 'steep'],
    sensitivity_analysis=True,
    save_results=True,
    output_directory='results/'
)

# Create analysis report
report = generator.create_analysis_report(
    results_data=results,
    include_plots=True,
    format='html'
)
```

---

### `hypersonic_reentry.visualization`

#### `PlotManager`
**Purpose**: Publication-quality matplotlib plotting

```python
from hypersonic_reentry.visualization import PlotManager

plotter = PlotManager(style='publication')

# Trajectory plots
fig = plotter.plot_trajectory_3d(
    trajectories=trajectory_data,
    uncertainty_bounds=True,
    show_atmosphere_layers=True
)

# Statistical plots
fig = plotter.plot_monte_carlo_results(
    results=mc_results,
    output_variables=['final_altitude', 'downrange', 'flight_time'],
    include_histograms=True,
    include_correlations=True
)

# Save plots
plotter.save_figure(fig, 'trajectory_analysis.pdf', dpi=300)
```

#### `AdvancedPlotter`
**Purpose**: Interactive Plotly visualizations and dashboards

```python
from hypersonic_reentry.visualization import AdvancedPlotter

plotter = AdvancedPlotter()

# Interactive 3D trajectory visualization
fig = plotter.create_interactive_trajectory_plot(
    trajectory_data=trajectories,
    uncertainty_data=monte_carlo_results,
    include_controls=True
)

# Monte Carlo dashboard
dashboard = plotter.create_monte_carlo_dashboard(
    input_parameters=uncertain_params,
    output_metrics=simulation_results,
    correlation_matrix=correlations
)

# Sensitivity analysis visualization
fig = plotter.plot_sobol_indices(
    sensitivity_results=sobol_results,
    parameters=parameter_names,
    outputs=output_names
)
```

---

### `hypersonic_reentry.utils`

#### `constants`
**Purpose**: Physical and mathematical constants

```python
from hypersonic_reentry.utils.constants import *

# Physical constants
EARTH_RADIUS          # 6.371e6 m
GRAVITATIONAL_PARAM   # 3.986e14 m³/s²
EARTH_ROTATION_RATE   # 7.292e-5 rad/s

# Atmospheric constants
GAS_CONSTANT_AIR      # 287.0 J/(kg·K)
SPECIFIC_HEAT_RATIO   # 1.4

# Mathematical constants
DEG_TO_RAD           # π/180
RAD_TO_DEG           # 180/π
```

#### `performance`
**Purpose**: Performance optimization utilities

```python
from hypersonic_reentry.utils.performance import ParallelProcessor, PerformanceProfiler

# Parallel processing
processor = ParallelProcessor(n_workers=8)
results = processor.parallel_monte_carlo(
    simulation_function=trajectory_sim,
    parameter_samples=parameter_matrix,
    chunk_size=100
)

# Performance profiling
profiler = PerformanceProfiler()
with profiler.profile_block('monte_carlo_simulation'):
    # Code to profile
    results = monte_carlo_analysis()

# Get profiling results
profile_stats = profiler.get_stats()
```

---

## Usage Examples

### Complete Workflow Example

```python
import numpy as np
from hypersonic_reentry import *

# 1. Define uncertain parameters
parameters = [
    UncertainParameter('mass', 5000, 250, 'normal'),
    UncertainParameter('drag_coefficient', 1.2, 0.12, 'normal'),
    UncertainParameter('atmospheric_density_factor', 1.0, 0.15, 'lognormal')
]

# 2. Initialize framework components
dynamics = VehicleDynamics(mass=5000.0, reference_area=15.0)
atmosphere = USStandard1976()
uq = UncertaintyQuantifier(parameters)

# 3. Define simulation function
def trajectory_simulation(param_dict):
    # Update vehicle parameters
    dynamics.update_parameters(param_dict)
    
    # Simulate trajectory
    trajectory = integrate_trajectory(dynamics, atmosphere)
    
    # Return metrics of interest
    return {
        'final_altitude': trajectory[-1].altitude,
        'downrange': compute_downrange(trajectory),
        'max_heat_rate': compute_max_heat_rate(trajectory),
        'flight_time': trajectory[-1].time
    }

# 4. Run Monte Carlo analysis
mc_results = uq.monte_carlo_analysis(
    num_samples=1000,
    simulation_function=trajectory_simulation
)

# 5. Perform sensitivity analysis
sensitivity_results = uq.sobol_analysis(
    simulation_function=trajectory_simulation,
    num_base_samples=512
)

# 6. Statistical analysis
analyzer = StatisticalAnalyzer()
statistical_summary = analyzer.analyze_results(mc_results)

# 7. Generate visualizations
plotter = PlotManager()
trajectory_plot = plotter.plot_monte_carlo_results(mc_results)
sensitivity_plot = plotter.plot_sensitivity_analysis(sensitivity_results)

# 8. Create comprehensive report
generator = ResultsGenerator()
final_report = generator.create_analysis_report(
    monte_carlo_results=mc_results,
    sensitivity_results=sensitivity_results,
    statistical_analysis=statistical_summary
)
```

### Optimization Example

```python
# Define optimization problem
objective = OptimizationObjective(
    function=lambda controls: -compute_downrange(controls),
    gradient=compute_downrange_gradient
)

constraints = [
    # Final altitude constraint: 30 ± 2 km  
    OptimizationConstraint(
        function=lambda controls: final_altitude(controls) - 30000.0,
        jacobian=altitude_jacobian,
        bounds=(-2000.0, 2000.0),
        type='equality'
    ),
    # Heat rate constraint: ≤ 5 MW/m²
    OptimizationConstraint(
        function=lambda controls: max_heat_rate(controls),
        jacobian=heat_rate_jacobian, 
        bounds=(0.0, 5e6),
        type='inequality'
    )
]

# Solve optimization
optimizer = GradientBasedOptimizer(objective, constraints)
optimal_solution = optimizer.optimize(
    initial_guess=nominal_controls,
    bounds=control_bounds,
    options={'maxiter': 100, 'ftol': 1e-6}
)
```

---

## Error Handling

All framework modules include comprehensive error handling:

```python
try:
    results = uq.monte_carlo_analysis(num_samples=1000, 
                                    simulation_function=trajectory_sim)
except ValidationError as e:
    print(f"Parameter validation failed: {e}")
except ConvergenceError as e:
    print(f"Simulation convergence failed: {e}")
except ComputationError as e:
    print(f"Numerical computation error: {e}")
```

**Common Exception Types:**
- `ValidationError`: Parameter or input validation failures
- `ConvergenceError`: Integration or optimization convergence issues  
- `ComputationError`: Numerical computation problems
- `ConfigurationError`: Framework setup or configuration errors

---

## Performance Considerations

### Memory Management
- Use `chunk_size` parameter for large Monte Carlo analyses
- Enable `parallel_processing` for multi-core execution
- Consider `data_compression` for large result sets

### Computational Efficiency
- Pre-compile simulation functions when possible
- Use vectorized operations in NumPy/SciPy
- Cache atmospheric property calculations
- Leverage JAX for automatic differentiation in optimization

### Recommended Hardware
- **Minimum**: 8 GB RAM, 4 CPU cores
- **Recommended**: 32 GB RAM, 16 CPU cores  
- **Large Studies**: 64 GB RAM, 32+ CPU cores, GPU acceleration

---

For detailed examples and tutorials, see the `/examples/` and `/notebooks/` directories.