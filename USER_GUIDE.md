# User Guide

## Hypersonic Reentry Trajectory Optimization Framework

This comprehensive user guide provides detailed instructions for using the Stochastic Simulation-Based Trajectory Optimization framework for Hypersonic Reentry Vehicles.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Framework Overview](#framework-overview)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Configuration](#configuration)
6. [Examples](#examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### 5-Minute Example

```python
import numpy as np
from hypersonic_reentry.dynamics import VehicleDynamics, VehicleState
from hypersonic_reentry.atmosphere import USStandard1976
from hypersonic_reentry.utils.constants import DEG_TO_RAD

# 1. Create atmosphere model
atmosphere = USStandard1976()

# 2. Define vehicle parameters
vehicle_params = {
    'mass': 5000.0,  # kg
    'reference_area': 15.0,  # m^2
    'drag_coefficient': 1.2,
    'lift_coefficient': 0.8,
    'ballistic_coefficient': 400.0,  # kg/m^2
    'nose_radius': 0.5,  # m
    'length': 10.0,  # m
    'diameter': 2.0,  # m
}

# 3. Create vehicle dynamics
dynamics = VehicleDynamics(vehicle_params, atmosphere_model=atmosphere)

# 4. Define initial conditions
initial_state = VehicleState(
    altitude=120000.0,  # 120 km
    latitude=28.5 * DEG_TO_RAD,  # Kennedy Space Center
    longitude=-80.6 * DEG_TO_RAD,
    velocity=7800.0,  # m/s
    flight_path_angle=-1.5 * DEG_TO_RAD,  # shallow entry
    azimuth=90.0 * DEG_TO_RAD,  # eastward
    time=0.0
)

# 5. Run simulation
trajectory = dynamics.integrate_trajectory(
    initial_state=initial_state,
    time_span=(0.0, 2000.0),  # 2000 seconds
    time_step=1.0
)

# 6. Analyze results
print(f"Final altitude: {trajectory['altitude'][-1]/1000:.1f} km")
print(f"Downrange: {trajectory['downrange'][-1]/1000:.1f} km")
print(f"Flight time: {trajectory['time'][-1]/60:.1f} minutes")
```

## Framework Overview

### Core Components

1. **Vehicle Dynamics** (`hypersonic_reentry.dynamics`)
   - 3-DOF point mass equations of motion
   - Aerodynamic force and moment calculations
   - Atmospheric flight mechanics

2. **Atmosphere Models** (`hypersonic_reentry.atmosphere`)
   - US Standard Atmosphere 1976
   - Custom atmospheric profiles
   - Uncertainty modeling

3. **Uncertainty Quantification** (`hypersonic_reentry.uncertainty`)
   - Monte Carlo simulation
   - Polynomial Chaos Expansion
   - Sensitivity analysis

4. **Optimization** (`hypersonic_reentry.optimization`)
   - Gradient-based methods (SQP)
   - Constraint handling
   - Multi-objective optimization

5. **Analysis Tools** (`hypersonic_reentry.analysis`)
   - Statistical analysis
   - Results generation
   - Performance metrics

6. **Visualization** (`hypersonic_reentry.visualization`)
   - 2D/3D trajectory plots
   - Statistical visualizations
   - Interactive dashboards

## Basic Usage

### 1. Simple Trajectory Simulation

```python
from hypersonic_reentry import *

# Load default configuration
config = load_default_config()

# Create components
atmosphere = USStandard1976()
dynamics = VehicleDynamics(config['vehicle'], atmosphere_model=atmosphere)

# Define initial conditions
initial_state = VehicleState(
    **config['initial_conditions']
)

# Run simulation
trajectory = dynamics.integrate_trajectory(
    initial_state=initial_state,
    time_span=(0.0, config['simulation']['max_time']),
    time_step=config['simulation']['time_step']
)

# Create plots
plot_manager = PlotManager()
fig = plot_manager.plot_trajectory_2d(trajectory)
fig.show()
```

### 2. Monte Carlo Uncertainty Analysis

```python
from hypersonic_reentry.uncertainty import UncertaintyQuantifier, UncertainParameter

# Define uncertain parameters
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
    ),
    UncertainParameter(
        name="atmospheric_density_factor",
        distribution_type="log_normal",
        parameters={"mean": 1.0, "std": 0.15}
    )
]

# Create uncertainty quantifier
uq = UncertaintyQuantifier(
    vehicle_dynamics=dynamics,
    uncertain_parameters=uncertain_params,
    random_seed=42
)

# Run Monte Carlo analysis
mc_result = uq.run_monte_carlo_analysis(
    initial_state=initial_state,
    time_span=(0.0, 2000.0),
    num_samples=1000,
    parallel=True,
    num_workers=4
)

# Analyze results
print("Monte Carlo Results:")
print(f"Samples: {mc_result.num_samples}")
print(f"Mean final altitude: {mc_result.mean_values['final_altitude']:.1f} m")
print(f"Std final altitude: {mc_result.std_values['final_altitude']:.1f} m")
```

### 3. Trajectory Optimization

```python
from hypersonic_reentry.optimization import GradientBasedOptimizer
from hypersonic_reentry.optimization import OptimizationObjective, OptimizationConstraint

# Define objectives
objectives = [
    OptimizationObjective(
        name="downrange",
        objective_type="maximize",
        weight=1.0
    )
]

# Define constraints
constraints = [
    OptimizationConstraint(
        name="final_altitude",
        constraint_type="equality",
        target_value=30000.0,  # 30 km
        tolerance=1000.0  # ±1 km
    ),
    OptimizationConstraint(
        name="max_heat_rate",
        constraint_type="inequality",
        target_value=5.0e6,  # 5 MW/m^2
        tolerance=0.0
    )
]

# Define control bounds
control_bounds = {
    "bank_angle": (-60.0 * DEG_TO_RAD, 60.0 * DEG_TO_RAD),
    "angle_of_attack": (0.0 * DEG_TO_RAD, 20.0 * DEG_TO_RAD)
}

# Create optimizer
optimizer = GradientBasedOptimizer(
    vehicle_dynamics=dynamics,
    objectives=objectives,
    constraints=constraints,
    control_bounds=control_bounds
)

# Run optimization
opt_result = optimizer.optimize(
    initial_state=initial_state,
    time_span=(0.0, 1500.0)
)

print(f"Optimization success: {opt_result.success}")
print(f"Final objective value: {opt_result.objective_value:.2f}")
print(f"Iterations: {opt_result.num_iterations}")
```

## Advanced Features

### 1. Custom Vehicle Models

```python
class CustomVehicle(VehicleDynamics):
    def __init__(self, vehicle_params, atmosphere_model=None):
        super().__init__(vehicle_params, atmosphere_model)
        
        # Add custom parameters
        self.custom_param = vehicle_params.get('custom_param', 1.0)
    
    def calculate_aerodynamic_coefficients(self, velocity, altitude, angle_of_attack):
        """Custom aerodynamic model."""
        mach = velocity / self.get_sound_speed(altitude)
        
        # Custom drag coefficient model
        cd_base = self.vehicle_params['drag_coefficient']
        cd = cd_base * (1.0 + 0.1 * mach**2) * self.custom_param
        
        # Custom lift coefficient model
        cl_base = self.vehicle_params['lift_coefficient']
        cl = cl_base * np.sin(angle_of_attack) * (1.0 + 0.05 * mach)
        
        return {'drag_coefficient': cd, 'lift_coefficient': cl}
```

### 2. Custom Atmosphere Models

```python
from hypersonic_reentry.atmosphere.base import AtmosphereModel

class MarsAtmosphere(AtmosphereModel):
    """Mars atmospheric model."""
    
    def get_properties(self, altitude, latitude=0.0, longitude=0.0, time=0.0):
        """Get atmospheric properties at given conditions."""
        # Mars atmospheric scale height
        H = 11100.0  # meters
        
        # Surface conditions
        rho_0 = 0.020  # kg/m³
        T_0 = 210.0    # K
        P_0 = 610.0    # Pa
        
        # Exponential atmosphere
        rho = rho_0 * np.exp(-altitude / H)
        T = T_0 * (1.0 - 0.0065 * altitude / 1000.0)  # Linear temperature profile
        P = P_0 * np.exp(-altitude / H)
        
        # Mars atmospheric composition (mostly CO2)
        R_specific = 188.9  # J/(kg·K) for CO2
        sound_speed = np.sqrt(1.3 * R_specific * T)
        
        return {
            'density': rho,
            'temperature': T,
            'pressure': P,
            'sound_speed': sound_speed,
            'dynamic_viscosity': 1.422e-5 * T**1.5 / (T + 240.0)
        }
```

### 3. Multi-Objective Optimization

```python
# Define multiple objectives
objectives = [
    OptimizationObjective(
        name="downrange",
        objective_type="maximize",
        weight=0.6
    ),
    OptimizationObjective(
        name="final_velocity",
        objective_type="minimize",
        weight=0.4
    )
]

# Add heat rate constraint
constraints = [
    OptimizationConstraint(
        name="max_heat_rate",
        constraint_type="inequality",
        target_value=4.0e6  # Stricter limit
    )
]

# Run multi-objective optimization
multi_opt_result = optimizer.optimize(
    initial_state=initial_state,
    time_span=(0.0, 1800.0),
    multi_objective=True
)
```

### 4. Polynomial Chaos Expansion

```python
# Use PCE for efficient uncertainty propagation
pce_result = uq.run_polynomial_chaos_analysis(
    initial_state=initial_state,
    time_span=(0.0, 2000.0),
    polynomial_order=3,
    num_quadrature_points=100
)

# PCE provides analytical expressions for uncertainty
print("PCE Coefficients:")
for output, coeffs in pce_result.pce_coefficients.items():
    print(f"{output}: {len(coeffs)} terms")

# Fast Monte Carlo using PCE surrogate
surrogate_samples = pce_result.evaluate_pce(num_samples=10000)
print(f"Surrogate evaluation time: {pce_result.evaluation_time:.4f} seconds")
```

## Configuration

### Configuration File Structure

```yaml
# config.yaml
vehicle:
  mass: 5000.0
  reference_area: 15.0
  drag_coefficient: 1.2
  lift_coefficient: 0.8
  ballistic_coefficient: 400.0
  nose_radius: 0.5
  length: 10.0
  diameter: 2.0

initial_conditions:
  altitude: 120000.0
  latitude: 0.4974  # 28.5 degrees in radians
  longitude: -1.4066
  velocity: 7800.0
  flight_path_angle: -0.0262  # -1.5 degrees
  azimuth: 1.5708  # 90 degrees
  time: 0.0

simulation:
  max_time: 3000.0
  time_step: 1.0
  integration_method: "RK45"
  tolerance: 1e-9

uncertainty:
  num_samples: 1000
  sampling_method: "latin_hypercube"
  random_seed: 42
  confidence_level: 0.95

optimization:
  algorithm: "SLSQP"
  max_iterations: 100
  tolerance: 1e-6
  gradient_step: 1e-8

visualization:
  figure_size: [12, 8]
  dpi: 300
  style: "seaborn"
  color_palette: "viridis"

performance:
  parallel_workers: 4
  chunk_size: 100
  memory_limit_gb: 8.0
  enable_profiling: false
```

### Loading Configuration

```python
from hypersonic_reentry.utils.config import load_config

# Load custom configuration
config = load_config("my_config.yaml")

# Override specific parameters
config['uncertainty']['num_samples'] = 2000
config['performance']['parallel_workers'] = 8

# Use configuration
dynamics = VehicleDynamics(
    config['vehicle'], 
    atmosphere_model=USStandard1976()
)
```

## Examples

### Example 1: Parameter Sweep

```python
# Parameter sweep for different entry angles
entry_angles = np.linspace(-1.0, -10.0, 10)  # degrees
results = []

for angle in entry_angles:
    initial_state.flight_path_angle = angle * DEG_TO_RAD
    
    trajectory = dynamics.integrate_trajectory(
        initial_state, (0.0, 2000.0), 1.0
    )
    
    results.append({
        'entry_angle': angle,
        'final_altitude': trajectory['altitude'][-1],
        'downrange': trajectory['downrange'][-1],
        'max_heat_rate': np.max(trajectory['heat_rate'])
    })

# Create DataFrame for analysis
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

### Example 2: Sensitivity Analysis

```python
from hypersonic_reentry.analysis import SensitivityAnalyzer

# Create sensitivity analyzer
sensitivity = SensitivityAnalyzer(
    vehicle_dynamics=dynamics,
    uncertain_parameters=uncertain_params
)

# Run Sobol sensitivity analysis
sobol_result = sensitivity.sobol_analysis(
    initial_state=initial_state,
    time_span=(0.0, 2000.0),
    num_samples=1000,
    outputs=['final_altitude', 'downrange', 'max_heat_rate']
)

# Display results
for output in sobol_result.outputs:
    print(f"\n{output} Sensitivity:")
    for param, index in sobol_result.first_order_indices[output].items():
        print(f"  {param}: {index:.4f}")
```

### Example 3: Mission Planning

```python
# Mission planning with multiple launch windows
launch_times = np.arange(0, 24, 2)  # Every 2 hours
mission_results = []

for launch_time in launch_times:
    # Adjust initial conditions for Earth rotation
    longitude_offset = launch_time * 15.0 * DEG_TO_RAD  # 15°/hour
    initial_state.longitude += longitude_offset
    
    # Run trajectory optimization
    opt_result = optimizer.optimize(
        initial_state=initial_state,
        time_span=(0.0, 1800.0)
    )
    
    mission_results.append({
        'launch_time': launch_time,
        'success': opt_result.success,
        'downrange': opt_result.final_trajectory['downrange'][-1],
        'fuel_usage': opt_result.control_effort
    })

# Find optimal launch window
best_mission = max(mission_results, key=lambda x: x['downrange'])
print(f"Optimal launch time: {best_mission['launch_time']} hours")
```

## Best Practices

### 1. Performance Optimization

```python
# Use appropriate sample sizes
# - Development/testing: 50-100 samples
# - Preliminary analysis: 500-1000 samples  
# - Production analysis: 2000+ samples

# Enable parallel processing
mc_result = uq.run_monte_carlo_analysis(
    num_samples=1000,
    parallel=True,
    num_workers=min(8, mp.cpu_count())  # Don't exceed available cores
)

# Use chunked processing for large datasets
results_generator = ResultsGenerator(
    chunk_size=100,  # Process in chunks to manage memory
    save_intermediate=True  # Save intermediate results
)
```

### 2. Numerical Stability

```python
# Use appropriate tolerances
dynamics = VehicleDynamics(
    vehicle_params,
    atmosphere_model=atmosphere,
    integration_tolerance=1e-9,  # Tighter tolerance for accuracy
    max_step_size=1.0  # Limit step size for stability
)

# Check for numerical issues
trajectory = dynamics.integrate_trajectory(initial_state, (0.0, 2000.0), 1.0)

# Verify trajectory validity
if np.any(np.isnan(trajectory['altitude'])):
    print("Warning: NaN values detected in trajectory")
    
if np.any(trajectory['altitude'] < 0):
    print("Warning: Negative altitude detected")
```

### 3. Result Validation

```python
# Always validate results
def validate_trajectory(trajectory):
    """Validate trajectory for physical consistency."""
    checks = []
    
    # Check for monotonic altitude decrease (initially)
    if trajectory['altitude'][10] > trajectory['altitude'][0]:
        checks.append("WARNING: Altitude increasing initially")
    
    # Check velocity bounds
    if np.any(trajectory['velocity'] < 0) or np.any(trajectory['velocity'] > 15000):
        checks.append("WARNING: Unrealistic velocities")
    
    # Check heat rates
    if np.any(trajectory['heat_rate'] > 10e6):
        checks.append("WARNING: Extremely high heat rates")
    
    return checks

# Validate each trajectory
validation_results = validate_trajectory(trajectory)
for warning in validation_results:
    print(warning)
```

### 4. Error Handling

```python
import logging
from hypersonic_reentry.utils.exceptions import SimulationError, OptimizationError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Run analysis with error handling
    mc_result = uq.run_monte_carlo_analysis(
        initial_state=initial_state,
        time_span=(0.0, 2000.0),
        num_samples=1000
    )
    
except SimulationError as e:
    logger.error(f"Simulation failed: {e}")
    # Implement fallback or retry logic
    
except OptimizationError as e:
    logger.error(f"Optimization failed: {e}")
    # Try different optimization settings
    
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

## Troubleshooting

### Common Issues and Solutions

1. **Slow Performance**
   ```python
   # Enable performance profiling
   from hypersonic_reentry.utils.performance import PerformanceProfiler
   
   profiler = PerformanceProfiler(enable_profiling=True)
   
   with profiler.profile_function("monte_carlo"):
       mc_result = uq.run_monte_carlo_analysis(...)
   
   # Check performance summary
   summary = profiler.get_performance_summary()
   print(summary)
   ```

2. **Memory Issues**
   ```python
   # Monitor memory usage
   from hypersonic_reentry.utils.performance import MemoryOptimizer
   
   memory_optimizer = MemoryOptimizer(memory_threshold_gb=6.0)
   memory_stats = memory_optimizer.check_memory_usage()
   print(f"Memory usage: {memory_stats['process_memory_gb']:.2f} GB")
   
   # Use chunked processing
   results = results_generator.generate_monte_carlo_study(
       scenario=scenario,
       num_samples=10000,
       chunk_size=500  # Process in smaller chunks
   )
   ```

3. **Optimization Convergence**
   ```python
   # Try different optimization settings
   optimizer.algorithm = "trust-constr"  # More robust algorithm
   optimizer.max_iterations = 200  # More iterations
   optimizer.tolerance = 1e-4  # Relaxed tolerance
   
   # Add optimization callbacks
   def optimization_callback(xk, convergence_info):
       print(f"Iteration {convergence_info['nit']}: f = {convergence_info['fun']:.6f}")
       return False  # Continue optimization
   
   opt_result = optimizer.optimize(
       initial_state=initial_state,
       time_span=(0.0, 1500.0),
       callback=optimization_callback
   )
   ```

4. **Visualization Problems**
   ```python
   # Use alternative backends
   import matplotlib
   matplotlib.use('Agg')  # Non-interactive backend
   
   # Save figures instead of displaying
   plot_manager = PlotManager(output_directory="plots", show_plots=False)
   fig = plot_manager.plot_trajectory_2d(trajectory)
   plot_manager.save_figure(fig, "trajectory_2d.png")
   ```

### Getting Additional Help

1. Check the examples in `examples/` directory
2. Review test cases in `tests/` for usage patterns
3. Enable debug logging: `logging.getLogger().setLevel(logging.DEBUG)`
4. Use the validation suite to check your installation
5. Create minimal reproducible examples for issues

This concludes the comprehensive user guide. For more advanced topics and specific use cases, refer to the API documentation and example scripts.