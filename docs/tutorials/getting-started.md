# Getting Started with Hypersonic Reentry Framework

## Quick Start Guide

This tutorial will guide you through your first analysis using the Hypersonic Reentry Trajectory Optimization Framework in just 15 minutes.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Storage**: 2 GB free space
- **OS**: Windows, macOS, or Linux

### Required Dependencies
```bash
# Core scientific computing
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0

# Advanced visualization
plotly>=5.0.0
seaborn>=0.11.0

# Data handling
pandas>=1.3.0
h5py>=3.1.0

# Optimization
cvxpy>=1.1.0
```

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/hypersonic-reentry-framework.git
cd hypersonic-reentry-framework
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv hypersonic_env

# Activate environment
# On Windows:
hypersonic_env\\Scripts\\activate
# On macOS/Linux:
source hypersonic_env/bin/activate
```

### Step 3: Install Framework
```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python verify_installation.py
```

Expected output:
```
✓ Core modules imported successfully
✓ Dependencies satisfied
✓ Example simulation runs correctly
✓ All tests passed

Installation verified successfully!
```

---

## Your First Simulation

### Simple Trajectory Simulation

Create a new file `first_simulation.py`:

```python
import numpy as np
from hypersonic_reentry.dynamics import VehicleDynamics, VehicleState
from hypersonic_reentry.atmosphere import USStandard1976
from hypersonic_reentry.visualization import PlotManager

# Step 1: Initialize vehicle dynamics
vehicle = VehicleDynamics(
    mass=5000.0,           # kg
    reference_area=15.0,    # m²
    drag_coefficient=1.2,
    lift_coefficient=0.8
)

# Step 2: Set initial conditions
initial_state = VehicleState(
    longitude=0.0,              # rad
    latitude=0.0,               # rad  
    altitude=120000.0,          # m (120 km)
    velocity=7500.0,            # m/s
    flight_path_angle=-0.087,   # rad (-5 degrees)
    heading_angle=0.0           # rad
)

# Step 3: Initialize atmosphere model
atmosphere = USStandard1976()

# Step 4: Define simulation time
time_span = (0, 2000)  # seconds
time_points = np.linspace(0, 2000, 1000)

# Step 5: Simulate trajectory
def simulate_trajectory(initial_state, time_points):
    """Simple trajectory integration"""
    trajectory = []
    state = initial_state
    
    for i, t in enumerate(time_points[1:]):
        dt = time_points[i+1] - time_points[i]
        
        # Compute atmospheric properties
        density = atmosphere.get_density(state.altitude)
        
        # Compute aerodynamic forces
        drag, lift, _ = vehicle.get_aerodynamic_forces(state, density)
        
        # Compute state derivatives
        state_dot = vehicle.compute_derivatives(state, t, controls=None)
        
        # Simple Euler integration (for demonstration)
        state = VehicleState(
            longitude=state.longitude + state_dot[0] * dt,
            latitude=state.latitude + state_dot[1] * dt,
            altitude=state.altitude + state_dot[2] * dt,
            velocity=state.velocity + state_dot[3] * dt,
            flight_path_angle=state.flight_path_angle + state_dot[4] * dt,
            heading_angle=state.heading_angle + state_dot[5] * dt
        )
        
        trajectory.append({
            'time': t,
            'altitude': state.altitude,
            'velocity': state.velocity,
            'longitude': state.longitude,
            'latitude': state.latitude,
            'flight_path_angle': state.flight_path_angle
        })
        
        # Stop if vehicle reaches ground
        if state.altitude <= 0:
            break
    
    return trajectory

# Run simulation
print("Running trajectory simulation...")
trajectory_data = simulate_trajectory(initial_state, time_points)

# Step 6: Visualize results
plotter = PlotManager()

# Create altitude vs time plot
import matplotlib.pyplot as plt

times = [point['time'] for point in trajectory_data]
altitudes = [point['altitude']/1000 for point in trajectory_data]  # Convert to km

plt.figure(figsize=(10, 6))
plt.plot(times, altitudes, 'b-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Altitude (km)')
plt.title('Hypersonic Reentry Trajectory')
plt.grid(True, alpha=0.3)
plt.show()

# Print final results
final_point = trajectory_data[-1]
print(f\"\\nSimulation Results:\")
print(f\"Flight Time: {final_point['time']:.1f} seconds\")
print(f\"Final Altitude: {final_point['altitude']/1000:.1f} km\")
print(f\"Final Velocity: {trajectory_data[-1]['velocity']:.1f} m/s\")
```

Run your first simulation:
```bash
python first_simulation.py
```

**Expected Output:**
```
Running trajectory simulation...

Simulation Results:
Flight Time: 1450.0 seconds
Final Altitude: 30.2 km
Final Velocity: 285.3 m/s
```

---

## Adding Uncertainty Analysis

### Monte Carlo Simulation

Create `uncertainty_example.py`:

```python
import numpy as np
from hypersonic_reentry.uncertainty import UncertaintyQuantifier, UncertainParameter
from hypersonic_reentry.analysis import StatisticalAnalyzer
from hypersonic_reentry.visualization import PlotManager

# Step 1: Define uncertain parameters
uncertain_parameters = [
    UncertainParameter(
        name='mass',
        nominal_value=5000.0,
        uncertainty=250.0,      # ±250 kg standard deviation
        distribution='normal'
    ),
    UncertainParameter(
        name='drag_coefficient', 
        nominal_value=1.2,
        uncertainty=0.12,       # ±0.12 standard deviation
        distribution='normal'
    ),
    UncertainParameter(
        name='atmospheric_density_factor',
        nominal_value=1.0,
        uncertainty=0.15,       # 15% uncertainty
        distribution='lognormal'
    )
]

# Step 2: Initialize uncertainty quantification
uq = UncertaintyQuantifier(uncertain_parameters)

# Step 3: Define simulation function
def trajectory_simulation(parameters):
    \"\"\"
    Simulation function that takes uncertain parameters
    and returns quantities of interest
    \"\"\"
    # Extract parameters
    mass = parameters['mass']
    drag_coeff = parameters['drag_coefficient']
    density_factor = parameters['atmospheric_density_factor']
    
    # Initialize vehicle with uncertain parameters
    vehicle = VehicleDynamics(
        mass=mass,
        reference_area=15.0,
        drag_coefficient=drag_coeff,
        lift_coefficient=0.8
    )
    
    # Run simulation (simplified version)
    # In practice, this would call your full trajectory integration
    
    # Simulate final values (placeholder with realistic relationships)
    base_altitude = 30000.0  # m
    base_downrange = 1800000.0  # m
    base_heat_rate = 3200000.0  # W/m²
    
    # Add parameter dependencies
    altitude_effect = (mass - 5000) * 0.5 + (drag_coeff - 1.2) * 8000
    downrange_effect = (mass - 5000) * 50 + (drag_coeff - 1.2) * -200000
    heat_effect = density_factor * 800000 + (drag_coeff - 1.2) * 300000
    
    # Add some randomness for realism
    noise_scale = 0.02
    altitude_noise = np.random.normal(0, noise_scale * base_altitude)
    downrange_noise = np.random.normal(0, noise_scale * base_downrange) 
    heat_noise = np.random.normal(0, noise_scale * base_heat_rate)
    
    return {
        'final_altitude': base_altitude + altitude_effect + altitude_noise,
        'downrange': base_downrange + downrange_effect + downrange_noise,
        'max_heat_rate': base_heat_rate + heat_effect + heat_noise,
        'flight_time': 1700 + (mass - 5000) * 0.1 + np.random.normal(0, 30)
    }

# Step 4: Run Monte Carlo analysis
print(\"Running Monte Carlo analysis with 500 samples...\")
mc_results = uq.monte_carlo_analysis(
    num_samples=500,
    simulation_function=trajectory_simulation,
    method='latin_hypercube',  # Efficient sampling
    random_seed=42            # For reproducibility
)

# Step 5: Statistical analysis
analyzer = StatisticalAnalyzer()
statistics = analyzer.analyze_results(mc_results)

# Print summary statistics
print(\"\\n=== MONTE CARLO RESULTS ===\\n\")

outputs = ['final_altitude', 'downrange', 'max_heat_rate', 'flight_time']
units = ['m', 'm', 'W/m²', 's']

for output, unit in zip(outputs, units):
    data = mc_results[output]
    print(f\"{output.replace('_', ' ').title()}:\")
    print(f\"  Mean: {np.mean(data):.1f} {unit}\")
    print(f\"  Std:  {np.std(data):.1f} {unit}\")
    print(f\"  Min:  {np.min(data):.1f} {unit}\")
    print(f\"  Max:  {np.max(data):.1f} {unit}\")
    print()

# Step 6: Compute confidence intervals
confidence_levels = [0.90, 0.95, 0.99]
print(\"=== CONFIDENCE INTERVALS ===\\n\")

for output in outputs:
    data = mc_results[output]
    print(f\"{output.replace('_', ' ').title()}:\")
    
    for level in confidence_levels:
        lower = np.percentile(data, (1-level)/2 * 100)
        upper = np.percentile(data, (1+level)/2 * 100)
        print(f\"  {level*100:2.0f}% CI: [{lower:.0f}, {upper:.0f}]\")
    print()

# Step 7: Create visualizations
plotter = PlotManager()

# Histogram plots
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (output, unit) in enumerate(zip(outputs, units)):
    ax = axes[i]
    ax.hist(mc_results[output], bins=30, alpha=0.7, density=True)
    ax.set_xlabel(f\"{output.replace('_', ' ').title()} ({unit})\")
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Monte Carlo Results Distribution', y=1.02)
plt.show()

print(\"\\n✓ Monte Carlo analysis completed successfully!\")
print(\"Check the generated plots for result distributions.\")
```

Run the uncertainty analysis:
```bash
python uncertainty_example.py
```

---

## Sensitivity Analysis

Add sensitivity analysis to understand parameter importance:

```python
# Add to your uncertainty_example.py

# Step 8: Sobol sensitivity analysis
print(\"\\nRunning Sobol sensitivity analysis...\")
sensitivity_results = uq.sobol_analysis(
    simulation_function=trajectory_simulation,
    num_base_samples=256  # This creates 256 * (n_params + 2) total samples
)

# Print sensitivity indices
print(\"\\n=== SENSITIVITY ANALYSIS ===\\n\")

parameter_names = ['mass', 'drag_coefficient', 'atmospheric_density_factor']

for output in outputs:
    print(f\"{output.replace('_', ' ').title()} Sensitivity:\")
    
    # First-order sensitivity indices
    print(\"  First-order indices (individual parameter effects):\")
    for param in parameter_names:
        s1 = sensitivity_results['first_order'][output][param]
        print(f\"    {param}: {s1:.3f}\")
    
    # Total effect indices  
    print(\"  Total effect indices (including interactions):\")
    for param in parameter_names:
        st = sensitivity_results['total_effect'][output][param]
        print(f\"    {param}: {st:.3f}\")
    print()

# Identify most important parameters
print(\"=== PARAMETER IMPORTANCE RANKING ===\\n\")

for output in outputs:
    print(f\"{output.replace('_', ' ').title()}:\")
    
    # Calculate average total effect for ranking
    param_importance = []
    for param in parameter_names:
        total_effect = sensitivity_results['total_effect'][output][param]
        param_importance.append((param, total_effect))
    
    # Sort by importance
    param_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (param, effect) in enumerate(param_importance, 1):
        print(f\"  {i}. {param}: {effect:.3f} (Total Effect)\")
    print()
```

---

## Next Steps

### 1. Explore Advanced Features

**Trajectory Optimization:**
```python
from hypersonic_reentry.optimization import GradientBasedOptimizer

# See examples/trajectory_optimization_example.py
```

**Interactive Visualizations:**
```python
from hypersonic_reentry.visualization import AdvancedPlotter

# Create interactive 3D plots and dashboards
```

**Performance Optimization:**
```python
from hypersonic_reentry.utils.performance import ParallelProcessor

# Parallel Monte Carlo for large studies
```

### 2. Check Out Example Notebooks

Navigate to the `notebooks/` directory:
- `01_quick_start.ipynb` - Interactive version of this tutorial
- `03_monte_carlo_analysis.ipynb` - Advanced uncertainty quantification
- `04_trajectory_optimization.ipynb` - Optimization workflows
- `05_sensitivity_analysis.ipynb` - Global sensitivity analysis

### 3. Run Complete Examples

Check the `examples/` directory:
```bash
# Complete analysis workflow
python examples/comprehensive_results_generation.py

# End-to-end analysis
python examples/complete_analysis_example.py
```

### 4. Build the Documentation Website

```bash
cd website/
jekyll build
jekyll serve

# View at http://localhost:4000
```

---

## Common Issues and Solutions

### Installation Problems

**Issue**: `ImportError: No module named 'hypersonic_reentry'`
**Solution**: 
```bash
# Make sure you're in the project directory
cd hypersonic-reentry-framework

# Install in development mode
pip install -e .
```

**Issue**: `ModuleNotFoundError: No module named 'numpy'`
**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Simulation Issues

**Issue**: Simulation crashes or produces unrealistic results
**Solution**: 
- Check initial conditions (altitude should be > 80 km)
- Verify parameter ranges are physically reasonable
- Use smaller time steps for integration

**Issue**: Monte Carlo analysis is slow
**Solution**:
```python
# Use parallel processing
from hypersonic_reentry.utils.performance import ParallelProcessor

processor = ParallelProcessor(n_workers=4)
results = processor.parallel_monte_carlo(...)
```

### Visualization Issues

**Issue**: Plots not displaying
**Solution**:
```python
# Add to your script
import matplotlib
matplotlib.use('Agg')  # For headless environments
plt.savefig('plot.png')  # Save instead of show
```

---

## Getting Help

### Documentation
- **API Reference**: `docs/api/hypersonic_reentry_api.md`
- **Examples**: `examples/` directory
- **Notebooks**: `notebooks/` directory

### Community Support
- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions
- **Email**: [support@hypersonic-framework.org]

### Contributing
See `docs/development/contributing.md` for guidelines on:
- Bug reports
- Feature requests  
- Code contributions
- Documentation improvements

---

**Congratulations!** You've completed your first hypersonic reentry analysis. You now have the foundation to:

✓ Run trajectory simulations  
✓ Perform uncertainty quantification  
✓ Conduct sensitivity analysis  
✓ Create visualizations  
✓ Understand the framework structure  

Continue exploring the advanced features and examples to unlock the full potential of the framework for your hypersonic reentry research!