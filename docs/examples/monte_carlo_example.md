# Monte Carlo Analysis Example

## Overview

This example demonstrates how to perform comprehensive Monte Carlo uncertainty analysis for hypersonic reentry trajectories, including parameter setup, simulation execution, statistical analysis, and visualization.

---

## Prerequisites

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Framework imports
from hypersonic_reentry.uncertainty import UncertaintyQuantifier, UncertainParameter
from hypersonic_reentry.dynamics import VehicleDynamics, VehicleState
from hypersonic_reentry.atmosphere import USStandard1976
from hypersonic_reentry.analysis import StatisticalAnalyzer
from hypersonic_reentry.visualization import PlotManager
from hypersonic_reentry.utils.performance import ParallelProcessor
```

---

## 1. Parameter Definition

### 1.1 Define Uncertain Parameters

```python
def setup_uncertain_parameters():
    \"\"\"Define uncertain parameters for Monte Carlo analysis\"\"\"
    
    parameters = [
        # Vehicle mass uncertainty (±5% = ±250 kg for 5000 kg vehicle)
        UncertainParameter(
            name='mass',
            nominal_value=5000.0,          # kg
            uncertainty=250.0,             # kg (1-sigma)
            distribution='normal',
            bounds=(4000.0, 6000.0),
            description='Vehicle dry mass uncertainty due to manufacturing tolerances'
        ),
        
        # Drag coefficient uncertainty (±10%)
        UncertainParameter(
            name='drag_coefficient', 
            nominal_value=1.2,             # dimensionless
            uncertainty=0.12,              # 1-sigma
            distribution='normal',
            bounds=(0.8, 1.8),
            description='Drag coefficient uncertainty from CFD and wind tunnel data'
        ),
        
        # Lift coefficient uncertainty (±10%)  
        UncertainParameter(
            name='lift_coefficient',
            nominal_value=0.8,             # dimensionless
            uncertainty=0.08,              # 1-sigma
            distribution='normal',
            bounds=(0.5, 1.2),
            description='Lift coefficient uncertainty from aerodynamic modeling'
        ),
        
        # Reference area uncertainty (±5%)
        UncertainParameter(
            name='reference_area',
            nominal_value=15.0,            # m²
            uncertainty=0.75,              # m² (1-sigma)
            distribution='normal',
            bounds=(13.0, 17.0),
            description='Reference area geometric uncertainty'
        ),
        
        # Atmospheric density scaling factor (±15% log-normal)
        UncertainParameter(
            name='atmospheric_density_factor',
            nominal_value=1.0,             # dimensionless
            uncertainty=0.15,              # 1-sigma for underlying normal
            distribution='lognormal',
            bounds=(0.6, 1.6),
            description='Atmospheric density uncertainty due to weather variations'
        ),
        
        # Entry velocity uncertainty (±2% navigation accuracy)
        UncertainParameter(
            name='entry_velocity',
            nominal_value=7500.0,          # m/s
            uncertainty=150.0,             # m/s (1-sigma)
            distribution='normal', 
            bounds=(7000.0, 8000.0),
            description='Entry velocity uncertainty from navigation system'
        ),
        
        # Entry flight path angle uncertainty (±0.5° navigation)
        UncertainParameter(
            name='entry_flight_path_angle',
            nominal_value=-0.087,          # rad (-5°)
            uncertainty=0.0087,           # rad (±0.5°, 1-sigma)
            distribution='normal',
            bounds=(-0.175, -0.017),       # ±10° to ±1°
            description='Entry angle uncertainty from navigation system'
        )
    ]
    
    return parameters

# Initialize parameters
uncertain_parameters = setup_uncertain_parameters()

print(f\"Defined {len(uncertain_parameters)} uncertain parameters:\")
for param in uncertain_parameters:
    print(f\"  {param.name}: {param.nominal_value} ± {param.uncertainty} ({param.distribution})\")
```

### 1.2 Parameter Correlations (Optional)

```python
def define_parameter_correlations():
    \"\"\"Define correlations between uncertain parameters\"\"\"
    
    correlations = {
        # Mass and reference area are positively correlated
        # (larger vehicles tend to be heavier)
        ('mass', 'reference_area'): 0.4,
        
        # Drag and lift coefficients are correlated due to 
        # shared aerodynamic uncertainties
        ('drag_coefficient', 'lift_coefficient'): 0.6,
        
        # Entry velocity and angle may be correlated due to
        # trajectory planning constraints
        ('entry_velocity', 'entry_flight_path_angle'): -0.2
    }
    
    return correlations

correlations = define_parameter_correlations()
```

---

## 2. Simulation Function

### 2.1 Trajectory Simulation

```python
class ReentrySimulator:
    \"\"\"Hypersonic reentry trajectory simulator\"\"\"
    
    def __init__(self):
        self.atmosphere = USStandard1976()
        
    def simulate_reentry(self, parameters):
        \"\"\"
        Simulate reentry trajectory with uncertain parameters
        
        Parameters:
        -----------
        parameters : dict
            Dictionary of parameter values for this Monte Carlo sample
            
        Returns:
        --------
        dict : Simulation results and quantities of interest
        \"\"\"
        
        try:
            # Extract parameters
            mass = parameters.get('mass', 5000.0)
            C_D = parameters.get('drag_coefficient', 1.2)
            C_L = parameters.get('lift_coefficient', 0.8)
            S_ref = parameters.get('reference_area', 15.0)
            rho_factor = parameters.get('atmospheric_density_factor', 1.0)
            V_entry = parameters.get('entry_velocity', 7500.0)
            gamma_entry = parameters.get('entry_flight_path_angle', -0.087)
            
            # Initialize vehicle dynamics
            vehicle = VehicleDynamics(
                mass=mass,
                reference_area=S_ref,
                drag_coefficient=C_D,
                lift_coefficient=C_L
            )
            
            # Set initial state
            initial_state = VehicleState(
                longitude=0.0,                    # rad
                latitude=0.0,                     # rad  
                altitude=120000.0,                # m
                velocity=V_entry,                 # m/s
                flight_path_angle=gamma_entry,    # rad
                heading_angle=0.0                 # rad
            )
            
            # Integrate trajectory
            trajectory = self._integrate_trajectory(
                vehicle, initial_state, rho_factor
            )
            
            # Compute quantities of interest
            qoi = self._compute_quantities_of_interest(trajectory)
            
            # Add success flag
            qoi['simulation_success'] = True
            qoi['failure_reason'] = None
            
            return qoi
            
        except Exception as e:
            # Return failure case
            return {
                'final_altitude': np.nan,
                'final_velocity': np.nan,
                'downrange_distance': np.nan,
                'crossrange_distance': np.nan,
                'flight_time': np.nan,
                'max_deceleration': np.nan,
                'max_heat_rate': np.nan,
                'max_dynamic_pressure': np.nan,
                'impact_velocity': np.nan,
                'simulation_success': False,
                'failure_reason': str(e)
            }
    
    def _integrate_trajectory(self, vehicle, initial_state, density_factor):
        \"\"\"Integrate trajectory using simplified dynamics\"\"\"
        
        # Simplified integration for example
        # In practice, use scipy.integrate.solve_ivp with full dynamics
        
        dt = 1.0  # time step (s)
        max_time = 3000.0  # maximum simulation time
        
        # Initialize trajectory storage
        times = [0.0]
        states = [initial_state]
        
        current_state = initial_state
        t = 0.0
        
        while t < max_time and current_state.altitude > 0:
            # Get atmospheric density
            rho_nominal = self.atmosphere.get_density(current_state.altitude)
            rho = rho_nominal * density_factor
            
            # Compute aerodynamic forces
            drag_force, lift_force, _ = vehicle.get_aerodynamic_forces(
                current_state, rho
            )
            
            # Simple point-mass integration (Euler method for example)
            # Real implementation would use RK4 or adaptive methods
            
            # Gravitational acceleration
            R_E = 6.371e6  # Earth radius
            mu = 3.986e14  # Gravitational parameter
            g = mu / (R_E + current_state.altitude)**2
            
            # Equations of motion (simplified)
            dV_dt = -g * np.sin(current_state.flight_path_angle) - drag_force/vehicle.mass
            dgamma_dt = (lift_force/(vehicle.mass * current_state.velocity) - 
                        g * np.cos(current_state.flight_path_angle)/current_state.velocity +
                        current_state.velocity * np.cos(current_state.flight_path_angle)/(R_E + current_state.altitude))
            dh_dt = current_state.velocity * np.sin(current_state.flight_path_angle)
            dlat_dt = current_state.velocity * np.cos(current_state.flight_path_angle) / (R_E + current_state.altitude)
            
            # Update state
            new_velocity = current_state.velocity + dV_dt * dt
            new_gamma = current_state.flight_path_angle + dgamma_dt * dt
            new_altitude = current_state.altitude + dh_dt * dt
            new_latitude = current_state.latitude + dlat_dt * dt
            
            # Create new state
            current_state = VehicleState(
                longitude=current_state.longitude,
                latitude=new_latitude,
                altitude=max(0.0, new_altitude),  # Don't go below ground
                velocity=max(0.0, new_velocity),  # Don't go negative
                flight_path_angle=new_gamma,
                heading_angle=current_state.heading_angle
            )
            
            # Store trajectory point
            t += dt
            times.append(t)
            states.append(current_state)
        
        return {'times': times, 'states': states}
    
    def _compute_quantities_of_interest(self, trajectory):
        \"\"\"Extract quantities of interest from trajectory\"\"\"
        
        times = trajectory['times']
        states = trajectory['states']
        
        # Final state quantities
        final_state = states[-1]
        final_altitude = final_state.altitude
        final_velocity = final_state.velocity
        flight_time = times[-1]
        
        # Downrange and crossrange
        R_E = 6.371e6
        downrange = R_E * final_state.latitude  # Small angle approximation
        crossrange = R_E * final_state.longitude * np.cos(final_state.latitude)
        
        # Maximum values during flight
        altitudes = [state.altitude for state in states]
        velocities = [state.velocity for state in states]
        
        # Heat rate calculation (simplified Sutton-Graves)
        heat_rates = []
        decelerations = []
        dynamic_pressures = []
        
        for state in states:
            # Atmospheric density
            rho = self.atmosphere.get_density(state.altitude)
            
            # Heat rate (W/m²) - Sutton-Graves equation
            heat_rate = 1.83e-4 * np.sqrt(rho) * (state.velocity/1000.0)**3
            heat_rates.append(heat_rate)
            
            # Dynamic pressure
            q = 0.5 * rho * state.velocity**2
            dynamic_pressures.append(q)
            
            # Deceleration (simplified - would need acceleration from dynamics)
            # Using approximation based on drag
            drag_decel = 0.5 * rho * state.velocity**2 * 1.2 * 15.0 / 5000.0  # Approximate
            deceleration_g = drag_decel / 9.81
            decelerations.append(deceleration_g)
        
        # Impact conditions (assuming ground impact)
        impact_velocity = final_velocity if final_altitude <= 1000 else np.nan
        
        return {
            'final_altitude': final_altitude,                    # m
            'final_velocity': final_velocity,                    # m/s  
            'downrange_distance': downrange,                     # m
            'crossrange_distance': crossrange,                   # m
            'flight_time': flight_time,                         # s
            'max_deceleration': max(decelerations) if decelerations else 0.0,  # g
            'max_heat_rate': max(heat_rates) if heat_rates else 0.0,           # W/m²
            'max_dynamic_pressure': max(dynamic_pressures) if dynamic_pressures else 0.0,  # Pa
            'impact_velocity': impact_velocity,                  # m/s
            'peak_heat_rate_altitude': altitudes[np.argmax(heat_rates)] if heat_rates else 0.0,  # m
            'peak_decel_altitude': altitudes[np.argmax(decelerations)] if decelerations else 0.0  # m
        }

# Create simulator instance
simulator = ReentrySimulator()

def simulation_wrapper(param_dict):
    \"\"\"Wrapper function for Monte Carlo analysis\"\"\"
    return simulator.simulate_reentry(param_dict)
```

---

## 3. Monte Carlo Analysis

### 3.1 Setup and Execution

```python
# Initialize uncertainty quantification framework
uq = UncertaintyQuantifier(
    parameters=uncertain_parameters,
    correlations=correlations
)

print(\"Starting Monte Carlo Analysis...\")
print(f\"Number of uncertain parameters: {len(uncertain_parameters)}\")
print(f\"Parameter correlations defined: {len(correlations)}\")

# Monte Carlo settings
num_samples = 1000
sampling_method = 'latin_hypercube'  # Options: 'random', 'latin_hypercube', 'sobol'
random_seed = 42  # For reproducibility

print(f\"\\nMonte Carlo Configuration:\")
print(f\"  Samples: {num_samples}\")
print(f\"  Sampling method: {sampling_method}\")
print(f\"  Random seed: {random_seed}\")

# Run Monte Carlo analysis
import time
start_time = time.time()

mc_results = uq.monte_carlo_analysis(
    num_samples=num_samples,
    simulation_function=simulation_wrapper,
    method=sampling_method,
    random_seed=random_seed,
    parallel=True,        # Enable parallel processing
    n_workers=4,          # Number of CPU cores to use
    verbose=True          # Show progress
)

execution_time = time.time() - start_time

print(f\"\\n✅ Monte Carlo analysis completed!\")
print(f\"   Execution time: {execution_time:.1f} seconds\")
print(f\"   Throughput: {num_samples/execution_time:.1f} samples/second\")
```

### 3.2 Success Rate Analysis

```python
# Analyze simulation success rate
successful_sims = sum(1 for result in mc_results['results'] if result['simulation_success'])
success_rate = successful_sims / num_samples

print(f\"\\n📊 Simulation Success Analysis:\")
print(f\"   Successful simulations: {successful_sims}/{num_samples}\")
print(f\"   Success rate: {success_rate:.1%}\")

# Analyze failure modes
if successful_sims < num_samples:
    failures = [result for result in mc_results['results'] if not result['simulation_success']]
    failure_reasons = {}
    
    for failure in failures:
        reason = failure.get('failure_reason', 'Unknown')
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    print(f\"\\n   Failure modes:\")
    for reason, count in failure_reasons.items():
        print(f\"     {reason}: {count} ({count/num_samples:.1%})\")

# Filter successful results for analysis
successful_results = [result for result in mc_results['results'] if result['simulation_success']]
print(f\"\\n   Proceeding with {len(successful_results)} successful simulations\")
```

---

## 4. Statistical Analysis

### 4.1 Descriptive Statistics

```python
# Extract quantities of interest from successful simulations
outputs = [
    'final_altitude', 'final_velocity', 'downrange_distance', 'flight_time',
    'max_deceleration', 'max_heat_rate', 'max_dynamic_pressure'
]

# Create data arrays
output_data = {}
for output in outputs:
    data = [result[output] for result in successful_results if not np.isnan(result[output])]
    output_data[output] = np.array(data)

# Compute descriptive statistics
print(f\"\\n📈 Descriptive Statistics:\")
print(f\"{'Quantity':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'CV (%)':<8}\")
print(\"-\" * 80)

stats_summary = {}
for output in outputs:
    if output in output_data and len(output_data[output]) > 0:
        data = output_data[output]
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data) 
        max_val = np.max(data)
        cv = 100 * std_val / mean_val if mean_val != 0 else 0
        
        stats_summary[output] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'cv': cv,
            'count': len(data)
        }
        
        # Format for display
        if 'altitude' in output or 'distance' in output:
            # Convert to km for display
            print(f\"{output:<25} {mean_val/1000:<12.1f} {std_val/1000:<12.1f} {min_val/1000:<12.1f} {max_val/1000:<12.1f} {cv:<8.1f}\")
        elif 'velocity' in output:
            print(f\"{output:<25} {mean_val:<12.1f} {std_val:<12.1f} {min_val:<12.1f} {max_val:<12.1f} {cv:<8.1f}\")
        elif 'heat_rate' in output:
            # Convert to MW/m²
            print(f\"{output:<25} {mean_val/1e6:<12.2f} {std_val/1e6:<12.2f} {min_val/1e6:<12.2f} {max_val/1e6:<12.2f} {cv:<8.1f}\")
        else:
            print(f\"{output:<25} {mean_val:<12.2f} {std_val:<12.2f} {min_val:<12.2f} {max_val:<12.2f} {cv:<8.1f}\")
```

### 4.2 Confidence Intervals

```python
# Compute confidence intervals
confidence_levels = [0.90, 0.95, 0.99]

print(f\"\\n🎯 Confidence Intervals:\")

for output in outputs:
    if output in output_data and len(output_data[output]) > 10:
        data = output_data[output]
        print(f\"\\n{output.replace('_', ' ').title()}:\")
        
        for level in confidence_levels:
            # Percentile method
            alpha = 1 - level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            
            lower = np.percentile(data, lower_percentile)
            upper = np.percentile(data, upper_percentile)
            
            # Format based on quantity type
            if 'altitude' in output or 'distance' in output:
                print(f\"  {level*100:2.0f}% CI: [{lower/1000:.1f}, {upper/1000:.1f}] km\")
            elif 'heat_rate' in output:
                print(f\"  {level*100:2.0f}% CI: [{lower/1e6:.2f}, {upper/1e6:.2f}] MW/m²\")
            else:
                print(f\"  {level*100:2.0f}% CI: [{lower:.1f}, {upper:.1f}]\")
```

### 4.3 Distribution Fitting

```python
# Fit probability distributions to key outputs
from scipy import stats

# Test distributions
distributions_to_test = [
    ('normal', stats.norm),
    ('lognormal', stats.lognorm), 
    ('weibull', stats.weibull_min),
    ('gamma', stats.gamma)
]

print(f\"\\n🔍 Distribution Fitting Analysis:\")

distribution_fits = {}

for output in ['downrange_distance', 'max_heat_rate', 'flight_time']:
    if output in output_data and len(output_data[output]) > 50:
        data = output_data[output]
        
        print(f\"\\n{output.replace('_', ' ').title()}:\")
        
        best_fit = None
        best_aic = np.inf
        fit_results = {}
        
        for dist_name, distribution in distributions_to_test:
            try:
                # Fit distribution parameters
                if dist_name == 'normal':
                    params = distribution.fit(data)
                    fitted_dist = distribution(*params)
                elif dist_name == 'lognormal':
                    # Fit to log of data
                    log_data = np.log(data[data > 0])  # Remove zeros/negatives
                    if len(log_data) > 10:
                        params = distribution.fit(data, floc=0)
                        fitted_dist = distribution(*params)
                    else:
                        continue
                else:
                    params = distribution.fit(data, floc=0)
                    fitted_dist = distribution(*params)
                
                # Compute log-likelihood and AIC
                log_likelihood = np.sum(fitted_dist.logpdf(data))
                aic = 2 * len(params) - 2 * log_likelihood
                
                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = stats.kstest(data, fitted_dist.cdf)
                
                fit_results[dist_name] = {
                    'params': params,
                    'aic': aic,
                    'ks_statistic': ks_statistic,
                    'ks_p_value': ks_p_value,
                    'fitted_dist': fitted_dist
                }
                
                print(f\"  {dist_name:12} AIC: {aic:10.1f}, KS p-value: {ks_p_value:.3f}\")
                
                if aic < best_aic:
                    best_aic = aic
                    best_fit = dist_name
                    
            except Exception as e:
                print(f\"  {dist_name:12} Failed to fit: {e}\")
        
        if best_fit:
            print(f\"  Best fit: {best_fit} (AIC = {best_aic:.1f})\")
            distribution_fits[output] = fit_results
        
        # Store best fit in results
        if best_fit and best_fit in fit_results:
            stats_summary[output]['best_distribution'] = best_fit
            stats_summary[output]['distribution_params'] = fit_results[best_fit]['params']
            stats_summary[output]['distribution_aic'] = fit_results[best_fit]['aic']
```

---

## 5. Risk and Reliability Analysis

### 5.1 Mission Success Criteria

```python
# Define mission success criteria
success_criteria = {
    'final_altitude': {'min': 15000, 'max': 45000, 'unit': 'm', 'description': 'Final altitude in acceptable range'},
    'max_heat_rate': {'min': 0, 'max': 5e6, 'unit': 'W/m²', 'description': 'Heat rate below TPS limit'},
    'max_deceleration': {'min': 0, 'max': 12, 'unit': 'g', 'description': 'Deceleration below crew limit'},
    'downrange_distance': {'min': 1500000, 'max': 2500000, 'unit': 'm', 'description': 'Landing accuracy requirement'}
}

print(f\"\\n🎯 Mission Success Analysis:\")

# Evaluate each criterion
criterion_success_rates = {}
overall_success_count = 0

for criterion, limits in success_criteria.items():
    if criterion in output_data and len(output_data[criterion]) > 0:
        data = output_data[criterion]
        
        # Check how many samples meet the criterion
        if 'min' in limits and 'max' in limits:
            success_mask = (data >= limits['min']) & (data <= limits['max'])
        elif 'min' in limits:
            success_mask = data >= limits['min']
        elif 'max' in limits:
            success_mask = data <= limits['max']
        else:
            continue
        
        success_count = np.sum(success_mask)
        success_rate = success_count / len(data)
        criterion_success_rates[criterion] = success_rate
        
        print(f\"  {criterion:<20}: {success_rate:>6.1%} ({success_count}/{len(data)}) - {limits['description']}\")

# Overall mission success (all criteria met)
if len(criterion_success_rates) > 0:
    # Compute joint success probability
    all_criteria_met = np.ones(len(successful_results), dtype=bool)
    
    for i, result in enumerate(successful_results):
        for criterion, limits in success_criteria.items():
            if criterion in result:
                value = result[criterion]
                
                # Check criterion
                if 'min' in limits and value < limits['min']:
                    all_criteria_met[i] = False
                if 'max' in limits and value > limits['max']:
                    all_criteria_met[i] = False
    
    overall_success_rate = np.mean(all_criteria_met)
    print(f\"\\n  Overall Mission Success: {overall_success_rate:>6.1%}\")
```

### 5.2 Extreme Value Analysis

```python
# Analyze extreme values for critical parameters
print(f\"\\n⚠️  Extreme Value Analysis:\")

extreme_percentiles = [1, 5, 95, 99]  # 1st, 5th, 95th, 99th percentiles

for output in ['max_heat_rate', 'max_deceleration', 'downrange_distance']:
    if output in output_data and len(output_data[output]) > 0:
        data = output_data[output]
        
        print(f\"\\n{output.replace('_', ' ').title()}:\")
        
        for percentile in extreme_percentiles:
            value = np.percentile(data, percentile)
            
            if 'heat_rate' in output:
                print(f\"  {percentile:2d}th percentile: {value/1e6:.2f} MW/m²\")
            elif 'distance' in output:
                print(f\"  {percentile:2d}th percentile: {value/1000:.1f} km\")
            else:
                print(f\"  {percentile:2d}th percentile: {value:.2f}\")
        
        # Probability of exceeding critical thresholds
        if output == 'max_heat_rate':
            threshold = 5e6  # 5 MW/m²
            exceedance_prob = np.mean(data > threshold)
            print(f\"  Probability of exceeding {threshold/1e6:.1f} MW/m²: {exceedance_prob:.1%}\")
            
        elif output == 'max_deceleration':
            threshold = 12  # 12 g's  
            exceedance_prob = np.mean(data > threshold)
            print(f\"  Probability of exceeding {threshold:.0f} g: {exceedance_prob:.1%}\")
```

---

## 6. Visualization

### 6.1 Distribution Plots

```python
# Create comprehensive visualization
plt.style.use('seaborn-v0_8')  # Use seaborn style for better appearance
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot key outputs
plot_outputs = ['final_altitude', 'downrange_distance', 'flight_time', 
               'max_heat_rate', 'max_deceleration', 'final_velocity']

for i, output in enumerate(plot_outputs):
    if i < len(axes) and output in output_data:
        ax = axes[i]
        data = output_data[output]
        
        # Histogram
        n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, 
                                  edgecolor='black', linewidth=0.5)
        
        # Overlay fitted distribution if available
        if output in distribution_fits:
            best_dist_name = stats_summary[output].get('best_distribution')
            if best_dist_name and best_dist_name in distribution_fits[output]:
                fitted_dist = distribution_fits[output][best_dist_name]['fitted_dist']
                x_range = np.linspace(np.min(data), np.max(data), 100)
                y_fitted = fitted_dist.pdf(x_range)
                ax.plot(x_range, y_fitted, 'r-', linewidth=2, 
                       label=f'{best_dist_name.title()} Fit')
                ax.legend()
        
        # Formatting
        if 'altitude' in output or 'distance' in output:
            ax.set_xlabel(f'{output.replace(\"_\", \" \").title()} (km)')
            # Convert x-axis to km
            ax.set_xticklabels([f'{x/1000:.0f}' for x in ax.get_xticks()])
        elif 'heat_rate' in output:
            ax.set_xlabel(f'{output.replace(\"_\", \" \").title()} (MW/m²)')
            ax.set_xticklabels([f'{x/1e6:.1f}' for x in ax.get_xticks()])
        else:
            ax.set_xlabel(output.replace('_', ' ').title())
            
        ax.set_ylabel('Probability Density')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        mean_val = np.mean(data)
        std_val = np.std(data)
        if 'altitude' in output or 'distance' in output:
            stats_text = f'μ = {mean_val/1000:.1f} km\\nσ = {std_val/1000:.1f} km'
        elif 'heat_rate' in output:
            stats_text = f'μ = {mean_val/1e6:.2f} MW/m²\\nσ = {std_val/1e6:.2f} MW/m²'
        else:
            stats_text = f'μ = {mean_val:.1f}\\nσ = {std_val:.1f}'
            
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Remove empty subplots
for j in range(len(plot_outputs), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Monte Carlo Results - Output Distributions', y=1.02, fontsize=16)
plt.savefig('monte_carlo_distributions.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

### 6.2 Correlation Analysis

```python
# Create correlation matrix for input parameters and key outputs
correlation_variables = []
correlation_data = []

# Input parameters
input_params = ['mass', 'drag_coefficient', 'lift_coefficient', 'reference_area', 
               'atmospheric_density_factor', 'entry_velocity', 'entry_flight_path_angle']

for param in input_params:
    param_values = [result.get('input_parameters', {}).get(param, np.nan) 
                   for result in successful_results]
    if not all(np.isnan(param_values)):
        correlation_variables.append(param.replace('_', ' ').title())
        correlation_data.append(param_values)

# Key outputs
output_params = ['downrange_distance', 'max_heat_rate', 'flight_time', 'final_altitude']
for output in output_params:
    if output in output_data:
        correlation_variables.append(output.replace('_', ' ').title()) 
        correlation_data.append(output_data[output])

# Compute correlation matrix
if len(correlation_data) > 1:
    correlation_matrix = np.corrcoef(correlation_data)
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    im = plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=20)
    
    # Set ticks and labels
    plt.xticks(range(len(correlation_variables)), correlation_variables, rotation=45, ha='right')
    plt.yticks(range(len(correlation_variables)), correlation_variables)
    
    # Add correlation values as text
    for i in range(len(correlation_variables)):
        for j in range(len(correlation_variables)):
            text = plt.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha='center', va='center', 
                           color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
    
    plt.title('Parameter and Output Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.show()
```

### 6.3 Scatter Plot Analysis

```python
# Create scatter plots for key parameter-output relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Define parameter-output pairs to analyze
scatter_pairs = [
    ('atmospheric_density_factor', 'max_heat_rate', 'Atm. Density Factor', 'Max Heat Rate (MW/m²)'),
    ('drag_coefficient', 'downrange_distance', 'Drag Coefficient', 'Downrange (km)'),
    ('mass', 'final_velocity', 'Vehicle Mass (kg)', 'Final Velocity (m/s)'),
    ('entry_velocity', 'flight_time', 'Entry Velocity (m/s)', 'Flight Time (s)')
]

for i, (x_param, y_param, x_label, y_label) in enumerate(scatter_pairs):
    if i < len(axes):
        ax = axes[i]
        
        # Extract data
        x_data = []
        y_data = []
        
        for result in successful_results:
            x_val = result.get('input_parameters', {}).get(x_param)
            y_val = result.get(y_param)
            
            if x_val is not None and y_val is not None and not np.isnan(y_val):
                x_data.append(x_val)
                if 'heat_rate' in y_param:
                    y_data.append(y_val / 1e6)  # Convert to MW/m²
                elif 'distance' in y_param:
                    y_data.append(y_val / 1000)  # Convert to km
                else:
                    y_data.append(y_val)
        
        if len(x_data) > 10:
            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.6, s=20)
            
            # Fit linear trend line
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), \"r--\", alpha=0.8, linewidth=2)
            
            # Compute and display correlation
            correlation = np.corrcoef(x_data, y_data)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Parameter-Output Relationships', y=1.02, fontsize=16)
plt.savefig('parameter_scatter_plots.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 7. Results Summary and Export

### 7.1 Summary Report

```python
# Generate summary report
print(f\"\\n📋 MONTE CARLO ANALYSIS SUMMARY REPORT\")
print(\"=\" * 60)

print(f\"\\n🔧 Analysis Configuration:\")
print(f\"  Study Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\")
print(f\"  Number of Samples: {num_samples}\")
print(f\"  Sampling Method: {sampling_method}\")
print(f\"  Success Rate: {success_rate:.1%}\")
print(f\"  Execution Time: {execution_time:.1f} seconds\")

print(f\"\\n📊 Key Statistical Results:\")
for output in ['downrange_distance', 'max_heat_rate', 'flight_time', 'final_altitude']:
    if output in stats_summary:
        stats = stats_summary[output]
        print(f\"\\n  {output.replace('_', ' ').title()}:\")
        
        if 'altitude' in output or 'distance' in output:
            print(f\"    Mean: {stats['mean']/1000:.1f} ± {stats['std']/1000:.1f} km\")
            print(f\"    Range: [{stats['min']/1000:.1f}, {stats['max']/1000:.1f}] km\")
        elif 'heat_rate' in output:
            print(f\"    Mean: {stats['mean']/1e6:.2f} ± {stats['std']/1e6:.2f} MW/m²\")
            print(f\"    Range: [{stats['min']/1e6:.2f}, {stats['max']/1e6:.2f}] MW/m²\")
        else:
            print(f\"    Mean: {stats['mean']:.1f} ± {stats['std']:.1f}\")
            print(f\"    Range: [{stats['min']:.1f}, {stats['max']:.1f}]\")
            
        print(f\"    Coefficient of Variation: {stats['cv']:.1f}%\")
        
        if 'best_distribution' in stats:
            print(f\"    Best-fit Distribution: {stats['best_distribution'].title()}\")

print(f\"\\n🎯 Mission Success Analysis:\")
for criterion, rate in criterion_success_rates.items():
    print(f\"  {criterion.replace('_', ' ').title()}: {rate:.1%}\")
print(f\"  Overall Mission Success: {overall_success_rate:.1%}\")

print(f\"\\n⚠️  Risk Assessment:\")
# Heat rate risk
if 'max_heat_rate' in output_data:
    heat_data = output_data['max_heat_rate']
    heat_5mw_prob = np.mean(heat_data > 5e6)
    heat_4mw_prob = np.mean(heat_data > 4e6)
    print(f\"  Probability of exceeding 5 MW/m² heat rate: {heat_5mw_prob:.1%}\")
    print(f\"  Probability of exceeding 4 MW/m² heat rate: {heat_4mw_prob:.1%}\")

# Deceleration risk  
if 'max_deceleration' in output_data:
    decel_data = output_data['max_deceleration']
    decel_12g_prob = np.mean(decel_data > 12)
    decel_10g_prob = np.mean(decel_data > 10)
    print(f\"  Probability of exceeding 12g deceleration: {decel_12g_prob:.1%}\")
    print(f\"  Probability of exceeding 10g deceleration: {decel_10g_prob:.1%}\")
```

### 7.2 Data Export

```python
# Export results to various formats
import pandas as pd

# Create comprehensive results DataFrame
results_df = pd.DataFrame()

# Add input parameters
for param in uncertain_parameters:
    param_values = []
    for result in successful_results:
        param_val = result.get('input_parameters', {}).get(param.name, np.nan)
        param_values.append(param_val)
    results_df[f'input_{param.name}'] = param_values

# Add output quantities
for output in outputs:
    if output in output_data:
        # Pad with NaN if needed
        output_values = [result.get(output, np.nan) for result in successful_results]
        results_df[f'output_{output}'] = output_values

# Export to CSV
results_df.to_csv('monte_carlo_results.csv', index=False)
print(f\"\\n💾 Results exported to monte_carlo_results.csv\")

# Export to Excel with multiple sheets
with pd.ExcelWriter('monte_carlo_analysis.xlsx') as writer:
    # Raw results
    results_df.to_excel(writer, sheet_name='Raw_Results', index=False)
    
    # Summary statistics
    summary_df = pd.DataFrame({
        'Parameter': list(stats_summary.keys()),
        'Mean': [stats['mean'] for stats in stats_summary.values()],
        'Std': [stats['std'] for stats in stats_summary.values()],
        'Min': [stats['min'] for stats in stats_summary.values()],
        'Max': [stats['max'] for stats in stats_summary.values()],
        'CV_percent': [stats['cv'] for stats in stats_summary.values()]
    })
    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    # Success rates
    success_df = pd.DataFrame({
        'Criterion': list(criterion_success_rates.keys()) + ['Overall_Mission'],
        'Success_Rate': list(criterion_success_rates.values()) + [overall_success_rate]
    })
    success_df.to_excel(writer, sheet_name='Success_Analysis', index=False)

print(f\"💾 Comprehensive analysis exported to monte_carlo_analysis.xlsx\")

# Export configuration and metadata
config_data = {
    'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'num_samples': num_samples,
    'sampling_method': sampling_method,
    'random_seed': random_seed,
    'execution_time_seconds': execution_time,
    'success_rate': success_rate,
    'uncertain_parameters': [
        {
            'name': param.name,
            'nominal_value': param.nominal_value,
            'uncertainty': param.uncertainty,
            'distribution': param.distribution,
            'bounds': param.bounds,
            'description': param.description
        }
        for param in uncertain_parameters
    ],
    'success_criteria': success_criteria,
    'mission_success_rate': overall_success_rate
}

import json
with open('monte_carlo_config.json', 'w') as f:
    json.dump(config_data, f, indent=2, default=str)

print(f\"💾 Configuration saved to monte_carlo_config.json\")
print(f\"\\n✅ Monte Carlo analysis completed successfully!\")
```

---

## 8. Advanced Analysis Extensions

### 8.1 Sensitivity Screening

```python
# Perform quick sensitivity screening using correlation analysis
print(f\"\\n🔍 Sensitivity Screening (Correlation-based):\")

# Identify most influential parameters for each output
for output in ['downrange_distance', 'max_heat_rate', 'flight_time']:
    if output in output_data:
        print(f\"\\n{output.replace('_', ' ').title()} - Most Influential Parameters:\")
        
        correlations = []
        for param in uncertain_parameters:
            param_values = [result.get('input_parameters', {}).get(param.name, np.nan) 
                           for result in successful_results]
            
            if not all(np.isnan(param_values)):
                # Compute correlation
                valid_indices = [i for i, (x, y) in enumerate(zip(param_values, output_data[output])) 
                               if not (np.isnan(x) or np.isnan(y))]
                
                if len(valid_indices) > 10:
                    x_vals = [param_values[i] for i in valid_indices]
                    y_vals = [output_data[output][i] for i in valid_indices]
                    
                    corr = abs(np.corrcoef(x_vals, y_vals)[0, 1])
                    correlations.append((param.name, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        for i, (param_name, corr) in enumerate(correlations[:5]):
            print(f\"  {i+1}. {param_name}: |r| = {corr:.3f}\")
```

### 8.2 Robustness Analysis

```python
# Analyze robustness by computing success rates for different uncertainty levels
print(f\"\\n🛡️  Robustness Analysis:\")

# Test with reduced uncertainty (50% of nominal)
print(f\"\\nRunning reduced uncertainty analysis (50% of nominal)...\")

reduced_params = []
for param in uncertain_parameters:
    reduced_param = UncertainParameter(
        name=param.name,
        nominal_value=param.nominal_value,
        uncertainty=param.uncertainty * 0.5,  # 50% of original uncertainty
        distribution=param.distribution,
        bounds=param.bounds
    )
    reduced_params.append(reduced_param)

# Quick analysis with fewer samples
uq_reduced = UncertaintyQuantifier(reduced_params)
mc_reduced = uq_reduced.monte_carlo_analysis(
    num_samples=200,  # Fewer samples for quick comparison
    simulation_function=simulation_wrapper,
    method='latin_hypercube',
    random_seed=42
)

# Compare success rates
reduced_successful = [r for r in mc_reduced['results'] if r['simulation_success']]
reduced_success_rate = len(reduced_successful) / 200

print(f\"  Reduced uncertainty success rate: {reduced_success_rate:.1%}\")
print(f\"  Original uncertainty success rate: {success_rate:.1%}\")
print(f\"  Robustness improvement: {(reduced_success_rate - success_rate)*100:.1f} percentage points\")

# Analyze which parameters contribute most to failures
if success_rate < 0.95:  # If we have significant failures
    print(f\"\\n  Analyzing failure modes...\")
    
    failed_results = [r for r in mc_results['results'] if not r['simulation_success']]
    successful_params = {}
    failed_params = {}
    
    # Collect parameter statistics for successful vs failed cases
    for param in uncertain_parameters:
        successful_vals = [r.get('input_parameters', {}).get(param.name, np.nan) 
                          for r in successful_results]
        failed_vals = [r.get('input_parameters', {}).get(param.name, np.nan) 
                      for r in failed_results]
        
        successful_vals = [v for v in successful_vals if not np.isnan(v)]
        failed_vals = [v for v in failed_vals if not np.isnan(v)]
        
        if len(successful_vals) > 5 and len(failed_vals) > 5:
            successful_params[param.name] = np.mean(successful_vals)
            failed_params[param.name] = np.mean(failed_vals)
            
            # Statistical test for difference
            from scipy.stats import ttest_ind
            statistic, p_value = ttest_ind(successful_vals, failed_vals)
            
            if p_value < 0.05:  # Significant difference
                diff = abs(successful_params[param.name] - failed_params[param.name])
                print(f\"    {param.name}: Significant difference (p = {p_value:.3f})\")
                print(f\"      Successful mean: {successful_params[param.name]:.3f}\")
                print(f\"      Failed mean: {failed_params[param.name]:.3f}\")
```

---

## Conclusion

This comprehensive Monte Carlo example demonstrates:

✅ **Complete Parameter Setup**: Uncertain parameters with appropriate distributions  
✅ **Robust Simulation**: Error handling and success rate monitoring  
✅ **Statistical Analysis**: Descriptive statistics, distribution fitting, confidence intervals  
✅ **Risk Assessment**: Mission success criteria and extreme value analysis  
✅ **Visualization**: Comprehensive plots for interpretation  
✅ **Data Export**: Multiple formats for further analysis  
✅ **Advanced Features**: Sensitivity screening and robustness analysis  

The framework provides all necessary tools for production-quality uncertainty quantification studies in hypersonic reentry analysis.