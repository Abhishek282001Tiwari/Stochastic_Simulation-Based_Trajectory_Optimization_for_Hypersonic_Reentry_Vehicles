# Notebooks Directory

This directory contains Jupyter notebooks demonstrating the hypersonic reentry framework capabilities and analysis workflows.

## Notebook Categories

### 1. **Getting Started**
- `01_quick_start.ipynb` - Basic framework usage and simple trajectory simulation
- `02_framework_overview.ipynb` - Comprehensive framework capabilities overview

### 2. **Analysis Workflows**
- `03_monte_carlo_analysis.ipynb` - Complete Monte Carlo uncertainty analysis
- `04_trajectory_optimization.ipynb` - Trajectory optimization with constraints
- `05_sensitivity_analysis.ipynb` - Parameter sensitivity and importance ranking

### 3. **Advanced Topics**
- `06_custom_vehicle_models.ipynb` - Creating custom vehicle and atmosphere models
- `07_publication_plots.ipynb` - Creating publication-quality visualizations
- `08_performance_optimization.ipynb` - Framework performance tuning and benchmarking

### 4. **Research Studies**
- `09_comprehensive_study.ipynb` - Complete research study example
- `10_parametric_studies.ipynb` - Design space exploration and trade-offs

## Usage

Start Jupyter server from the project root:

```bash
# Install Jupyter if not already available
pip install jupyter notebook

# Start server
jupyter notebook

# Navigate to notebooks/ directory and open desired notebook
```

## Requirements

Notebooks require the hypersonic reentry framework to be installed:

```bash
# Install framework in development mode
pip install -e .

# Install additional notebook dependencies
pip install jupyter matplotlib seaborn plotly ipywidgets
```

## Interactive Features

Notebooks include:
- Interactive parameter widgets for real-time exploration
- 3D trajectory visualizations with Plotly
- Statistical analysis with automatic plot generation
- Code examples ready to copy and modify

## Data Dependencies

Some notebooks require simulation data:
- Run `examples/comprehensive_results_generation.py` first to generate required datasets
- Alternatively, notebooks will generate small-scale data for demonstration

## Best Practices

1. **Clear Documentation**: Each notebook includes comprehensive markdown explanations
2. **Reproducible Results**: Set random seeds and document versions
3. **Performance Awareness**: Large-scale analysis may take significant time
4. **Error Handling**: Notebooks include try/catch blocks for robustness