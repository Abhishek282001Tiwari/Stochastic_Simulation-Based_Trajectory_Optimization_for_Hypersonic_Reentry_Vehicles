# Installation Guide

## Hypersonic Reentry Trajectory Optimization Framework

This guide provides comprehensive installation instructions for the Stochastic Simulation-Based Trajectory Optimization framework for Hypersonic Reentry Vehicles.

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows 10/11
- **Python**: 3.8 or higher
- **Memory**: 8 GB RAM minimum, 16 GB recommended
- **Storage**: 5 GB free space for installation and results
- **CPU**: Multi-core processor recommended (4+ cores)

### Recommended Configuration
- **Operating System**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.9 or 3.10
- **Memory**: 32 GB RAM or more
- **Storage**: SSD with 20 GB+ free space
- **CPU**: 8+ cores with hyperthreading
- **GPU**: Optional, for accelerated computations

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Create virtual environment
python -m venv hypersonic_env
source hypersonic_env/bin/activate  # Linux/macOS
# hypersonic_env\Scripts\activate   # Windows

# Install the framework
pip install -e .

# Install optional dependencies for enhanced functionality
pip install -r requirements-optional.txt
```

### Method 2: Conda Environment

```bash
# Create conda environment
conda create -n hypersonic python=3.9
conda activate hypersonic

# Install dependencies
conda install numpy scipy matplotlib plotly pandas h5py pyyaml
conda install -c conda-forge casadi

# Install the framework
pip install -e .
```

### Method 3: Development Installation

```bash
# Clone repository (if not already done)
git clone <repository_url>
cd Stochastic_Simulation-Based_Trajectory_Optimization_for_Hypersonic_Reentry_Vehicles

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

## Dependency Installation

### Core Dependencies
```bash
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.5.0
pip install plotly>=5.0.0
pip install pandas>=1.3.0
pip install h5py>=3.1.0
pip install pyyaml>=5.4.0
```

### Optimization Dependencies
```bash
pip install casadi>=3.5.5
pip install pyomo>=6.0
pip install ipopt  # May require system-level installation
```

### Uncertainty Quantification Dependencies
```bash
pip install scikit-learn>=1.0.0
pip install pymc3>=3.11.0
pip install emcee>=3.0.0
pip install SALib>=1.4.0
```

### Visualization Dependencies
```bash
pip install seaborn>=0.11.0
pip install bokeh>=2.4.0
pip install dash>=2.0.0
pip install jupyter>=1.0.0
```

## System-Level Dependencies

### Linux (Ubuntu/Debian)
```bash
# Install system packages
sudo apt-get update
sudo apt-get install -y build-essential gfortran
sudo apt-get install -y libopenblas-dev liblapack-dev
sudo apt-get install -y libhdf5-dev
sudo apt-get install -y coinor-libipopt-dev

# For plotting (optional)
sudo apt-get install -y texlive-latex-base texlive-latex-extra
```

### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install gcc openblas lapack hdf5
brew install ipopt

# For plotting (optional)
brew install --cask mactex
```

### Windows
- Install Microsoft Visual Studio Build Tools
- Install Intel MKL or OpenBLAS libraries
- Consider using Windows Subsystem for Linux (WSL) for better compatibility

## Verification

### Quick Test
```python
# Test basic import
import hypersonic_reentry
print(f"Framework version: {hypersonic_reentry.__version__}")

# Test core functionality
from hypersonic_reentry.dynamics import VehicleDynamics, VehicleState
from hypersonic_reentry.atmosphere import USStandard1976

# Create basic components
atmosphere = USStandard1976()
props = atmosphere.get_properties(50000.0)  # 50 km altitude
print(f"Atmospheric properties at 50 km: {props}")
```

### Comprehensive Validation
```bash
# Run validation suite (may take 10-30 minutes)
python tests/test_comprehensive_validation.py

# Run specific test categories
python -m pytest tests/ -v --tb=short

# Run performance benchmarks
python examples/performance_benchmarks.py
```

### Example Simulation
```bash
# Run a simple trajectory simulation
python examples/simple_trajectory.py

# Run Monte Carlo analysis
python examples/monte_carlo_example.py

# Generate comprehensive results (may take 1-4 hours)
python examples/comprehensive_results_generation.py
```

## Configuration

### Default Configuration
The framework uses default configuration from `config/default_config.yaml`. Key parameters:

```yaml
vehicle:
  mass: 5000.0  # kg
  reference_area: 15.0  # m^2
  drag_coefficient: 1.2
  lift_coefficient: 0.8

simulation:
  max_time: 3000.0  # seconds
  time_step: 1.0  # seconds
  integration_method: "RK45"

uncertainty:
  num_samples: 1000
  sampling_method: "latin_hypercube"
  random_seed: 42

optimization:
  algorithm: "SLSQP"
  max_iterations: 100
  tolerance: 1e-6
```

### Custom Configuration
Create your own configuration file:

```yaml
# my_config.yaml
vehicle:
  mass: 7500.0  # Heavier vehicle
  reference_area: 20.0
  
simulation:
  max_time: 2000.0
  time_step: 0.5  # Higher resolution
```

Use in code:
```python
from hypersonic_reentry.utils.config import load_config
config = load_config("my_config.yaml")
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'hypersonic_reentry'**
   - Ensure virtual environment is activated
   - Run `pip install -e .` in project directory

2. **CasADi Installation Failed**
   - Try: `conda install -c conda-forge casadi`
   - Or build from source with system IPOPT

3. **Out of Memory Errors**
   - Reduce Monte Carlo sample sizes
   - Use chunked processing: `chunk_size=100`
   - Enable memory monitoring: `enable_memory_tracking=True`

4. **Slow Performance**
   - Enable parallel processing: `parallel=True, num_workers=4`
   - Use optimized BLAS libraries (Intel MKL, OpenBLAS)
   - Consider reducing simulation time span

5. **Visualization Issues**
   - Update matplotlib: `pip install --upgrade matplotlib`
   - For Jupyter: `pip install ipywidgets`
   - For interactive plots: `pip install --upgrade plotly`

### Performance Optimization

1. **Parallel Processing**
   ```python
   # Enable parallel Monte Carlo
   mc_result = uq.run_monte_carlo_analysis(
       num_samples=1000,
       parallel=True,
       num_workers=8  # Use available CPU cores
   )
   ```

2. **Memory Management**
   ```python
   # Use chunked processing for large datasets
   results_generator = ResultsGenerator(
       chunk_size=100,
       memory_limit_gb=8.0
   )
   ```

3. **Caching**
   ```python
   # Enable result caching
   from hypersonic_reentry.utils.performance import CacheManager
   cache_manager = CacheManager(max_cache_size=1000)
   ```

### Getting Help

1. **Documentation**: Check `docs/` directory
2. **Examples**: See `examples/` directory
3. **Tests**: Review `tests/` for usage patterns
4. **Issues**: Create GitHub issue with:
   - Operating system and Python version
   - Complete error message
   - Minimal code example to reproduce issue

## Advanced Installation

### HPC/Cluster Installation

For high-performance computing environments:

```bash
# Module loading (example for SLURM systems)
module load python/3.9.0
module load gcc/9.3.0
module load openmpi/4.0.5
module load hdf5/1.12.0

# Install with MPI support
pip install mpi4py
export HDF5_MPI="ON"
pip install h5py --no-binary=h5py

# Install framework
pip install -e .
```

### Docker Installation

```dockerfile
# Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential gfortran \
    libopenblas-dev liblapack-dev \
    libhdf5-dev coinor-libipopt-dev

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["python", "examples/simple_trajectory.py"]
```

Build and run:
```bash
docker build -t hypersonic-reentry .
docker run -v $(pwd)/results:/app/results hypersonic-reentry
```

### Development Environment

For contributors:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install code quality tools
pip install black isort flake8 mypy
pip install pre-commit
pre-commit install

# Install testing tools
pip install pytest pytest-cov pytest-xdist
pip install hypothesis

# Install documentation tools
pip install sphinx sphinx-rtd-theme
pip install jupyter-book
```

## Next Steps

After successful installation:

1. **Run Examples**: Start with `examples/simple_trajectory.py`
2. **Read Documentation**: Review methodology in `website/methodology.md`
3. **Explore Results**: Check pre-generated results in `website/results.md`
4. **Customize**: Modify configuration files for your specific use case
5. **Extend**: Add custom vehicle models or optimization objectives

For detailed usage instructions, see the [User Guide](USER_GUIDE.md).