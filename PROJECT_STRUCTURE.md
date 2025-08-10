# Project Structure Documentation

## Hypersonic Reentry Trajectory Optimization Framework

This document provides a comprehensive overview of the project structure, explaining the purpose and organization of each directory and major file.

## 📁 Root Directory Structure

```
Stochastic_Simulation-Based_Trajectory_Optimization_for_Hypersonic_Reentry_Vehicles/
├── 📂 src/                          # Core framework source code
├── 📂 examples/                     # Usage examples and demonstrations
├── 📂 notebooks/                    # Jupyter notebooks for interactive analysis
├── 📂 tests/                        # Test suite and validation scripts
├── 📂 data/                         # Input data and simulation datasets
├── 📂 results/                      # Analysis results and outputs
├── 📂 config/                       # Configuration files
├── 📂 docs/                         # Documentation and guides
├── 📂 website/                      # Jekyll research website
├── 📂 scripts/                      # Utility and automation scripts
├── 🔧 setup.py                      # Package installation configuration
├── 🔧 requirements.txt              # Python dependencies
├── 📖 README.md                     # Main project documentation
├── 📖 INSTALL.md                    # Installation guide
├── 📖 USER_GUIDE.md                 # Comprehensive user documentation
├── 📖 PROJECT_SUMMARY.md            # Project overview and summary
├── 🧪 verify_installation.py        # Installation verification script
└── ⚙️  .gitignore                   # Version control ignore patterns
```

## 🔬 Core Framework (`src/`)

The source code is organized into logical modules representing different aspects of hypersonic reentry analysis:

```
src/hypersonic_reentry/
├── __init__.py                      # Package initialization and main exports
├── 📂 dynamics/                     # Vehicle dynamics and equations of motion
│   ├── __init__.py                  # Dynamics module exports
│   ├── vehicle_dynamics.py         # 3-DOF point mass dynamics
│   ├── aerodynamics.py              # Aerodynamic force calculations
│   └── coordinate_transforms.py     # Coordinate system transformations
├── 📂 atmosphere/                   # Atmospheric models and properties
│   ├── __init__.py                  # Atmosphere module exports
│   ├── us_standard_1976.py         # US Standard Atmosphere implementation
│   └── atmosphere_model.py         # Base atmospheric model interface
├── 📂 uncertainty/                  # Uncertainty quantification methods
│   ├── __init__.py                  # UQ module exports
│   ├── uncertainty_quantifier.py   # Main UQ orchestration class
│   ├── monte_carlo.py               # Monte Carlo simulation methods
│   ├── polynomial_chaos.py         # Polynomial Chaos Expansion
│   └── sensitivity_analysis.py     # Global sensitivity analysis
├── 📂 optimization/                 # Trajectory optimization algorithms
│   ├── __init__.py                  # Optimization module exports
│   ├── gradient_based.py            # SQP and gradient-based methods
│   └── trajectory_optimizer.py     # Base optimization interface
├── 📂 analysis/                     # Results analysis and statistics
│   ├── __init__.py                  # Analysis module exports
│   ├── results_generator.py        # Comprehensive results generation
│   └── statistical_analyzer.py     # Statistical analysis tools
├── 📂 visualization/                # Plotting and visualization tools
│   ├── __init__.py                  # Visualization module exports
│   ├── plot_manager.py              # Publication-quality plots
│   └── advanced_plots.py           # Interactive and advanced visualizations
├── 📂 utils/                        # Utilities and support functions
│   ├── __init__.py                  # Utils module exports
│   ├── constants.py                 # Physical and mathematical constants
│   └── performance.py               # Performance optimization utilities
└── 📂 control/                      # Control systems (under development)
    └── README.md                    # Control module development status
```

### Module Descriptions

#### `dynamics/` - Vehicle Dynamics
- **Purpose**: Implements the mathematical models for hypersonic vehicle motion
- **Key Components**:
  - 3-DOF point mass equations of motion in spherical coordinates
  - Aerodynamic force and moment calculations using hypersonic theory
  - Coordinate transformations between reference frames
- **Main Classes**: `VehicleDynamics`, `VehicleState`, `AerodynamicsModel`

#### `atmosphere/` - Atmospheric Modeling
- **Purpose**: Provides atmospheric properties and uncertainty modeling
- **Key Components**:
  - US Standard Atmosphere 1976 with 7-layer implementation
  - Atmospheric property calculations (density, temperature, pressure)
  - Uncertainty modeling for atmospheric variability
- **Main Classes**: `USStandard1976`, `AtmosphereModel`

#### `uncertainty/` - Uncertainty Quantification
- **Purpose**: Propagates uncertainties through simulation models
- **Key Components**:
  - Monte Carlo simulation with Latin Hypercube Sampling
  - Polynomial Chaos Expansion for efficient uncertainty propagation
  - Sobol sensitivity analysis for parameter importance ranking
- **Main Classes**: `UncertaintyQuantifier`, `UncertainParameter`, `SensitivityAnalyzer`

#### `optimization/` - Trajectory Optimization
- **Purpose**: Finds optimal trajectories subject to constraints
- **Key Components**:
  - Sequential Quadratic Programming (SQP) implementation
  - Constraint handling and objective function formulation
  - Multi-objective optimization capabilities
- **Main Classes**: `GradientBasedOptimizer`, `OptimizationObjective`, `OptimizationConstraint`

#### `analysis/` - Statistical Analysis
- **Purpose**: Analyzes simulation results and generates insights
- **Key Components**:
  - Comprehensive statistical analysis tools
  - Distribution fitting and goodness-of-fit testing
  - Confidence interval calculations and reliability analysis
- **Main Classes**: `StatisticalAnalyzer`, `ResultsGenerator`

#### `visualization/` - Plotting and Visualization
- **Purpose**: Creates publication-quality plots and interactive visualizations
- **Key Components**:
  - 2D/3D trajectory plotting with matplotlib
  - Interactive dashboards with Plotly
  - Uncertainty visualization and statistical plots
- **Main Classes**: `PlotManager`, `AdvancedPlotter`

## 📚 Examples and Demonstrations (`examples/`)

```
examples/
├── complete_analysis_example.py     # End-to-end analysis demonstration
├── comprehensive_results_generation.py  # Large-scale results generation
└── README.md                        # Examples documentation
```

**Purpose**: Provides working examples demonstrating framework capabilities
- **Target Users**: New users learning the framework, researchers adapting code
- **Content**: Complete workflows from basic simulation to comprehensive analysis

## 📓 Interactive Notebooks (`notebooks/`)

```
notebooks/
├── 01_quick_start.ipynb             # 5-minute framework introduction
├── 03_monte_carlo_analysis.ipynb    # Uncertainty quantification tutorial
├── 04_trajectory_optimization.ipynb # Optimization workflow (planned)
├── 05_sensitivity_analysis.ipynb    # Sensitivity analysis guide (planned)
└── README.md                        # Notebooks documentation
```

**Purpose**: Interactive learning and exploration of framework capabilities
- **Target Users**: Researchers, students, and practitioners
- **Content**: Step-by-step tutorials with explanations, code, and visualizations

## 🧪 Testing and Validation (`tests/`)

```
tests/
├── test_basic_functionality.py      # Basic framework functionality tests
├── test_comprehensive_validation.py # Extensive validation suite
└── README.md                        # Testing documentation
```

**Purpose**: Ensures framework correctness and reliability
- **Coverage**: Mathematical verification, physical behavior validation, performance testing
- **Types**: Unit tests, integration tests, regression tests, validation against analytical solutions

## 📊 Data Management (`data/`)

```
data/
├── 📂 atmospheric/                  # Atmospheric model data
├── 📂 vehicle/                      # Vehicle configuration files
├── 📂 trajectories/                 # Raw trajectory simulation data
├── 📂 monte_carlo/                  # Monte Carlo input parameters and samples
├── 📂 experimental/                 # Experimental or validation data
├── .gitkeep                         # Ensures directory tracking
└── README.md                        # Data directory documentation
```

**Purpose**: Organizes input data, configuration files, and raw simulation datasets
- **File Formats**: HDF5 for large datasets, CSV for tabular data, JSON for metadata
- **Version Control**: Large files use Git LFS, structure tracked with .gitkeep

## 📈 Results and Analysis (`results/`)

```
results/
├── 📂 data/                         # Processed analysis results
├── 📂 plots/                        # Generated visualizations
├── 📂 reports/                      # Analysis reports and summaries
├── 📂 statistical/                  # Statistical analysis outputs
├── 📂 optimization/                 # Trajectory optimization results
├── 📂 monte_carlo/                  # Monte Carlo analysis results
├── 📂 sensitivity/                  # Sensitivity analysis outputs
└── README.md                        # Results directory documentation
```

**Purpose**: Stores all analysis outputs in organized, accessible format
- **Organization**: By analysis type and date for easy navigation
- **Formats**: Multiple formats for different use cases (CSV, JSON, HDF5, PNG, PDF, HTML)

## ⚙️ Configuration (`config/`)

```
config/
├── default_config.yaml              # Default framework configuration
└── README.md                        # Configuration documentation
```

**Purpose**: Centralized configuration management
- **Content**: Vehicle parameters, simulation settings, optimization parameters
- **Format**: YAML for human readability and easy modification

## 📖 Documentation (`docs/`)

```
docs/
├── 📂 api/                          # Detailed API reference documentation
├── 📂 modules/                      # Module-specific documentation
├── 📂 reference/                    # Quick reference guides
├── 📂 tutorials/                    # Step-by-step tutorials
├── 📂 examples/                     # Detailed example walkthroughs
├── 📂 workflows/                    # Standard analysis workflows
├── 📂 theory/                       # Mathematical foundations
├── 📂 validation/                   # Model validation documentation
├── 📂 performance/                  # Performance benchmarks
├── 📂 development/                  # Contributing guidelines
├── 📂 architecture/                 # Framework design documentation
├── 📂 testing/                      # Testing procedures
└── README.md                        # Documentation overview
```

**Purpose**: Comprehensive documentation for all user types
- **Target Audiences**: End users, developers, researchers, contributors
- **Formats**: Markdown for source, HTML/PDF for distribution

## 🌐 Research Website (`website/`)

```
website/
├── _config.yml                      # Jekyll configuration
├── 📂 _data/                        # Structured data files
│   ├── navigation.yml               # Site navigation structure
│   └── results.yml                  # Research results data
├── 📂 _includes/                    # Reusable HTML components
│   ├── navigation.html              # Navigation component
│   └── plot_embed.html             # Plot embedding template
├── 📂 _layouts/                     # Page templates
│   └── default.html                # Main page layout
├── 📂 assets/                       # Static assets
│   ├── 📂 css/                      # Stylesheets
│   ├── 📂 images/                   # Images and figures
│   └── 📂 js/                       # JavaScript files
├── index.md                         # Homepage
├── methodology.md                   # Research methodology
└── results.md                       # Results and analysis
```

**Purpose**: Professional research presentation and dissemination
- **Features**: Responsive design, interactive visualizations, downloadable results
- **Technology**: Jekyll static site generator with GitHub Pages compatibility

## 🔧 Configuration and Setup Files

### `setup.py` - Package Installation
- **Purpose**: Defines package metadata and installation requirements
- **Features**: Development and production installation modes, optional dependencies

### `requirements.txt` - Dependencies
- **Purpose**: Specifies Python package dependencies with versions
- **Organization**: Core requirements with optional dependencies for different use cases

### `verify_installation.py` - Installation Verification
- **Purpose**: Comprehensive testing of installation and basic functionality
- **Features**: Automated testing of all major components, clear pass/fail reporting

## 📋 Documentation Files

### `README.md` - Main Project Documentation
- **Purpose**: Project overview, quick start guide, and navigation hub
- **Content**: Installation instructions, basic usage, links to detailed documentation

### `INSTALL.md` - Installation Guide
- **Purpose**: Detailed installation instructions for all platforms
- **Content**: System requirements, dependency management, troubleshooting

### `USER_GUIDE.md` - User Documentation
- **Purpose**: Comprehensive usage guide with examples
- **Content**: Framework concepts, detailed examples, best practices, troubleshooting

## 🔄 Version Control and Development

### `.gitignore` - Version Control
- **Purpose**: Specifies files and directories to exclude from version control
- **Features**: Framework-specific patterns, large file handling, development environment support

## 🚀 Usage Workflows

### 1. **New User Workflow**
```
README.md → INSTALL.md → verify_installation.py → notebooks/01_quick_start.ipynb
```

### 2. **Research Workflow**
```
examples/ → notebooks/ → src/ → results/ → website/
```

### 3. **Development Workflow**
```
src/ → tests/ → docs/ → examples/ → verification
```

### 4. **Analysis Workflow**
```
config/ → examples/comprehensive_results_generation.py → results/ → website/
```

## 🎯 Key Design Principles

1. **Modularity**: Clear separation of concerns with well-defined interfaces
2. **Extensibility**: Easy to add new models, methods, and analysis capabilities
3. **Reproducibility**: Version control, configuration management, and random seeds
4. **Performance**: Optimized for large-scale analysis with parallel processing
5. **Usability**: Comprehensive documentation, examples, and interactive notebooks
6. **Professional Quality**: Production-ready code with extensive testing and validation

This structure supports the complete lifecycle of hypersonic reentry research from initial exploration to publication and dissemination of results.