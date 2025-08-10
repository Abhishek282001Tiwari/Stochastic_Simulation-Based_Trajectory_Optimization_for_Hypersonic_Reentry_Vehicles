# Contributing to the Hypersonic Reentry Framework

## Welcome Contributors!

We welcome contributions to the Hypersonic Reentry Trajectory Optimization Framework! This document provides guidelines for contributing code, documentation, bug reports, and feature requests.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Contribution Workflow](#code-contribution-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation Standards](#documentation-standards)
7. [Issue Reporting](#issue-reporting)
8. [Feature Requests](#feature-requests)
9. [Code Review Process](#code-review-process)
10. [Release Process](#release-process)

---

## Getting Started

### Prerequisites

- **Python 3.8+** with development tools
- **Git** for version control
- **GitHub account** for pull requests
- **Basic knowledge** of hypersonic flight mechanics (helpful but not required)

### Areas for Contribution

We welcome contributions in the following areas:

- **Core Framework**: Vehicle dynamics, atmospheric models, optimization algorithms
- **Uncertainty Quantification**: New sampling methods, sensitivity analysis techniques
- **Visualization**: Interactive plots, dashboard improvements, publication-quality figures
- **Performance**: Parallel processing, memory optimization, computational efficiency
- **Testing**: Unit tests, integration tests, validation cases
- **Documentation**: API documentation, tutorials, examples, theory guides
- **Examples**: Real-world applications, benchmark problems, case studies

---

## Development Environment Setup

### 1. Fork and Clone Repository

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/hypersonic-reentry-framework.git
cd hypersonic-reentry-framework

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL-OWNER/hypersonic-reentry-framework.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv hypersonic_dev_env
source hypersonic_dev_env/bin/activate  # On Windows: hypersonic_dev_env\\Scripts\\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install framework in development mode
pip install -e .
```

### 3. Pre-commit Hooks Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually (optional)
pre-commit run --all-files
```

### 4. Verify Installation

```bash
# Run verification script
python verify_installation.py

# Run test suite
python -m pytest tests/ -v

# Check code style
flake8 src/
black --check src/
```

---

## Code Contribution Workflow

### 1. Create Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow coding standards (see below)
- Add tests for new functionality
- Update documentation as needed
- Commit changes with clear messages

### 3. Test Changes

```bash
# Run full test suite
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_dynamics/ -v

# Check test coverage
pytest --cov=src/hypersonic_reentry tests/

# Run performance benchmarks (if applicable)
python benchmarks/run_benchmarks.py
```

### 4. Submit Pull Request

```bash
# Push changes to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Include clear description and link to relevant issues
```

---

## Coding Standards

### 1. Python Code Style

We follow **PEP 8** with some modifications:

```python
# Line length: 88 characters (Black formatter default)
# Use Black for automatic formatting
black src/

# Use isort for import sorting
isort src/

# Use flake8 for linting
flake8 src/
```

### 2. Code Structure

```python
#!/usr/bin/env python3
\"\"\"
Module docstring describing purpose and functionality.

This module implements [specific functionality] for the hypersonic
reentry framework.
\"\"\"

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
from matplotlib import pyplot as plt

from hypersonic_reentry.utils.constants import EARTH_RADIUS


class ExampleClass:
    \"\"\"
    Brief description of class purpose.
    
    Longer description with details about the class functionality,
    key algorithms, and usage patterns.
    
    Attributes:
        public_attribute (float): Description of public attribute
        
    Example:
        >>> example = ExampleClass(parameter=1.0)
        >>> result = example.method()
        >>> print(result)
        42.0
    \"\"\"
    
    def __init__(self, parameter: float, optional_param: Optional[str] = None):
        \"\"\"
        Initialize ExampleClass instance.
        
        Args:
            parameter: Required floating point parameter
            optional_param: Optional string parameter
            
        Raises:
            ValueError: If parameter is negative
            TypeError: If parameter is not numeric
        \"\"\"
        if parameter < 0:
            raise ValueError(\"Parameter must be non-negative\")
            
        self.parameter = float(parameter)
        self.optional_param = optional_param
        self._private_attribute = None  # Private attributes use underscore
    
    def public_method(self, input_value: float) -> float:
        \"\"\"
        Public method with clear documentation.
        
        Args:
            input_value: Input to be processed
            
        Returns:
            Processed result
            
        Raises:
            ValueError: If input_value is invalid
        \"\"\"
        if not isinstance(input_value, (int, float)):
            raise TypeError(\"input_value must be numeric\")
            
        result = self._private_method(input_value)
        return result
    
    def _private_method(self, value: float) -> float:
        \"\"\"Private method for internal computations.\"\"\"
        return value * self.parameter

    def __repr__(self) -> str:
        \"\"\"String representation for debugging.\"\"\"
        return f\"ExampleClass(parameter={self.parameter})\"\n    \n    def __str__(self) -> str:\n        \"\"\"Human-readable string representation.\"\"\"\n        return f\"Example with parameter {self.parameter}\"\n\n\ndef public_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:\n    \"\"\"\n    Public function with NumPy arrays.\n    \n    Args:\n        x: Input array\n        y: Second input array\n        \n    Returns:\n        Computed result array\n        \n    Example:\n        >>> x = np.array([1, 2, 3])\n        >>> y = np.array([4, 5, 6])\n        >>> result = public_function(x, y)\n        >>> print(result)\n        [5 7 9]\n    \"\"\"\n    # Validate inputs\n    x = np.asarray(x)\n    y = np.asarray(y)\n    \n    if x.shape != y.shape:\n        raise ValueError(\"Arrays must have same shape\")\n    \n    # Perform computation\n    return x + y\n```\n\n### 3. Type Hints\n\nUse type hints for all public functions and methods:\n\n```python\nfrom typing import Dict, List, Optional, Tuple, Union\nimport numpy as np\n\n# Function type hints\ndef compute_trajectory(\n    initial_state: VehicleState,\n    time_span: Tuple[float, float],\n    parameters: Dict[str, float],\n    options: Optional[Dict[str, any]] = None\n) -> Dict[str, np.ndarray]:\n    \"\"\"Function with comprehensive type hints.\"\"\"\n    pass\n\n# Class attribute type hints\nclass VehicleDynamics:\n    mass: float\n    reference_area: float\n    drag_coefficient: float\n    \n    def __init__(self, mass: float, reference_area: float):\n        self.mass = mass\n        self.reference_area = reference_area\n```\n\n### 4. Error Handling\n\n```python\n# Use specific exception types\nclass ValidationError(ValueError):\n    \"\"\"Raised when parameter validation fails.\"\"\"\n    pass\n\nclass ConvergenceError(RuntimeError):\n    \"\"\"Raised when iterative algorithm fails to converge.\"\"\"\n    pass\n\n# Comprehensive error handling\ndef validate_parameters(parameters: Dict[str, float]) -> None:\n    \"\"\"Validate input parameters.\"\"\"\n    required_params = ['mass', 'reference_area']\n    \n    for param in required_params:\n        if param not in parameters:\n            raise ValidationError(f\"Missing required parameter: {param}\")\n        \n        if not isinstance(parameters[param], (int, float)):\n            raise ValidationError(f\"Parameter {param} must be numeric\")\n        \n        if parameters[param] <= 0:\n            raise ValidationError(f\"Parameter {param} must be positive\")\n\ndef safe_computation(x: float) -> float:\n    \"\"\"Example of robust error handling.\"\"\"\n    try:\n        if x <= 0:\n            raise ValueError(\"Input must be positive\")\n        \n        result = np.sqrt(x)\n        \n        if not np.isfinite(result):\n            raise ValueError(\"Computation resulted in non-finite value\")\n            \n        return result\n        \n    except ValueError as e:\n        raise ValueError(f\"safe_computation failed: {e}\")\n    except Exception as e:\n        raise RuntimeError(f\"Unexpected error in safe_computation: {e}\")\n```\n\n---\n\n## Testing Guidelines\n\n### 1. Test Structure\n\n```python\n# tests/test_module_name.py\nimport pytest\nimport numpy as np\nfrom unittest.mock import Mock, patch\n\nfrom hypersonic_reentry.module_name import ClassToTest, function_to_test\nfrom hypersonic_reentry.utils.constants import EARTH_RADIUS\n\n\nclass TestClassToTest:\n    \"\"\"Test suite for ClassToTest.\"\"\"\n    \n    def setup_method(self):\n        \"\"\"Setup method run before each test.\"\"\"\n        self.test_instance = ClassToTest(parameter=1.0)\n        \n    def teardown_method(self):\n        \"\"\"Cleanup method run after each test.\"\"\"\n        # Clean up resources if needed\n        pass\n    \n    def test_init_valid_parameters(self):\n        \"\"\"Test initialization with valid parameters.\"\"\"\n        instance = ClassToTest(parameter=2.5)\n        assert instance.parameter == 2.5\n        \n    def test_init_invalid_parameters(self):\n        \"\"\"Test initialization with invalid parameters.\"\"\"\n        with pytest.raises(ValueError, match=\"Parameter must be non-negative\"):\n            ClassToTest(parameter=-1.0)\n            \n    def test_method_basic_functionality(self):\n        \"\"\"Test basic method functionality.\"\"\"\n        result = self.test_instance.public_method(5.0)\n        expected = 5.0  # Based on known logic\n        assert result == expected\n        \n    def test_method_edge_cases(self):\n        \"\"\"Test method with edge cases.\"\"\"\n        # Test with zero\n        result = self.test_instance.public_method(0.0)\n        assert result == 0.0\n        \n        # Test with very large number\n        result = self.test_instance.public_method(1e10)\n        assert np.isfinite(result)\n        \n    def test_method_error_conditions(self):\n        \"\"\"Test method error handling.\"\"\"\n        with pytest.raises(TypeError):\n            self.test_instance.public_method(\"invalid\")\n            \n    @pytest.mark.parametrize(\"input_val,expected\", [\n        (1.0, 1.0),\n        (2.0, 2.0),\n        (10.0, 10.0),\n    ])\n    def test_method_parametrized(self, input_val, expected):\n        \"\"\"Parametrized test for multiple input values.\"\"\"\n        result = self.test_instance.public_method(input_val)\n        assert result == expected\n\n\nclass TestFunctionToTest:\n    \"\"\"Test suite for standalone functions.\"\"\"\n    \n    def test_function_basic_case(self):\n        \"\"\"Test function with basic inputs.\"\"\"\n        x = np.array([1, 2, 3])\n        y = np.array([4, 5, 6])\n        result = function_to_test(x, y)\n        expected = np.array([5, 7, 9])\n        np.testing.assert_array_equal(result, expected)\n        \n    def test_function_edge_cases(self):\n        \"\"\"Test function edge cases.\"\"\"\n        # Empty arrays\n        x = np.array([])\n        y = np.array([])\n        result = function_to_test(x, y)\n        assert len(result) == 0\n        \n    def test_function_error_conditions(self):\n        \"\"\"Test function error handling.\"\"\"\n        x = np.array([1, 2])\n        y = np.array([1, 2, 3])  # Different shape\n        \n        with pytest.raises(ValueError, match=\"Arrays must have same shape\"):\n            function_to_test(x, y)\n\n\n@pytest.mark.slow\nclass TestIntegrationScenarios:\n    \"\"\"Integration tests that may take longer to run.\"\"\"\n    \n    def test_complete_workflow(self):\n        \"\"\"Test complete analysis workflow.\"\"\"\n        # This would test end-to-end functionality\n        pass\n        \n    @pytest.mark.skip(reason=\"Requires external data file\")\n    def test_with_external_data(self):\n        \"\"\"Test that requires external data (skipped by default).\"\"\"\n        pass\n\n\n@pytest.fixture\ndef sample_trajectory_data():\n    \"\"\"Fixture providing sample trajectory data for tests.\"\"\"\n    times = np.linspace(0, 1000, 100)\n    altitudes = 120000 - times * 100  # Simple linear descent\n    velocities = 7500 - times * 5     # Simple deceleration\n    \n    return {\n        'times': times,\n        'altitudes': altitudes,\n        'velocities': velocities\n    }\n\n\ndef test_with_fixture(sample_trajectory_data):\n    \"\"\"Test using fixture data.\"\"\"\n    data = sample_trajectory_data\n    assert len(data['times']) == 100\n    assert data['altitudes'][0] == 120000\n```\n\n### 2. Test Categories\n\n```bash\n# Run all tests\npytest tests/ -v\n\n# Run only fast tests (exclude @pytest.mark.slow)\npytest tests/ -v -m \"not slow\"\n\n# Run specific test category\npytest tests/test_dynamics/ -v\n\n# Run with coverage report\npytest --cov=src/hypersonic_reentry tests/\n\n# Run specific test\npytest tests/test_dynamics.py::TestVehicleDynamics::test_compute_derivatives -v\n```\n\n### 3. Numerical Testing\n\n```python\ndef test_numerical_accuracy():\n    \"\"\"Test numerical accuracy with appropriate tolerances.\"\"\"\n    computed = compute_some_value()\n    expected = 1.23456789\n    \n    # Use appropriate tolerance for floating point comparison\n    np.testing.assert_allclose(computed, expected, rtol=1e-10, atol=1e-12)\n\ndef test_array_comparison():\n    \"\"\"Test NumPy array comparisons.\"\"\"\n    result = compute_array_result()\n    expected = np.array([1.0, 2.0, 3.0])\n    \n    # For arrays\n    np.testing.assert_array_almost_equal(result, expected, decimal=10)\n    \n    # Check shapes match\n    assert result.shape == expected.shape\n    \n    # Check for NaN/Inf values\n    assert np.all(np.isfinite(result))\n```\n\n---\n\n## Documentation Standards\n\n### 1. Docstring Format\n\nUse **Google-style docstrings**:\n\n```python\ndef complex_function(\n    x: np.ndarray,\n    y: np.ndarray, \n    method: str = 'default',\n    tolerance: float = 1e-6\n) -> Tuple[np.ndarray, Dict[str, float]]:\n    \"\"\"\n    Perform complex computation on input arrays.\n    \n    This function implements a sophisticated algorithm for processing\n    input data with various methods and numerical tolerances.\n    \n    Args:\n        x: Input array of shape (N,) containing x-coordinates\n        y: Input array of shape (N,) containing y-coordinates\n        method: Computation method, one of {'default', 'fast', 'accurate'}\n        tolerance: Numerical tolerance for convergence (default: 1e-6)\n        \n    Returns:\n        A tuple containing:\n            - result_array: Computed results of shape (N,)\n            - info_dict: Dictionary with computation metadata including:\n                - 'iterations': Number of iterations performed\n                - 'residual': Final residual norm\n                - 'convergence': Boolean convergence flag\n                \n    Raises:\n        ValueError: If x and y have different shapes\n        ValueError: If method is not recognized\n        ConvergenceError: If algorithm fails to converge within tolerance\n        \n    Example:\n        >>> x = np.linspace(0, 1, 10)\n        >>> y = np.sin(x)\n        >>> result, info = complex_function(x, y, method='accurate')\n        >>> print(f\"Converged in {info['iterations']} iterations\")\n        Converged in 15 iterations\n        \n    Note:\n        The 'fast' method sacrifices some accuracy for computational speed.\n        For critical applications, use method='accurate'.\n        \n    References:\n        Smith, J. et al. \"Advanced Algorithms for Hypersonic Analysis.\" \n        Journal of Computational Physics, 2023.\n    \"\"\"\n    pass\n```\n\n### 2. Module Documentation\n\nEach module should have comprehensive documentation:\n\n```python\n\"\"\"\nHypersonic Vehicle Dynamics Module\n==================================\n\nThis module implements the core vehicle dynamics for hypersonic reentry\ntrajectory simulation. It provides classes and functions for:\n\n- 3-DOF point mass vehicle dynamics\n- Aerodynamic force calculations\n- Coordinate system transformations\n- Integration with atmospheric models\n\nThe primary class is `VehicleDynamics`, which encapsulates the vehicle\nproperties and provides methods for computing time derivatives of the\nstate vector.\n\nExample Usage:\n    >>> from hypersonic_reentry.dynamics import VehicleDynamics, VehicleState\n    >>> vehicle = VehicleDynamics(mass=5000, reference_area=15.0)\n    >>> initial_state = VehicleState(altitude=120000, velocity=7500)\n    >>> derivatives = vehicle.compute_derivatives(initial_state, time=0)\n\nClasses:\n    VehicleDynamics: Main vehicle dynamics class\n    VehicleState: State vector container\n    AerodynamicsModel: Aerodynamic coefficient models\n    \nFunctions:\n    coordinate_transform: Transform between coordinate systems\n    compute_aerodynamic_coefficients: Calculate aerodynamic properties\n    \nConstants:\n    EARTH_RADIUS: Earth radius in meters\n    GRAVITATIONAL_PARAMETER: Earth's gravitational parameter\n    \nSee Also:\n    hypersonic_reentry.atmosphere: Atmospheric modeling\n    hypersonic_reentry.optimization: Trajectory optimization\n    \nReferences:\n    [1] Vinh, N.X. et al. \"Hypersonic and Planetary Entry Flight Mechanics\"\n    [2] Anderson, J.D. \"Hypersonic and High-Temperature Gas Dynamics\"\n\"\"\"\n```\n\n### 3. README Updates\n\nWhen adding new features, update relevant README files:\n\n```markdown\n# New Feature: Advanced Optimization\n\n## Overview\nBrief description of the new feature and its purpose.\n\n## Installation\nAny additional installation requirements.\n\n## Usage\nBasic usage examples:\n\n```python\nfrom hypersonic_reentry.optimization import NewOptimizer\noptimizer = NewOptimizer()\nresult = optimizer.optimize()\n```\n\n## Examples\nLink to detailed examples in the examples/ directory.\n\n## API Reference\nLink to detailed API documentation.\n```\n\n---\n\n## Issue Reporting\n\n### Bug Reports\n\nWhen reporting bugs, please include:\n\n1. **Environment Information**\n   ```python\n   import sys\n   import numpy\n   import scipy\n   import hypersonic_reentry\n   \n   print(f\"Python: {sys.version}\")\n   print(f\"NumPy: {numpy.__version__}\")\n   print(f\"SciPy: {scipy.__version__}\")\n   print(f\"Framework: {hypersonic_reentry.__version__}\")\n   ```\n\n2. **Minimal Reproducible Example**\n   ```python\n   # Minimal code that reproduces the issue\n   from hypersonic_reentry import some_module\n   \n   # Clear steps to reproduce\n   result = some_module.function_with_bug(input_data)\n   print(result)  # Shows unexpected output\n   ```\n\n3. **Expected vs. Actual Behavior**\n   - What you expected to happen\n   - What actually happened\n   - Error messages (full stack trace)\n\n4. **Additional Context**\n   - Operating system\n   - Installation method\n   - Any modifications to default settings\n\n### Issue Template\n\n```markdown\n**Bug Description**\nA clear and concise description of the bug.\n\n**To Reproduce**\nSteps to reproduce the behavior:\n1. Import module '...'\n2. Call function with '...'\n3. See error\n\n**Expected Behavior**\nWhat you expected to happen.\n\n**Environment**\n- OS: [e.g. Windows 10, Ubuntu 20.04]\n- Python version: [e.g. 3.9.5]\n- Framework version: [e.g. 1.2.3]\n- Dependencies: [output of pip freeze]\n\n**Additional Context**\nAny other context about the problem.\n```\n\n---\n\n## Feature Requests\n\n### Feature Request Template\n\n```markdown\n**Feature Description**\nClear description of the desired feature.\n\n**Motivation**\nWhy is this feature needed? What problem does it solve?\n\n**Proposed Implementation**\nIf you have ideas about how to implement this feature.\n\n**Examples**\nCode examples showing how the feature would be used.\n\n**Additional Context**\nAny other context or screenshots about the feature request.\n```\n\n### Research and Analysis\n\nBefore implementing new features:\n\n1. **Literature Review**: Research existing methods and implementations\n2. **API Design**: Design clean, consistent interfaces\n3. **Testing Strategy**: Plan comprehensive test coverage\n4. **Documentation**: Prepare documentation and examples\n5. **Performance**: Consider computational efficiency\n\n---\n\n## Code Review Process\n\n### Review Checklist\n\n**Functionality**\n- [ ] Code solves the intended problem\n- [ ] Edge cases are handled appropriately\n- [ ] Error conditions are managed properly\n- [ ] Performance is acceptable\n\n**Code Quality**\n- [ ] Code follows project style guidelines\n- [ ] Functions and classes are well-documented\n- [ ] Variable names are descriptive\n- [ ] Code is readable and maintainable\n\n**Testing**\n- [ ] New functionality has comprehensive tests\n- [ ] Tests cover edge cases and error conditions\n- [ ] All tests pass\n- [ ] Test coverage is maintained or improved\n\n**Documentation**\n- [ ] API documentation is updated\n- [ ] Examples demonstrate new functionality\n- [ ] README files are updated if needed\n- [ ] Changelog is updated\n\n**Integration**\n- [ ] Code integrates well with existing framework\n- [ ] Breaking changes are documented\n- [ ] Backward compatibility is maintained when possible\n- [ ] Dependencies are justified and minimal\n\n### Review Guidelines\n\n**For Authors:**\n- Keep pull requests focused and reasonably sized\n- Write clear commit messages and PR descriptions\n- Respond promptly to review feedback\n- Update documentation and tests\n\n**For Reviewers:**\n- Be constructive and respectful in feedback\n- Focus on code quality and maintainability\n- Ask questions when code is unclear\n- Suggest improvements and alternatives\n- Approve when ready, request changes when needed\n\n---\n\n## Release Process\n\n### Version Numbers\n\nWe use **Semantic Versioning** (semver.org):\n\n- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)\n- **MAJOR**: Breaking changes\n- **MINOR**: New features (backward compatible)\n- **PATCH**: Bug fixes (backward compatible)\n\n### Release Checklist\n\n**Pre-release:**\n- [ ] All tests pass\n- [ ] Documentation is updated\n- [ ] Changelog is updated\n- [ ] Version number is bumped\n- [ ] Examples are tested\n- [ ] Performance benchmarks are run\n\n**Release:**\n- [ ] Create release tag\n- [ ] Build and test distribution packages\n- [ ] Upload to PyPI\n- [ ] Create GitHub release\n- [ ] Update documentation website\n\n**Post-release:**\n- [ ] Monitor for issues\n- [ ] Update installation guides\n- [ ] Announce release\n\n---\n\n## Communication Guidelines\n\n### Channels\n\n- **GitHub Issues**: Bug reports, feature requests, technical discussions\n- **GitHub Discussions**: General questions, ideas, community support\n- **Email**: Private matters, security issues\n\n### Code of Conduct\n\nWe are committed to providing a welcoming and inclusive environment:\n\n- Be respectful and professional\n- Focus on constructive feedback\n- Welcome newcomers and help them learn\n- Respect different viewpoints and experiences\n- Report inappropriate behavior\n\n### Getting Help\n\n**For Contributors:**\n- Read this contributing guide thoroughly\n- Check existing issues and discussions\n- Start with small contributions\n- Ask questions when stuck\n\n**For Users:**\n- Check documentation and examples first\n- Search existing issues for similar problems\n- Provide complete information when asking for help\n- Be patient while maintainers respond\n\n---\n\n## Recognition\n\nWe appreciate all contributions to the project!\n\n**Contributor Recognition:**\n- Contributors are listed in CONTRIBUTORS.md\n- Significant contributions are highlighted in release notes\n- Active contributors may be invited to become maintainers\n\n**Types of Contributions:**\n- Code contributions (features, bug fixes, optimizations)\n- Documentation improvements\n- Bug reports and testing\n- Community support and mentoring\n- Outreach and education\n\n---\n\n## Resources\n\n### Development Tools\n- **Black**: Code formatting\n- **isort**: Import sorting\n- **flake8**: Linting\n- **mypy**: Type checking\n- **pytest**: Testing framework\n- **pre-commit**: Git hooks\n\n### Documentation\n- **Sphinx**: Documentation generation\n- **MkDocs**: Alternative documentation tool\n- **Jupyter**: Interactive examples\n\n### References\n- [PEP 8](https://www.python.org/dev/peps/pep-0008/): Python style guide\n- [NumPy Documentation Guidelines](https://numpydoc.readthedocs.io/)\n- [Semantic Versioning](https://semver.org/)\n- [Keep a Changelog](https://keepachangelog.com/)\n\n---\n\nThank you for contributing to the Hypersonic Reentry Framework! Your contributions help advance the state of hypersonic vehicle analysis and benefit the entire aerospace research community.\n\n**Questions?** Feel free to open a discussion or contact the maintainers directly.