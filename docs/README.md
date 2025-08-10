# Documentation Directory

This directory contains comprehensive documentation for the hypersonic reentry framework.

## Documentation Structure

### API Documentation
- `api/` - Detailed API reference for all modules and classes
- `modules/` - Module-specific documentation and examples
- `reference/` - Quick reference guides and cheat sheets

### User Guides
- `tutorials/` - Step-by-step tutorials for common tasks
- `examples/` - Detailed example walkthroughs
- `workflows/` - Standard analysis workflows and best practices

### Technical Documentation
- `theory/` - Mathematical foundations and algorithms
- `validation/` - Model validation and verification documentation
- `performance/` - Performance benchmarks and optimization guides

### Development Documentation
- `development/` - Contributing guidelines and development setup
- `architecture/` - Framework architecture and design decisions
- `testing/` - Testing procedures and validation protocols

## File Formats

- **Markdown (.md)**: Primary documentation format
- **Jupyter (.ipynb)**: Interactive documentation with code examples
- **PDF**: Compiled documentation for offline use
- **HTML**: Web-based documentation with search functionality

## Building Documentation

### Sphinx Documentation
```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs/
make html

# Build PDF documentation
make latexpdf
```

### Jupyter Book
```bash
# Install Jupyter Book
pip install jupyter-book

# Build interactive documentation
jupyter-book build docs/

# Create web version
jupyter-book publish docs/
```

## Documentation Standards

1. **Clear Structure**: Logical organization with consistent formatting
2. **Code Examples**: Working code snippets with expected outputs
3. **Mathematical Notation**: Proper LaTeX formatting for equations
4. **Cross-References**: Links between related topics and sections
5. **Version Control**: Track documentation changes with code changes

## Contribution Guidelines

When adding documentation:
- Follow existing formatting and style conventions
- Include working code examples
- Add cross-references to related sections
- Update table of contents and index files
- Test all code examples before committing

## Access Methods

- **Local**: Browse markdown files directly in repository
- **Web**: Build HTML version for browser viewing
- **PDF**: Generate compilation for offline reference
- **Interactive**: Use Jupyter notebooks for hands-on learning