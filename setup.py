"""Setup script for the hypersonic reentry trajectory optimization package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hypersonic-reentry-optimization",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Stochastic simulation-based trajectory optimization for hypersonic reentry vehicles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research/hypersonic-reentry-optimization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy", "sphinx"],
        "notebooks": ["jupyter", "ipywidgets"],
    },
)