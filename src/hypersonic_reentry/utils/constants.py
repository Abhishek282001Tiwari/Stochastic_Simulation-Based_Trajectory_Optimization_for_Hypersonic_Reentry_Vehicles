"""Physical and mathematical constants for hypersonic reentry simulation."""

import numpy as np

# Earth parameters
EARTH_RADIUS = 6371000.0  # m, Earth's mean radius
GRAVITATIONAL_PARAMETER = 3.986004418e14  # m^3/s^2, Earth's gravitational parameter
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s, Earth's rotation rate
EARTH_J2 = 1.08262668e-3  # Earth's second zonal harmonic

# Atmospheric parameters
SEA_LEVEL_PRESSURE = 101325.0  # Pa, standard sea level pressure
SEA_LEVEL_TEMPERATURE = 288.15  # K, standard sea level temperature
SEA_LEVEL_DENSITY = 1.225  # kg/m^3, standard sea level density
GAS_CONSTANT = 287.0  # J/(kg*K), specific gas constant for air
SPECIFIC_HEAT_RATIO = 1.4  # dimensionless, specific heat ratio for air

# Mathematical constants
PI = np.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# Unit conversions
M_TO_FT = 3.28084  # meters to feet
KM_TO_M = 1000.0  # kilometers to meters
NM_TO_M = 1852.0  # nautical miles to meters

# Physical constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2*K^4), Stefan-Boltzmann constant
UNIVERSAL_GAS_CONSTANT = 8314.46261815324  # J/(kmol*K)

# Tolerance values for numerical computations
DEFAULT_ABSOLUTE_TOLERANCE = 1e-9
DEFAULT_RELATIVE_TOLERANCE = 1e-6
CONVERGENCE_TOLERANCE = 1e-6