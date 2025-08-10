"""Coordinate transformation utilities for hypersonic reentry vehicles.

This module provides functions for transforming between different coordinate
systems commonly used in atmospheric flight mechanics.
"""

import numpy as np
from typing import Tuple, Dict
from ..utils.constants import EARTH_RADIUS, DEG_TO_RAD, RAD_TO_DEG


class CoordinateTransforms:
    """Coordinate transformation utilities for atmospheric flight mechanics.
    
    Provides transformations between:
    - Geodetic and geocentric coordinates
    - Spherical and Cartesian coordinates
    - Body and wind coordinate frames
    - Inertial and rotating reference frames
    """
    
    def __init__(self):
        """Initialize coordinate transformation utilities."""
        # Earth ellipsoid parameters (WGS84)
        self.earth_semimajor_axis = 6378137.0  # m
        self.earth_semiminor_axis = 6356752.314245  # m
        self.earth_eccentricity_squared = 0.00669437999014
        
    def geodetic_to_geocentric(self, 
                              latitude_deg: float, 
                              longitude_deg: float, 
                              altitude: float) -> Tuple[float, float, float]:
        """Convert geodetic coordinates to geocentric coordinates.
        
        Args:
            latitude_deg: Geodetic latitude in degrees
            longitude_deg: Geodetic longitude in degrees
            altitude: Altitude above ellipsoid in meters
            
        Returns:
            Tuple of (geocentric_latitude_deg, longitude_deg, radius_m)
        """
        lat_rad = latitude_deg * DEG_TO_RAD
        lon_rad = longitude_deg * DEG_TO_RAD
        
        # Calculate prime vertical radius of curvature
        N = self.earth_semimajor_axis / np.sqrt(
            1 - self.earth_eccentricity_squared * np.sin(lat_rad)**2
        )
        
        # Cartesian coordinates
        x = (N + altitude) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + altitude) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - self.earth_eccentricity_squared) + altitude) * np.sin(lat_rad)
        
        # Geocentric coordinates
        radius = np.sqrt(x**2 + y**2 + z**2)
        geocentric_lat_rad = np.arcsin(z / radius)
        
        return (geocentric_lat_rad * RAD_TO_DEG, longitude_deg, radius)
    
    def spherical_to_cartesian(self, 
                              radius: float, 
                              latitude_rad: float, 
                              longitude_rad: float) -> Tuple[float, float, float]:
        """Convert spherical coordinates to Cartesian coordinates.
        
        Args:
            radius: Radial distance in meters
            latitude_rad: Latitude in radians
            longitude_rad: Longitude in radians
            
        Returns:
            Tuple of (x, y, z) in Earth-centered inertial frame
        """
        x = radius * np.cos(latitude_rad) * np.cos(longitude_rad)
        y = radius * np.cos(latitude_rad) * np.sin(longitude_rad)
        z = radius * np.sin(latitude_rad)
        
        return (x, y, z)
    
    def cartesian_to_spherical(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert Cartesian coordinates to spherical coordinates.
        
        Args:
            x, y, z: Cartesian coordinates in meters
            
        Returns:
            Tuple of (radius_m, latitude_rad, longitude_rad)
        """
        radius = np.sqrt(x**2 + y**2 + z**2)
        latitude_rad = np.arcsin(z / radius)
        longitude_rad = np.arctan2(y, x)
        
        return (radius, latitude_rad, longitude_rad)
    
    def wind_to_body_transform(self, 
                              angle_of_attack: float, 
                              sideslip_angle: float = 0.0) -> np.ndarray:
        """Create transformation matrix from wind to body coordinates.
        
        Args:
            angle_of_attack: Angle of attack in radians
            sideslip_angle: Sideslip angle in radians (optional)
            
        Returns:
            3x3 transformation matrix
        """
        cos_alpha = np.cos(angle_of_attack)
        sin_alpha = np.sin(angle_of_attack)
        cos_beta = np.cos(sideslip_angle)
        sin_beta = np.sin(sideslip_angle)
        
        # Wind to body transformation matrix
        T_wb = np.array([
            [cos_alpha * cos_beta, -cos_alpha * sin_beta, -sin_alpha],
            [sin_beta, cos_beta, 0.0],
            [sin_alpha * cos_beta, -sin_alpha * sin_beta, cos_alpha]
        ])
        
        return T_wb
    
    def body_to_wind_transform(self, 
                              angle_of_attack: float, 
                              sideslip_angle: float = 0.0) -> np.ndarray:
        """Create transformation matrix from body to wind coordinates.
        
        Args:
            angle_of_attack: Angle of attack in radians
            sideslip_angle: Sideslip angle in radians (optional)
            
        Returns:
            3x3 transformation matrix
        """
        # Body to wind is transpose of wind to body
        T_wb = self.wind_to_body_transform(angle_of_attack, sideslip_angle)
        return T_wb.T
    
    def stability_to_body_transform(self, bank_angle: float) -> np.ndarray:
        """Create transformation matrix from stability to body coordinates.
        
        Args:
            bank_angle: Bank angle in radians
            
        Returns:
            3x3 transformation matrix
        """
        cos_sigma = np.cos(bank_angle)
        sin_sigma = np.sin(bank_angle)
        
        T_sb = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_sigma, sin_sigma],
            [0.0, -sin_sigma, cos_sigma]
        ])
        
        return T_sb
    
    def velocity_frame_transform(self, 
                                flight_path_angle: float, 
                                azimuth_angle: float) -> np.ndarray:
        """Create transformation matrix from local horizon to velocity frame.
        
        Args:
            flight_path_angle: Flight path angle in radians
            azimuth_angle: Azimuth angle in radians
            
        Returns:
            3x3 transformation matrix
        """
        cos_gamma = np.cos(flight_path_angle)
        sin_gamma = np.sin(flight_path_angle)
        cos_psi = np.cos(azimuth_angle)
        sin_psi = np.sin(azimuth_angle)
        
        T_vh = np.array([
            [cos_gamma * cos_psi, cos_gamma * sin_psi, -sin_gamma],
            [-sin_psi, cos_psi, 0.0],
            [sin_gamma * cos_psi, sin_gamma * sin_psi, cos_gamma]
        ])
        
        return T_vh
    
    def calculate_relative_velocity(self, 
                                  velocity_inertial: np.ndarray,
                                  position_ecef: np.ndarray) -> np.ndarray:
        """Calculate velocity relative to rotating Earth.
        
        Args:
            velocity_inertial: Velocity in inertial frame (m/s)
            position_ecef: Position in Earth-centered, Earth-fixed frame (m)
            
        Returns:
            Velocity relative to Earth surface (m/s)
        """
        from ..utils.constants import EARTH_ROTATION_RATE
        
        # Earth rotation vector
        omega_earth = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
        
        # Calculate velocity due to Earth rotation
        velocity_rotation = np.cross(omega_earth, position_ecef)
        
        # Relative velocity
        velocity_relative = velocity_inertial - velocity_rotation
        
        return velocity_relative
    
    def local_horizontal_frame(self, 
                             latitude: float, 
                             longitude: float) -> np.ndarray:
        """Create transformation matrix to local horizontal frame.
        
        Args:
            latitude: Latitude in radians
            longitude: Longitude in radians
            
        Returns:
            3x3 transformation matrix from ECEF to local horizontal
        """
        cos_lat = np.cos(latitude)
        sin_lat = np.sin(latitude)
        cos_lon = np.cos(longitude)
        sin_lon = np.sin(longitude)
        
        # Transformation from ECEF to local horizontal (NED)
        T_lh = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0.0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])
        
        return T_lh
    
    def calculate_ground_track(self, 
                             trajectory: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate ground track coordinates from trajectory data.
        
        Args:
            trajectory: Dictionary containing trajectory data
            
        Returns:
            Dictionary with ground track coordinates
        """
        latitudes = trajectory['latitude'] * RAD_TO_DEG
        longitudes = trajectory['longitude'] * RAD_TO_DEG
        
        # Normalize longitude to [-180, 180] degrees
        longitudes = np.mod(longitudes + 180.0, 360.0) - 180.0
        
        # Calculate downrange and crossrange distances
        lat0, lon0 = latitudes[0], longitudes[0]
        
        downrange = np.zeros_like(latitudes)
        crossrange = np.zeros_like(latitudes)
        
        for i in range(len(latitudes)):
            # Great circle distance calculation
            dlat = (latitudes[i] - lat0) * DEG_TO_RAD
            dlon = (longitudes[i] - lon0) * DEG_TO_RAD
            
            a = np.sin(dlat/2)**2 + np.cos(lat0 * DEG_TO_RAD) * np.cos(latitudes[i] * DEG_TO_RAD) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            distance = EARTH_RADIUS * c
            bearing = np.arctan2(
                np.sin(dlon) * np.cos(latitudes[i] * DEG_TO_RAD),
                np.cos(lat0 * DEG_TO_RAD) * np.sin(latitudes[i] * DEG_TO_RAD) - 
                np.sin(lat0 * DEG_TO_RAD) * np.cos(latitudes[i] * DEG_TO_RAD) * np.cos(dlon)
            )
            
            downrange[i] = distance * np.cos(bearing)
            crossrange[i] = distance * np.sin(bearing)
        
        return {
            'latitude': latitudes,
            'longitude': longitudes,
            'downrange': downrange,
            'crossrange': crossrange
        }