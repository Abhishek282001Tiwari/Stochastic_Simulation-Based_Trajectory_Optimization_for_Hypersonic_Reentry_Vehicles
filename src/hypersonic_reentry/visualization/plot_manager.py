"""Comprehensive plotting manager for hypersonic reentry visualization.

This module provides high-level plotting functionality with publication-quality
output, consistent styling, and support for both static and interactive plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

from ..utils.constants import EARTH_RADIUS, DEG_TO_RAD, RAD_TO_DEG


class PlotManager:
    """Comprehensive plotting manager for trajectory visualization.
    
    Provides methods for creating publication-quality plots including:
    - 2D and 3D trajectory plots
    - Uncertainty visualization with confidence bands
    - Performance metric comparisons
    - Interactive web-based visualizations
    """
    
    def __init__(self, 
                 output_directory: str = "plots",
                 style_theme: str = "publication",
                 color_palette: str = "viridis",
                 dpi: int = 300):
        """Initialize plot manager.
        
        Args:
            output_directory: Directory for saving plots
            style_theme: Visual style theme
            color_palette: Color palette for plots
            dpi: Resolution for saved plots
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Style settings
        self.style_theme = style_theme
        self.color_palette = color_palette
        self.dpi = dpi
        
        # Set up matplotlib style
        self._setup_matplotlib_style()
        
        # Color schemes
        self.colors = self._get_color_scheme()
        
        self.logger.info(f"Initialized plot manager with {style_theme} theme")
    
    def _setup_matplotlib_style(self) -> None:
        """Set up matplotlib style for publication-quality plots."""
        if self.style_theme == "publication":
            plt.style.use(['default'])
            
            # Font settings
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'font.serif': ['Times', 'Computer Modern Roman'],
                'text.usetex': False,  # Set to True if LaTeX is available
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 11,
                'figure.titlesize': 18,
                'lines.linewidth': 2,
                'axes.linewidth': 1.2,
                'grid.linewidth': 0.8,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'figure.figsize': (10, 6),
                'figure.dpi': self.dpi,
                'savefig.dpi': self.dpi,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.1
            })
        
        # Set seaborn style for enhanced appearance
        sns.set_palette(self.color_palette)
    
    def _get_color_scheme(self) -> Dict[str, str]:
        """Get color scheme for consistent plotting."""
        if self.color_palette == "viridis":
            return {
                'primary': '#440154',
                'secondary': '#31688e', 
                'tertiary': '#35b779',
                'quaternary': '#fde725',
                'error': '#ff6b6b',
                'warning': '#ffa726',
                'success': '#66bb6a',
                'info': '#42a5f5'
            }
        elif self.color_palette == "plasma":
            return {
                'primary': '#0d0887',
                'secondary': '#7201a8',
                'tertiary': '#bd3786',
                'quaternary': '#f89441',
                'error': '#ff6b6b',
                'warning': '#ffa726', 
                'success': '#66bb6a',
                'info': '#42a5f5'
            }
        else:
            # Default color scheme
            return {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'tertiary': '#2ca02c',
                'quaternary': '#d62728',
                'error': '#ff6b6b',
                'warning': '#ffa726',
                'success': '#66bb6a', 
                'info': '#42a5f5'
            }
    
    def plot_trajectory_2d(self, 
                          trajectory: Dict[str, np.ndarray],
                          ground_track: Optional[Dict[str, np.ndarray]] = None,
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
        """Create 2D trajectory plots showing altitude and velocity profiles.
        
        Args:
            trajectory: Dictionary containing trajectory data
            ground_track: Optional ground track data
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hypersonic Reentry Trajectory Analysis', fontsize=16, fontweight='bold')
        
        time_hours = trajectory['time'] / 3600  # Convert to hours for readability
        
        # Altitude vs Time
        axes[0, 0].plot(time_hours, trajectory['altitude'] / 1000, 
                       color=self.colors['primary'], linewidth=2)
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Altitude (km)')
        axes[0, 0].set_title('Altitude Profile')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Velocity vs Time
        axes[0, 1].plot(time_hours, trajectory['velocity'] / 1000,
                       color=self.colors['secondary'], linewidth=2)
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Velocity (km/s)')
        axes[0, 1].set_title('Velocity Profile')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Altitude vs Velocity (trajectory shape)
        axes[1, 0].plot(trajectory['velocity'] / 1000, trajectory['altitude'] / 1000,
                       color=self.colors['tertiary'], linewidth=2)
        axes[1, 0].set_xlabel('Velocity (km/s)')
        axes[1, 0].set_ylabel('Altitude (km)')
        axes[1, 0].set_title('Trajectory Shape')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Flight path angle vs Time
        axes[1, 1].plot(time_hours, trajectory['flight_path_angle'] * RAD_TO_DEG,
                       color=self.colors['quaternary'], linewidth=2)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Flight Path Angle (deg)')
        axes[1, 1].set_title('Flight Path Angle')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            self._save_plot(fig, save_path)
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_trajectory_3d(self, 
                          trajectory: Dict[str, np.ndarray],
                          earth_sphere: bool = True,
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
        """Create 3D trajectory visualization.
        
        Args:
            trajectory: Dictionary containing trajectory data
            earth_sphere: Whether to show Earth sphere
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert spherical coordinates to Cartesian
        lat = trajectory['latitude']
        lon = trajectory['longitude'] 
        alt = trajectory['altitude']
        r = EARTH_RADIUS + alt
        
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        
        # Plot trajectory
        ax.plot(x / 1000, y / 1000, z / 1000, 
               color=self.colors['primary'], linewidth=3, label='Trajectory')
        
        # Mark start and end points
        ax.scatter(x[0] / 1000, y[0] / 1000, z[0] / 1000, 
                  color=self.colors['success'], s=100, label='Entry Point')
        ax.scatter(x[-1] / 1000, y[-1] / 1000, z[-1] / 1000,
                  color=self.colors['error'], s=100, label='Exit Point')
        
        # Draw Earth sphere if requested
        if earth_sphere:
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            earth_x = EARTH_RADIUS/1000 * np.outer(np.cos(u), np.sin(v))
            earth_y = EARTH_RADIUS/1000 * np.outer(np.sin(u), np.sin(v))
            earth_z = EARTH_RADIUS/1000 * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='lightblue')
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('3D Trajectory Visualization')
        ax.legend()
        
        # Equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / (2.0*1000)
        mid_x = (x.max()+x.min()) * 0.5 / 1000
        mid_y = (y.max()+y.min()) * 0.5 / 1000
        mid_z = (z.max()+z.min()) * 0.5 / 1000
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save and show
        if save_path:
            self._save_plot(fig, save_path)
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_uncertainty_bands(self, 
                              trajectories: List[Dict[str, np.ndarray]],
                              confidence_levels: List[float] = [68, 95, 99],
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> plt.Figure:
        """Create uncertainty visualization with confidence bands.
        
        Args:
            trajectories: List of trajectory dictionaries from Monte Carlo
            confidence_levels: Confidence levels for uncertainty bands
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Trajectory Uncertainty Analysis', fontsize=16, fontweight='bold')
        
        # Extract data from all trajectories
        all_times = [traj['time'] for traj in trajectories]
        all_altitudes = [traj['altitude'] for traj in trajectories]
        all_velocities = [traj['velocity'] for traj in trajectories]
        all_mach = [traj.get('mach_number', np.zeros_like(traj['time'])) for traj in trajectories]
        
        # Find common time grid
        max_time = min([t.max() for t in all_times])
        time_grid = np.linspace(0, max_time, 1000)
        
        # Interpolate all trajectories to common grid
        altitude_samples = np.zeros((len(trajectories), len(time_grid)))
        velocity_samples = np.zeros((len(trajectories), len(time_grid)))
        mach_samples = np.zeros((len(trajectories), len(time_grid)))
        
        for i, traj in enumerate(trajectories):
            altitude_samples[i, :] = np.interp(time_grid, traj['time'], traj['altitude'])
            velocity_samples[i, :] = np.interp(time_grid, traj['time'], traj['velocity'])
            if 'mach_number' in traj:
                mach_samples[i, :] = np.interp(time_grid, traj['time'], traj['mach_number'])
        
        time_hours = time_grid / 3600
        
        # Plot altitude with uncertainty bands
        self._plot_uncertainty_bands_single(axes[0, 0], time_hours, altitude_samples / 1000,
                                          confidence_levels, 'Time (hours)', 'Altitude (km)', 
                                          'Altitude Uncertainty')
        
        # Plot velocity with uncertainty bands
        self._plot_uncertainty_bands_single(axes[0, 1], time_hours, velocity_samples / 1000,
                                          confidence_levels, 'Time (hours)', 'Velocity (km/s)',
                                          'Velocity Uncertainty')
        
        # Plot Mach number if available
        if np.any(mach_samples):
            self._plot_uncertainty_bands_single(axes[1, 0], time_hours, mach_samples,
                                              confidence_levels, 'Time (hours)', 'Mach Number',
                                              'Mach Number Uncertainty')
        
        # Plot altitude vs velocity with uncertainty
        # Use final altitude and velocity distributions
        final_altitudes = altitude_samples[:, -1] / 1000
        final_velocities = velocity_samples[:, -1] / 1000
        
        axes[1, 1].scatter(final_velocities, final_altitudes, alpha=0.6, 
                         color=self.colors['primary'], s=20)
        axes[1, 1].set_xlabel('Final Velocity (km/s)')
        axes[1, 1].set_ylabel('Final Altitude (km)')
        axes[1, 1].set_title('Final Conditions Scatter')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        if show_plot:
            plt.show()
        
        return fig
    
    def _plot_uncertainty_bands_single(self, ax, x_data, y_samples, confidence_levels, 
                                     xlabel, ylabel, title):
        """Plot uncertainty bands for single variable."""
        # Calculate mean
        mean_values = np.mean(y_samples, axis=0)
        
        # Plot mean line
        ax.plot(x_data, mean_values, color=self.colors['primary'], 
               linewidth=2, label='Mean')
        
        # Plot confidence bands
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        alphas = [0.3, 0.2, 0.1]
        
        for i, conf_level in enumerate(confidence_levels):
            alpha = (100 - conf_level) / 2
            lower_percentile = np.percentile(y_samples, alpha, axis=0)
            upper_percentile = np.percentile(y_samples, 100 - alpha, axis=0)
            
            ax.fill_between(x_data, lower_percentile, upper_percentile,
                          alpha=alphas[i], color=colors[i], 
                          label=f'{conf_level}% CI')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel) 
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_performance_comparison(self, 
                                  results: Dict[str, Dict[str, float]],
                                  metrics: Optional[List[str]] = None,
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> plt.Figure:
        """Create performance metrics comparison plot.
        
        Args:
            results: Dictionary of results for different cases
            metrics: List of metrics to compare
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            # Use all available metrics from first result
            first_result = list(results.values())[0]
            metrics = list(first_result.keys())
        
        n_metrics = len(metrics)
        n_cases = len(results)
        
        # Create subplots
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        case_names = list(results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, n_cases))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            values = [results[case][metric] for case in case_names]
            
            # Bar plot
            bars = ax.bar(case_names, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2e}' if abs(value) > 1000 or abs(value) < 0.01 else f'{value:.2f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            self._save_plot(fig, save_path)
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_interactive_3d_plot(self, 
                                  trajectory: Dict[str, np.ndarray],
                                  save_path: Optional[str] = None) -> go.Figure:
        """Create interactive 3D trajectory plot using Plotly.
        
        Args:
            trajectory: Dictionary containing trajectory data
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure object
        """
        # Convert to Cartesian coordinates
        lat = trajectory['latitude']
        lon = trajectory['longitude']
        alt = trajectory['altitude']
        r = EARTH_RADIUS + alt
        
        x = r * np.cos(lat) * np.cos(lon) / 1000  # km
        y = r * np.cos(lat) * np.sin(lon) / 1000  # km
        z = r * np.sin(lat) / 1000  # km
        
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=2, color=trajectory['velocity'], 
                       colorscale='Viridis', showscale=True,
                       colorbar=dict(title="Velocity (m/s)")),
            name='Trajectory',
            text=[f'Time: {t:.1f}s<br>Alt: {a/1000:.1f}km<br>Vel: {v/1000:.2f}km/s' 
                  for t, a, v in zip(trajectory['time'], alt, trajectory['velocity'])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add Earth sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        earth_x = EARTH_RADIUS/1000 * np.outer(np.cos(u), np.sin(v))
        earth_y = EARTH_RADIUS/1000 * np.outer(np.sin(u), np.sin(v))
        earth_z = EARTH_RADIUS/1000 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=earth_x, y=earth_y, z=earth_z,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            opacity=0.3,
            showscale=False,
            name='Earth'
        ))
        
        fig.update_layout(
            title='Interactive 3D Trajectory Visualization',
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)', 
                zaxis_title='Z (km)',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(self.output_dir / save_path)
        
        return fig
    
    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        """Save matplotlib figure with appropriate format."""
        save_path = self.output_dir / filename
        
        # Determine format from extension
        if save_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            # Default to PNG
            fig.savefig(save_path.with_suffix('.png'), dpi=self.dpi, bbox_inches='tight')
        
        self.logger.info(f"Plot saved to {save_path}")