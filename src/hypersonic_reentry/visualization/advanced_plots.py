"""Advanced visualization tools for hypersonic reentry analysis.

This module provides enhanced plotting capabilities including:
- Animated trajectory visualizations
- Interactive parameter sweep plots
- Statistical distribution visualizations
- Advanced uncertainty quantification plots
- Performance dashboard creation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import warnings

from ..utils.constants import EARTH_RADIUS, DEG_TO_RAD, RAD_TO_DEG


class AdvancedPlotter:
    """Advanced plotting system for comprehensive visualization of simulation results.
    
    Provides sophisticated visualization capabilities including animations,
    interactive plots, statistical visualizations, and dashboard creation.
    """
    
    def __init__(self, 
                 output_directory: str = "plots",
                 style_theme: str = "publication",
                 color_palette: str = "viridis",
                 dpi: int = 300):
        """Initialize advanced plotter.
        
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
        
        # Set up visualization styles
        self._setup_styles()
        
        self.logger.info("Initialized AdvancedPlotter")
    
    def _setup_styles(self):
        """Set up matplotlib and seaborn styles."""
        # Set matplotlib style
        plt.style.use(['default'])
        
        # Configure for publication quality
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
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
            'savefig.bbox': 'tight'
        })
        
        # Set seaborn style
        sns.set_palette(self.color_palette)
    
    def create_animated_trajectory(self, 
                                  trajectory: Dict[str, np.ndarray],
                                  save_path: Optional[str] = None,
                                  frame_interval: int = 100,
                                  show_earth: bool = True) -> None:
        """Create animated trajectory visualization.
        
        Args:
            trajectory: Dictionary containing trajectory data
            save_path: Path to save animation (GIF or MP4)
            frame_interval: Time between frames in milliseconds
            show_earth: Whether to show Earth sphere
        """
        self.logger.info("Creating animated trajectory visualization")
        
        # Convert to Cartesian coordinates
        lat = trajectory['latitude']
        lon = trajectory['longitude']
        alt = trajectory['altitude']
        r = EARTH_RADIUS + alt
        
        x = r * np.cos(lat) * np.cos(lon) / 1000  # km
        y = r * np.cos(lat) * np.sin(lon) / 1000  # km
        z = r * np.sin(lat) / 1000  # km
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Earth sphere
        if show_earth:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            earth_x = EARTH_RADIUS/1000 * np.outer(np.cos(u), np.sin(v))
            earth_y = EARTH_RADIUS/1000 * np.outer(np.sin(u), np.sin(v))
            earth_z = EARTH_RADIUS/1000 * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='lightblue')
        
        # Initialize trajectory elements
        trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        current_point, = ax.plot([], [], [], 'ro', markersize=8, label='Vehicle')
        trail_points, = ax.plot([], [], [], 'b-', alpha=0.5, linewidth=1)
        
        # Set up plot
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Animated Hypersonic Reentry Trajectory')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = max(np.ptp(x), np.ptp(y), np.ptp(z)) / 2
        mid_x, mid_y, mid_z = np.mean(x), np.mean(y), np.mean(z)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        def animate(frame):
            # Update trajectory line up to current frame
            trajectory_line.set_data_3d(x[:frame], y[:frame], z[:frame])
            
            # Update current point
            if frame > 0:
                current_point.set_data_3d([x[frame-1]], [y[frame-1]], [z[frame-1]])
                
                # Show trail (last 50 points)
                trail_start = max(0, frame - 50)
                trail_points.set_data_3d(x[trail_start:frame], y[trail_start:frame], z[trail_start:frame])
            
            return trajectory_line, current_point, trail_points
        
        # Create animation
        frames = len(x)
        anim = FuncAnimation(fig, animate, frames=frames, interval=frame_interval, 
                           blit=True, repeat=True)
        
        # Save animation
        if save_path:
            save_path = self.output_dir / save_path
            
            if save_path.suffix.lower() == '.gif':
                anim.save(save_path, writer='pillow', fps=1000//frame_interval)
            elif save_path.suffix.lower() == '.mp4':
                anim.save(save_path, writer='ffmpeg', fps=1000//frame_interval)
            
            self.logger.info(f"Animation saved: {save_path}")
        
        plt.show()
    
    def create_interactive_parameter_sweep(self, 
                                         results_data: Dict[str, Any],
                                         parameter_ranges: Dict[str, Tuple[float, float, int]],
                                         output_metric: str = 'downrange') -> go.Figure:
        """Create interactive parameter sweep visualization.
        
        Args:
            results_data: Dictionary containing simulation results
            parameter_ranges: Dictionary of parameter ranges (min, max, steps)
            output_metric: Output metric to visualize
            
        Returns:
            Plotly figure with interactive parameter sweep
        """
        self.logger.info("Creating interactive parameter sweep visualization")
        
        # Create parameter grid
        param_names = list(parameter_ranges.keys())
        
        if len(param_names) == 1:
            # 1D parameter sweep
            param_name = param_names[0]
            param_min, param_max, n_steps = parameter_ranges[param_name]
            param_values = np.linspace(param_min, param_max, n_steps)
            
            fig = go.Figure()
            
            # Simulate parameter sweep (placeholder - replace with actual sweep)
            output_values = self._simulate_parameter_sweep_1d(param_values, output_metric)
            
            fig.add_trace(go.Scatter(
                x=param_values,
                y=output_values,
                mode='lines+markers',
                name=f'{output_metric} vs {param_name}',
                line=dict(width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f'Parameter Sweep: {output_metric} vs {param_name}',
                xaxis_title=param_name,
                yaxis_title=output_metric,
                template='plotly_white'
            )
        
        elif len(param_names) == 2:
            # 2D parameter sweep (heatmap)
            param1, param2 = param_names
            
            param1_min, param1_max, n1 = parameter_ranges[param1]
            param2_min, param2_max, n2 = parameter_ranges[param2]
            
            param1_values = np.linspace(param1_min, param1_max, n1)
            param2_values = np.linspace(param2_min, param2_max, n2)
            
            # Create meshgrid
            P1, P2 = np.meshgrid(param1_values, param2_values)
            
            # Simulate parameter sweep (placeholder)
            Z = self._simulate_parameter_sweep_2d(P1, P2, output_metric)
            
            fig = go.Figure(data=go.Heatmap(
                x=param1_values,
                y=param2_values,
                z=Z,
                colorscale='Viridis',
                colorbar=dict(title=output_metric)
            ))
            
            fig.update_layout(
                title=f'Parameter Sweep: {output_metric} vs {param1} and {param2}',
                xaxis_title=param1,
                yaxis_title=param2,
                template='plotly_white'
            )
        
        return fig
    
    def create_statistical_dashboard(self, 
                                   statistical_results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> go.Figure:
        """Create comprehensive statistical analysis dashboard.
        
        Args:
            statistical_results: Results from statistical analysis
            save_path: Path to save dashboard HTML
            
        Returns:
            Plotly dashboard figure
        """
        self.logger.info("Creating statistical analysis dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Distribution Comparison', 'Correlation Heatmap', 'Outlier Detection',
                'Reliability Analysis', 'PCA Analysis', 'Clustering Results',
                'Hypothesis Test Results', 'Confidence Intervals', 'Risk Assessment'
            ],
            specs=[
                [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'table'}, {'type': 'xy'}, {'type': 'xy'}]
            ]
        )
        
        # 1. Distribution comparison (example with placeholder data)
        if 'distribution_analysis' in statistical_results:
            self._add_distribution_plot(fig, statistical_results['distribution_analysis'], 1, 1)
        
        # 2. Correlation heatmap
        if 'correlation_analysis' in statistical_results:
            self._add_correlation_heatmap(fig, statistical_results['correlation_analysis'], 1, 2)
        
        # 3. Outlier detection
        if 'outlier_analysis' in statistical_results:
            self._add_outlier_plot(fig, statistical_results['outlier_analysis'], 1, 3)
        
        # 4. Reliability analysis
        if 'reliability_analysis' in statistical_results:
            self._add_reliability_plot(fig, statistical_results['reliability_analysis'], 2, 1)
        
        # 5. PCA analysis
        if 'dimensionality_analysis' in statistical_results:
            self._add_pca_plot(fig, statistical_results['dimensionality_analysis'], 2, 2)
        
        # 6. Clustering results
        if 'clustering_analysis' in statistical_results:
            self._add_clustering_plot(fig, statistical_results['clustering_analysis'], 2, 3)
        
        # 7. Hypothesis test table
        if 'hypothesis_tests' in statistical_results:
            self._add_hypothesis_table(fig, statistical_results['hypothesis_tests'], 3, 1)
        
        # 8. Confidence intervals
        if 'descriptive_statistics' in statistical_results:
            self._add_confidence_intervals(fig, statistical_results['descriptive_statistics'], 3, 2)
        
        # 9. Risk assessment
        if 'reliability_analysis' in statistical_results:
            self._add_risk_assessment(fig, statistical_results['reliability_analysis'], 3, 3)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Statistical Analysis Dashboard",
            template='plotly_white',
            showlegend=True
        )
        
        if save_path:
            save_path = self.output_dir / save_path
            fig.write_html(save_path)
            self.logger.info(f"Dashboard saved: {save_path}")
        
        return fig
    
    def create_uncertainty_visualization(self, 
                                       mc_results: Dict[str, Any],
                                       confidence_levels: List[int] = [68, 90, 95, 99]) -> go.Figure:
        """Create comprehensive uncertainty visualization.
        
        Args:
            mc_results: Monte Carlo simulation results
            confidence_levels: Confidence levels to display
            
        Returns:
            Plotly figure with uncertainty visualization
        """
        self.logger.info("Creating uncertainty visualization")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Probability Distributions',
                'Confidence Intervals',
                'Uncertainty Bounds Over Time',
                'Risk Contours'
            ]
        )
        
        # Extract data
        if 'statistical_summary' in mc_results:
            metrics = list(mc_results['statistical_summary'].keys())
            
            # 1. Probability distributions
            for i, metric in enumerate(metrics[:3]):  # Show top 3 metrics
                if 'probability_distributions' in mc_results:
                    dist_data = mc_results['probability_distributions'][metric]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=dist_data['bin_centers'],
                            y=dist_data['probability_density'],
                            mode='lines',
                            name=f'{metric} PDF',
                            fill='tonexty' if i > 0 else 'tozeroy'
                        ),
                        row=1, col=1
                    )
            
            # 2. Confidence intervals
            metric_names = []
            means = []
            ci_lower = []
            ci_upper = []
            
            for metric in metrics:
                if metric in mc_results.get('uncertainty_bounds', {}):
                    metric_names.append(metric)
                    means.append(mc_results['statistical_summary'][metric]['mean'])
                    
                    # Use 95% confidence interval
                    bounds = mc_results['uncertainty_bounds'][metric]['95%']
                    ci_lower.append(bounds['lower'])
                    ci_upper.append(bounds['upper'])
            
            fig.add_trace(
                go.Scatter(
                    x=means,
                    y=metric_names,
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        arrayminus=[m - l for m, l in zip(means, ci_lower)],
                        arrayplus=[u - m for m, u in zip(means, ci_upper)]
                    ),
                    mode='markers',
                    marker=dict(size=8),
                    name='95% Confidence Intervals'
                ),
                row=1, col=2
            )
        
        # 3. Risk assessment
        if 'risk_metrics' in mc_results:
            self._add_risk_contours(fig, mc_results['risk_metrics'], 2, 2)
        
        fig.update_layout(
            height=800,
            title_text="Comprehensive Uncertainty Analysis",
            template='plotly_white'
        )
        
        return fig
    
    def create_optimization_comparison(self, 
                                     comparison_results: Dict[str, Any],
                                     save_path: Optional[str] = None) -> go.Figure:
        """Create optimization method comparison visualization.
        
        Args:
            comparison_results: Optimization comparison results
            save_path: Path to save plot
            
        Returns:
            Plotly comparison figure
        """
        self.logger.info("Creating optimization comparison visualization")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Convergence Comparison',
                'Performance Comparison',
                'Success Rate Analysis',
                'Computational Efficiency'
            ]
        )
        
        # Extract optimization results
        opt_results = comparison_results.get('optimization_results', {})
        perf_comparison = comparison_results.get('performance_comparison', {})
        
        methods = list(opt_results.keys())
        scenarios = list(opt_results[methods[0]].keys()) if methods else []
        
        # 1. Convergence comparison
        for method in methods:
            iterations = []
            scenario_names = []
            
            for scenario in scenarios:
                if opt_results[method][scenario]['success']:
                    iterations.append(opt_results[method][scenario]['num_iterations'])
                    scenario_names.append(scenario)
            
            fig.add_trace(
                go.Scatter(
                    x=scenario_names,
                    y=iterations,
                    mode='lines+markers',
                    name=f'{method} iterations',
                    line=dict(width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # 2. Performance comparison (objective values)
        for method in methods:
            objectives = []
            scenario_names = []
            
            for scenario in scenarios:
                if opt_results[method][scenario]['success']:
                    objectives.append(opt_results[method][scenario]['final_objective'])
                    scenario_names.append(scenario)
            
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=objectives,
                    name=f'{method} objective',
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # 3. Success rates
        if 'success_rates' in perf_comparison:
            methods = list(perf_comparison['success_rates'].keys())
            success_rates = [perf_comparison['success_rates'][method] * 100 for method in methods]
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=success_rates,
                    name='Success Rate (%)',
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )
        
        # 4. Computational efficiency
        if 'convergence_comparison' in perf_comparison:
            methods = list(perf_comparison['convergence_comparison'].keys())
            avg_times = [perf_comparison['convergence_comparison'][method]['avg_time'] 
                        for method in methods]
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=avg_times,
                    name='Average Time (s)',
                    marker_color='lightcoral'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Optimization Method Comparison",
            template='plotly_white'
        )
        
        if save_path:
            save_path = self.output_dir / save_path
            fig.write_html(save_path)
            self.logger.info(f"Optimization comparison saved: {save_path}")
        
        return fig
    
    def create_trajectory_ensemble_plot(self, 
                                      trajectories: List[Dict[str, np.ndarray]],
                                      nominal_trajectory: Optional[Dict[str, np.ndarray]] = None,
                                      confidence_levels: List[int] = [68, 95, 99]) -> go.Figure:
        """Create trajectory ensemble plot with uncertainty bands.
        
        Args:
            trajectories: List of trajectory dictionaries
            nominal_trajectory: Nominal trajectory for comparison
            confidence_levels: Confidence levels for uncertainty bands
            
        Returns:
            Plotly figure with trajectory ensemble
        """
        self.logger.info("Creating trajectory ensemble visualization")
        
        fig = go.Figure()
        
        # Process trajectory ensemble
        if trajectories:
            # Find common time grid
            max_time = min([traj['time'].max() for traj in trajectories])
            time_grid = np.linspace(0, max_time, 1000)
            
            # Interpolate all trajectories
            altitude_ensemble = []
            velocity_ensemble = []
            
            for traj in trajectories:
                alt_interp = np.interp(time_grid, traj['time'], traj['altitude'])
                vel_interp = np.interp(time_grid, traj['time'], traj['velocity'])
                
                altitude_ensemble.append(alt_interp)
                velocity_ensemble.append(vel_interp)
            
            altitude_ensemble = np.array(altitude_ensemble)
            velocity_ensemble = np.array(velocity_ensemble)
            
            # Calculate uncertainty bands
            alt_mean = np.mean(altitude_ensemble, axis=0)
            vel_mean = np.mean(velocity_ensemble, axis=0)
            
            # Add confidence bands
            colors = ['rgba(31,119,180,0.1)', 'rgba(31,119,180,0.2)', 'rgba(31,119,180,0.3)']
            
            for i, conf_level in enumerate(confidence_levels):
                alpha = (100 - conf_level) / 2
                
                alt_lower = np.percentile(altitude_ensemble, alpha, axis=0)
                alt_upper = np.percentile(altitude_ensemble, 100 - alpha, axis=0)
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([time_grid, time_grid[::-1]]),
                    y=np.concatenate([alt_upper/1000, alt_lower[::-1]/1000]),
                    fill='toself',
                    fillcolor=colors[i],
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{conf_level}% CI',
                    showlegend=True if i == 0 else False
                ))
            
            # Add mean trajectory
            fig.add_trace(go.Scatter(
                x=time_grid,
                y=alt_mean/1000,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Mean Trajectory'
            ))
        
        # Add nominal trajectory if provided
        if nominal_trajectory:
            fig.add_trace(go.Scatter(
                x=nominal_trajectory['time'],
                y=nominal_trajectory['altitude']/1000,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Nominal Trajectory'
            ))
        
        fig.update_layout(
            title='Trajectory Ensemble with Uncertainty Bands',
            xaxis_title='Time (s)',
            yaxis_title='Altitude (km)',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def _simulate_parameter_sweep_1d(self, param_values: np.ndarray, output_metric: str) -> np.ndarray:
        """Simulate 1D parameter sweep (placeholder implementation)."""
        # This is a placeholder - in practice, would run actual simulations
        if output_metric == 'downrange':
            return 1000000 + 500000 * np.sin(param_values) + 100000 * np.random.normal(0, 0.1, len(param_values))
        else:
            return np.random.normal(1000, 100, len(param_values))
    
    def _simulate_parameter_sweep_2d(self, P1: np.ndarray, P2: np.ndarray, output_metric: str) -> np.ndarray:
        """Simulate 2D parameter sweep (placeholder implementation)."""
        # This is a placeholder - in practice, would run actual simulations
        return np.sin(P1) * np.cos(P2) + 0.1 * np.random.normal(0, 1, P1.shape)
    
    def _add_distribution_plot(self, fig, dist_analysis, row, col):
        """Add distribution comparison plot to subplot."""
        # Placeholder implementation
        x = np.linspace(0, 10, 100)
        y = np.exp(-x/2)
        
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', name='Distribution'),
            row=row, col=col
        )
    
    def _add_correlation_heatmap(self, fig, corr_analysis, row, col):
        """Add correlation heatmap to subplot."""
        if 'pearson_correlation' in corr_analysis:
            corr_matrix = corr_analysis['pearson_correlation']['matrix']
            variables = list(corr_matrix.keys())
            
            # Convert to matrix format
            matrix = np.array([[corr_matrix[v1][v2] for v2 in variables] for v1 in variables])
            
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=variables,
                    y=variables,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=False
                ),
                row=row, col=col
            )
    
    def _add_outlier_plot(self, fig, outlier_analysis, row, col):
        """Add outlier detection plot to subplot."""
        # Placeholder implementation
        metrics = list(outlier_analysis.keys())[:3]
        outlier_counts = [outlier_analysis[metric]['iqr_outliers']['count'] for metric in metrics]
        
        fig.add_trace(
            go.Bar(x=metrics, y=outlier_counts, name='Outlier Count'),
            row=row, col=col
        )
    
    def _add_reliability_plot(self, fig, reliability_analysis, row, col):
        """Add reliability analysis plot to subplot."""
        if 'failure_probabilities' in reliability_analysis:
            metrics = list(reliability_analysis['failure_probabilities'].keys())
            reliabilities = [reliability_analysis['failure_probabilities'][m]['reliability'] 
                           for m in metrics]
            
            fig.add_trace(
                go.Bar(x=metrics, y=reliabilities, name='Reliability'),
                row=row, col=col
            )
    
    def _add_pca_plot(self, fig, pca_analysis, row, col):
        """Add PCA analysis plot to subplot."""
        if 'explained_variance_ratio' in pca_analysis:
            variance_ratios = pca_analysis['explained_variance_ratio'][:5]  # Top 5 components
            components = [f'PC{i+1}' for i in range(len(variance_ratios))]
            
            fig.add_trace(
                go.Bar(x=components, y=variance_ratios, name='Explained Variance'),
                row=row, col=col
            )
    
    def _add_clustering_plot(self, fig, clustering_analysis, row, col):
        """Add clustering analysis plot to subplot."""
        if 'silhouette_analysis' in clustering_analysis:
            k_values = clustering_analysis['silhouette_analysis']['k_values']
            sil_scores = clustering_analysis['silhouette_analysis']['silhouette_scores']
            
            fig.add_trace(
                go.Scatter(x=k_values, y=sil_scores, mode='lines+markers', name='Silhouette Score'),
                row=row, col=col
            )
    
    def _add_hypothesis_table(self, fig, hypothesis_tests, row, col):
        """Add hypothesis test results table to subplot."""
        # Create summary table data
        headers = ['Test', 'Variable', 'Statistic', 'P-Value', 'Result']
        values = [[], [], [], [], []]
        
        if 'normality_tests' in hypothesis_tests:
            for var, tests in hypothesis_tests['normality_tests'].items():
                for test_name, test_result in tests.items():
                    values[0].append(test_name)
                    values[1].append(var)
                    values[2].append(f"{test_result.get('statistic', 0):.4f}")
                    values[3].append(f"{test_result.get('p_value', 0):.4f}")
                    values[4].append('Normal' if test_result.get('is_normal', False) else 'Non-normal')
        
        fig.add_trace(
            go.Table(
                header=dict(values=headers),
                cells=dict(values=values)
            ),
            row=row, col=col
        )
    
    def _add_confidence_intervals(self, fig, desc_stats, row, col):
        """Add confidence intervals plot to subplot."""
        metrics = list(desc_stats.keys())[:5]  # Top 5 metrics
        means = [desc_stats[m]['mean'] for m in metrics]
        
        # Create error bars (using std as approximation)
        errors = [desc_stats[m]['std'] for m in metrics]
        
        fig.add_trace(
            go.Scatter(
                x=means,
                y=metrics,
                error_x=dict(type='data', array=errors),
                mode='markers',
                marker=dict(size=8),
                name='Confidence Intervals'
            ),
            row=row, col=col
        )
    
    def _add_risk_assessment(self, fig, reliability_analysis, row, col):
        """Add risk assessment plot to subplot."""
        if 'system_reliability' in reliability_analysis:
            system_rel = reliability_analysis['system_reliability']['overall_reliability']
            individual_rels = reliability_analysis['system_reliability']['individual_reliabilities']
            
            fig.add_trace(
                go.Bar(
                    x=['System'] + [f'Component {i+1}' for i in range(len(individual_rels))],
                    y=[system_rel] + individual_rels,
                    name='Reliability'
                ),
                row=row, col=col
            )
    
    def _add_risk_contours(self, fig, risk_metrics, row, col):
        """Add risk contour plot to subplot."""
        # Placeholder implementation for risk contours
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.random.random((10, 10))
        
        fig.add_trace(
            go.Heatmap(x=x, y=y, z=z, showscale=False),
            row=row, col=col
        )