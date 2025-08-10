"""Advanced statistical analysis tools for hypersonic reentry simulation results.

This module provides comprehensive statistical analysis including:
- Distribution fitting and testing
- Correlation analysis
- Statistical significance testing
- Risk assessment and reliability analysis
- Time series analysis for trajectory data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from scipy import stats
from scipy.stats import kstest, normaltest, jarque_bera
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class StatisticalAnalyzer:
    """Advanced statistical analysis for simulation results.
    
    Provides comprehensive statistical analysis including distribution fitting,
    correlation analysis, hypothesis testing, and reliability assessment.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize statistical analyzer.
        
        Args:
            confidence_level: Default confidence level for intervals and tests
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        
        # Statistical distributions for fitting
        self.distributions = [
            stats.norm, stats.lognorm, stats.gamma, stats.beta,
            stats.weibull_min, stats.weibull_max, stats.exponweib,
            stats.chi2, stats.t, stats.uniform
        ]
        
        self.logger.info(f"Initialized StatisticalAnalyzer with {confidence_level:.1%} confidence level")
    
    def comprehensive_analysis(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis of results.
        
        Args:
            results_data: Dictionary containing simulation results
            
        Returns:
            Dictionary containing complete statistical analysis
        """
        self.logger.info("Starting comprehensive statistical analysis")
        
        analysis = {
            'descriptive_statistics': {},
            'distribution_analysis': {},
            'correlation_analysis': {},
            'outlier_analysis': {},
            'reliability_analysis': {},
            'hypothesis_tests': {},
            'dimensionality_analysis': {},
            'clustering_analysis': {}
        }
        
        # Extract performance metrics data
        if 'raw_data' in results_data and 'performance_metrics' in results_data['raw_data']:
            df = pd.DataFrame(results_data['raw_data']['performance_metrics'])
            
            # Descriptive statistics
            analysis['descriptive_statistics'] = self.calculate_descriptive_statistics(df)
            
            # Distribution fitting and testing
            analysis['distribution_analysis'] = self.fit_distributions(df)
            
            # Correlation analysis
            analysis['correlation_analysis'] = self.correlation_analysis(df)
            
            # Outlier detection
            analysis['outlier_analysis'] = self.detect_outliers(df)
            
            # Reliability analysis
            analysis['reliability_analysis'] = self.reliability_analysis(df)
            
            # Hypothesis testing
            analysis['hypothesis_tests'] = self.hypothesis_testing(df)
            
            # Principal Component Analysis
            analysis['dimensionality_analysis'] = self.dimensionality_analysis(df)
            
            # Clustering analysis
            analysis['clustering_analysis'] = self.clustering_analysis(df)
        
        self.logger.info("Comprehensive statistical analysis completed")
        return analysis
    
    def calculate_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics.
        
        Args:
            df: DataFrame containing simulation results
            
        Returns:
            Dictionary containing descriptive statistics for each variable
        """
        desc_stats = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            data = df[column].dropna()
            
            if len(data) == 0:
                continue
            
            # Basic statistics
            stats_dict = {
                'count': int(len(data)),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'median': float(data.median()),
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75)),
                'iqr': float(data.quantile(0.75) - data.quantile(0.25))
            }
            
            # Shape statistics
            try:
                stats_dict.update({
                    'skewness': float(stats.skew(data)),
                    'kurtosis': float(stats.kurtosis(data)),
                    'excess_kurtosis': float(stats.kurtosis(data, fisher=True))
                })
            except:
                stats_dict.update({
                    'skewness': np.nan,
                    'kurtosis': np.nan,
                    'excess_kurtosis': np.nan
                })
            
            # Robust statistics
            stats_dict.update({
                'mad': float(stats.median_abs_deviation(data)),  # Median Absolute Deviation
                'trimmed_mean_10': float(stats.trim_mean(data, 0.1)),  # 10% trimmed mean
                'geometric_mean': float(stats.gmean(data)) if np.all(data > 0) else np.nan,
                'harmonic_mean': float(stats.hmean(data)) if np.all(data > 0) else np.nan
            })
            
            # Confidence intervals
            stats_dict['confidence_intervals'] = self._calculate_confidence_intervals(data)
            
            # Coefficient of variation
            if stats_dict['mean'] != 0:
                stats_dict['coefficient_of_variation'] = stats_dict['std'] / abs(stats_dict['mean'])
            else:
                stats_dict['coefficient_of_variation'] = np.inf
            
            desc_stats[column] = stats_dict
        
        return desc_stats
    
    def fit_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit statistical distributions to each variable.
        
        Args:
            df: DataFrame containing simulation results
            
        Returns:
            Dictionary containing distribution fitting results
        """
        distribution_results = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            data = df[column].dropna()
            
            if len(data) < 10:  # Need minimum samples for fitting
                continue
            
            column_results = {
                'fitted_distributions': {},
                'goodness_of_fit': {},
                'best_distribution': None,
                'normality_tests': {}
            }
            
            # Test for normality
            column_results['normality_tests'] = self._test_normality(data)
            
            # Fit distributions
            best_dist = None
            best_aic = np.inf
            
            for distribution in self.distributions:
                try:
                    # Fit distribution
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        params = distribution.fit(data)
                    
                    # Calculate AIC (Akaike Information Criterion)
                    log_likelihood = np.sum(distribution.logpdf(data, *params))
                    k = len(params)  # Number of parameters
                    aic = 2 * k - 2 * log_likelihood
                    
                    # Calculate BIC (Bayesian Information Criterion)
                    n = len(data)
                    bic = k * np.log(n) - 2 * log_likelihood
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = kstest(data, lambda x: distribution.cdf(x, *params))
                    
                    dist_results = {
                        'parameters': params,
                        'aic': float(aic),
                        'bic': float(bic),
                        'log_likelihood': float(log_likelihood),
                        'ks_statistic': float(ks_stat),
                        'ks_p_value': float(ks_p)
                    }
                    
                    column_results['fitted_distributions'][distribution.name] = dist_results
                    
                    # Track best distribution based on AIC
                    if aic < best_aic:
                        best_aic = aic
                        best_dist = distribution.name
                
                except Exception as e:
                    self.logger.warning(f"Failed to fit {distribution.name} to {column}: {str(e)}")
                    continue
            
            column_results['best_distribution'] = best_dist
            distribution_results[column] = column_results
        
        return distribution_results
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis.
        
        Args:
            df: DataFrame containing simulation results
            
        Returns:
            Dictionary containing correlation analysis results
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {}
        
        correlation_results = {
            'pearson_correlation': {},
            'spearman_correlation': {},
            'kendall_correlation': {},
            'partial_correlations': {},
            'correlation_significance': {}
        }
        
        # Pearson correlation
        pearson_corr = numeric_df.corr(method='pearson')
        correlation_results['pearson_correlation'] = {
            'matrix': pearson_corr.to_dict(),
            'strong_correlations': self._find_strong_correlations(pearson_corr, threshold=0.7)
        }
        
        # Spearman correlation (rank-based)
        spearman_corr = numeric_df.corr(method='spearman')
        correlation_results['spearman_correlation'] = {
            'matrix': spearman_corr.to_dict(),
            'strong_correlations': self._find_strong_correlations(spearman_corr, threshold=0.7)
        }
        
        # Kendall correlation (tau)
        kendall_corr = numeric_df.corr(method='kendall')
        correlation_results['kendall_correlation'] = {
            'matrix': kendall_corr.to_dict(),
            'strong_correlations': self._find_strong_correlations(kendall_corr, threshold=0.5)
        }
        
        # Correlation significance testing
        correlation_results['correlation_significance'] = self._test_correlation_significance(numeric_df)
        
        return correlation_results
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple methods.
        
        Args:
            df: DataFrame containing simulation results
            
        Returns:
            Dictionary containing outlier detection results
        """
        outlier_results = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            data = df[column].dropna()
            
            if len(data) < 10:
                continue
            
            column_outliers = {
                'z_score_outliers': {},
                'iqr_outliers': {},
                'modified_z_score_outliers': {},
                'isolation_forest_outliers': {}
            }
            
            # Z-score method (assumes normal distribution)
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]
            column_outliers['z_score_outliers'] = {
                'count': len(z_outliers),
                'indices': z_outliers.index.tolist(),
                'values': z_outliers.tolist(),
                'percentage': float(len(z_outliers) / len(data) * 100)
            }
            
            # IQR method (robust to distribution shape)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            column_outliers['iqr_outliers'] = {
                'count': len(iqr_outliers),
                'indices': iqr_outliers.index.tolist(),
                'values': iqr_outliers.tolist(),
                'percentage': float(len(iqr_outliers) / len(data) * 100),
                'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
            }
            
            # Modified Z-score (using median)
            median = np.median(data)
            mad = stats.median_abs_deviation(data)
            modified_z_scores = 0.6745 * (data - median) / mad
            mod_z_outliers = data[np.abs(modified_z_scores) > 3.5]
            
            column_outliers['modified_z_score_outliers'] = {
                'count': len(mod_z_outliers),
                'indices': mod_z_outliers.index.tolist(),
                'values': mod_z_outliers.tolist(),
                'percentage': float(len(mod_z_outliers) / len(data) * 100)
            }
            
            outlier_results[column] = column_outliers
        
        return outlier_results
    
    def reliability_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform reliability and risk analysis.
        
        Args:
            df: DataFrame containing simulation results
            
        Returns:
            Dictionary containing reliability analysis results
        """
        reliability_results = {
            'failure_probabilities': {},
            'reliability_metrics': {},
            'safety_margins': {},
            'system_reliability': {}
        }
        
        # Define failure criteria for different metrics
        failure_criteria = {
            'final_altitude': {'min': 25000, 'max': 35000},
            'max_heat_rate': {'max': 5e6},
            'downrange': {'min': 1000000, 'max': 3000000},
            'final_velocity': {'min': 200, 'max': 500}
        }
        
        for metric, criteria in failure_criteria.items():
            if metric in df.columns:
                data = df[metric].dropna()
                
                if len(data) == 0:
                    continue
                
                metric_reliability = {
                    'sample_size': len(data),
                    'failure_modes': {}
                }
                
                # Check different failure modes
                if 'min' in criteria:
                    below_min = data < criteria['min']
                    metric_reliability['failure_modes']['below_minimum'] = {
                        'probability': float(below_min.mean()),
                        'count': int(below_min.sum()),
                        'threshold': criteria['min']
                    }
                
                if 'max' in criteria:
                    above_max = data > criteria['max']
                    metric_reliability['failure_modes']['above_maximum'] = {
                        'probability': float(above_max.mean()),
                        'count': int(above_max.sum()),
                        'threshold': criteria['max']
                    }
                
                # Overall success probability
                success_mask = np.ones(len(data), dtype=bool)
                if 'min' in criteria:
                    success_mask &= (data >= criteria['min'])
                if 'max' in criteria:
                    success_mask &= (data <= criteria['max'])
                
                metric_reliability['success_probability'] = float(success_mask.mean())
                metric_reliability['reliability'] = float(success_mask.mean())
                
                # Calculate confidence interval for reliability
                n_success = success_mask.sum()
                n_total = len(data)
                
                # Wilson score interval
                p = n_success / n_total
                z = stats.norm.ppf(1 - self.alpha/2)
                
                denominator = 1 + z**2 / n_total
                centre_adjusted_p = (p + z**2 / (2 * n_total)) / denominator
                adjusted_std = np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denominator
                
                lower_bound = centre_adjusted_p - z * adjusted_std
                upper_bound = centre_adjusted_p + z * adjusted_std
                
                metric_reliability['reliability_confidence_interval'] = {
                    'lower': max(0, float(lower_bound)),
                    'upper': min(1, float(upper_bound)),
                    'confidence_level': self.confidence_level
                }
                
                reliability_results['failure_probabilities'][metric] = metric_reliability
        
        # System-level reliability (assuming independence)
        individual_reliabilities = []
        for metric_results in reliability_results['failure_probabilities'].values():
            individual_reliabilities.append(metric_results['reliability'])
        
        if individual_reliabilities:
            system_reliability = np.prod(individual_reliabilities)
            reliability_results['system_reliability'] = {
                'overall_reliability': float(system_reliability),
                'individual_reliabilities': individual_reliabilities,
                'assumption': 'independence_assumed'
            }
        
        return reliability_results
    
    def hypothesis_testing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical hypothesis tests.
        
        Args:
            df: DataFrame containing simulation results
            
        Returns:
            Dictionary containing hypothesis test results
        """
        test_results = {
            'normality_tests': {},
            'variance_tests': {},
            'mean_comparison_tests': {},
            'distribution_comparison_tests': {}
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Normality tests for each variable
        for column in numeric_columns:
            data = df[column].dropna()
            
            if len(data) < 8:  # Minimum sample size for tests
                continue
            
            column_tests = {}
            
            # Shapiro-Wilk test (best for small samples)
            if len(data) <= 5000:
                try:
                    sw_stat, sw_p = stats.shapiro(data)
                    column_tests['shapiro_wilk'] = {
                        'statistic': float(sw_stat),
                        'p_value': float(sw_p),
                        'is_normal': sw_p > self.alpha
                    }
                except:
                    pass
            
            # Anderson-Darling test
            try:
                ad_result = stats.anderson(data, dist='norm')
                column_tests['anderson_darling'] = {
                    'statistic': float(ad_result.statistic),
                    'critical_values': ad_result.critical_values.tolist(),
                    'significance_levels': ad_result.significance_level.tolist()
                }
            except:
                pass
            
            # Jarque-Bera test
            try:
                jb_stat, jb_p = jarque_bera(data)
                column_tests['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_p),
                    'is_normal': jb_p > self.alpha
                }
            except:
                pass
            
            # D'Agostino test
            try:
                dag_stat, dag_p = stats.normaltest(data)
                column_tests['dagostino'] = {
                    'statistic': float(dag_stat),
                    'p_value': float(dag_p),
                    'is_normal': dag_p > self.alpha
                }
            except:
                pass
            
            test_results['normality_tests'][column] = column_tests
        
        # Pairwise comparisons between variables
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                data1 = df[col1].dropna()
                data2 = df[col2].dropna()
                
                if len(data1) < 5 or len(data2) < 5:
                    continue
                
                pair_key = f"{col1}_vs_{col2}"
                pair_tests = {}
                
                # Two-sample t-test (assumes normality)
                try:
                    t_stat, t_p = stats.ttest_ind(data1, data2)
                    pair_tests['t_test'] = {
                        'statistic': float(t_stat),
                        'p_value': float(t_p),
                        'significantly_different': t_p < self.alpha
                    }
                except:
                    pass
                
                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    pair_tests['mann_whitney'] = {
                        'statistic': float(u_stat),
                        'p_value': float(u_p),
                        'significantly_different': u_p < self.alpha
                    }
                except:
                    pass
                
                # Kolmogorov-Smirnov test (distribution comparison)
                try:
                    ks_stat, ks_p = stats.ks_2samp(data1, data2)
                    pair_tests['ks_test'] = {
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p),
                        'different_distributions': ks_p < self.alpha
                    }
                except:
                    pass
                
                if pair_tests:
                    test_results['distribution_comparison_tests'][pair_key] = pair_tests
        
        return test_results
    
    def dimensionality_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform principal component analysis and dimensionality reduction.
        
        Args:
            df: DataFrame containing simulation results
            
        Returns:
            Dictionary containing PCA results
        """
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return {}
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Perform PCA
        pca = PCA()
        pca_data = pca.fit_transform(scaled_data)
        
        pca_results = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'singular_values': pca.singular_values_.tolist(),
            'components': pca.components_.tolist(),
            'feature_names': numeric_df.columns.tolist()
        }
        
        # Determine number of components for different variance thresholds
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        for threshold in [0.8, 0.9, 0.95, 0.99]:
            n_components = np.argmax(cumvar >= threshold) + 1
            pca_results[f'components_for_{int(threshold*100)}pct_variance'] = int(n_components)
        
        # Component interpretation
        component_importance = {}
        for i, component in enumerate(pca.components_[:5]):  # Top 5 components
            # Find the most important features for this component
            abs_loadings = np.abs(component)
            top_features_idx = np.argsort(abs_loadings)[-3:]  # Top 3 features
            
            component_info = {
                'explained_variance': float(pca.explained_variance_ratio_[i]),
                'top_features': []
            }
            
            for idx in reversed(top_features_idx):
                component_info['top_features'].append({
                    'feature': numeric_df.columns[idx],
                    'loading': float(component[idx]),
                    'abs_loading': float(abs_loadings[idx])
                })
            
            component_importance[f'PC{i+1}'] = component_info
        
        pca_results['component_interpretation'] = component_importance
        
        return pca_results
    
    def clustering_analysis(self, df: pd.DataFrame, max_clusters: int = 8) -> Dict[str, Any]:
        """Perform clustering analysis to identify patterns in results.
        
        Args:
            df: DataFrame containing simulation results
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Dictionary containing clustering analysis results
        """
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.empty or len(numeric_df) < 10:
            return {}
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        clustering_results = {
            'elbow_method': {},
            'silhouette_analysis': {},
            'optimal_clusters': {},
            'cluster_characteristics': {}
        }
        
        # Elbow method and silhouette analysis
        inertias = []
        silhouette_scores = []
        
        for k in range(2, min(max_clusters + 1, len(numeric_df))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                inertias.append(float(kmeans.inertia_))
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                sil_score = silhouette_score(scaled_data, cluster_labels)
                silhouette_scores.append(float(sil_score))
                
            except Exception as e:
                self.logger.warning(f"Clustering failed for k={k}: {str(e)}")
                continue
        
        clustering_results['elbow_method'] = {
            'k_values': list(range(2, 2 + len(inertias))),
            'inertias': inertias
        }
        
        clustering_results['silhouette_analysis'] = {
            'k_values': list(range(2, 2 + len(silhouette_scores))),
            'silhouette_scores': silhouette_scores
        }
        
        # Determine optimal number of clusters
        if silhouette_scores:
            optimal_k = 2 + np.argmax(silhouette_scores)
            clustering_results['optimal_clusters']['silhouette_method'] = int(optimal_k)
            
            # Perform final clustering with optimal k
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_labels = kmeans_final.fit_predict(scaled_data)
            
            # Analyze cluster characteristics
            cluster_chars = {}
            for cluster_id in range(optimal_k):
                cluster_mask = final_labels == cluster_id
                cluster_data = numeric_df[cluster_mask]
                
                cluster_chars[f'cluster_{cluster_id}'] = {
                    'size': int(cluster_mask.sum()),
                    'percentage': float(cluster_mask.sum() / len(numeric_df) * 100),
                    'centroid': cluster_data.mean().to_dict(),
                    'std': cluster_data.std().to_dict()
                }
            
            clustering_results['cluster_characteristics'] = cluster_chars
        
        return clustering_results
    
    def _calculate_confidence_intervals(self, data: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for mean and other statistics."""
        confidence_intervals = {}
        
        # Confidence interval for mean
        mean_ci = stats.t.interval(
            self.confidence_level, 
            len(data) - 1, 
            loc=data.mean(), 
            scale=stats.sem(data)
        )
        
        confidence_intervals['mean'] = {
            'lower': float(mean_ci[0]),
            'upper': float(mean_ci[1])
        }
        
        # Bootstrap confidence interval for median
        try:
            from scipy.stats import bootstrap
            
            def median_stat(x):
                return np.median(x, axis=1)
            
            bootstrap_result = bootstrap(
                (data.values,), 
                median_stat, 
                n_resamples=1000, 
                confidence_level=self.confidence_level,
                random_state=42
            )
            
            confidence_intervals['median'] = {
                'lower': float(bootstrap_result.confidence_interval.low),
                'upper': float(bootstrap_result.confidence_interval.high)
            }
        except:
            # Fallback to simple percentile method
            bootstrap_medians = []
            for _ in range(1000):
                sample = np.random.choice(data.values, size=len(data), replace=True)
                bootstrap_medians.append(np.median(sample))
            
            alpha = 1 - self.confidence_level
            lower = np.percentile(bootstrap_medians, 100 * alpha/2)
            upper = np.percentile(bootstrap_medians, 100 * (1 - alpha/2))
            
            confidence_intervals['median'] = {
                'lower': float(lower),
                'upper': float(upper)
            }
        
        return confidence_intervals
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test normality using multiple methods."""
        normality_tests = {}
        
        if len(data) < 8:
            return normality_tests
        
        # Shapiro-Wilk test
        if len(data) <= 5000:
            try:
                sw_stat, sw_p = stats.shapiro(data)
                normality_tests['shapiro_wilk'] = {
                    'statistic': float(sw_stat),
                    'p_value': float(sw_p),
                    'is_normal': sw_p > self.alpha
                }
            except:
                pass
        
        # Jarque-Bera test
        try:
            jb_stat, jb_p = jarque_bera(data)
            normality_tests['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'is_normal': jb_p > self.alpha
            }
        except:
            pass
        
        return normality_tests
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find pairs of variables with strong correlations."""
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) >= threshold:
                    strong_correlations.append({
                        'variable1': var1,
                        'variable2': var2,
                        'correlation': float(correlation),
                        'strength': 'strong' if abs(correlation) >= 0.8 else 'moderate'
                    })
        
        return strong_correlations
    
    def _test_correlation_significance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test statistical significance of correlations."""
        significance_results = {}
        n = len(df)
        
        corr_matrix = df.corr(method='pearson')
        
        for i, var1 in enumerate(corr_matrix.columns):
            for j, var2 in enumerate(corr_matrix.columns):
                if i >= j:  # Skip diagonal and duplicate pairs
                    continue
                
                r = corr_matrix.iloc[i, j]
                
                # Calculate t-statistic for correlation
                if abs(r) < 1:
                    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    t_stat = np.inf if r > 0 else -np.inf
                    p_value = 0
                
                pair_key = f"{var1}_vs_{var2}"
                significance_results[pair_key] = {
                    'correlation': float(r),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'is_significant': p_value < self.alpha,
                    'sample_size': n
                }
        
        return significance_results