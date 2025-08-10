"""Performance optimization utilities for hypersonic reentry simulation.

This module provides tools for:
- Memory and CPU profiling
- Parallel processing optimization
- Computational performance monitoring
- Resource usage tracking
"""

import numpy as np
import time
import psutil
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable
import logging
from functools import wraps, lru_cache
from dataclasses import dataclass
import sys
from pathlib import Path
import json
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    execution_time: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    parallel_efficiency: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    throughput: Optional[float] = None


class PerformanceProfiler:
    """Performance profiling and monitoring system.
    
    Provides comprehensive performance analysis including timing,
    memory usage, CPU utilization, and parallel processing efficiency.
    """
    
    def __init__(self, enable_profiling: bool = True):
        """Initialize performance profiler.
        
        Args:
            enable_profiling: Whether to enable performance monitoring
        """
        self.logger = logging.getLogger(__name__)
        self.enable_profiling = enable_profiling
        self.profile_data = {}
        self.start_times = {}
        self.memory_baseline = self._get_memory_usage()
        
        if enable_profiling:
            self.logger.info("Performance profiling enabled")
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator for automatic function profiling.
        
        Args:
            func_name: Optional name for the function (default: actual function name)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_profiling:
                    return func(*args, **kwargs)
                
                name = func_name or func.__name__
                
                # Start profiling
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                start_cpu = psutil.cpu_percent(interval=None)
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Record metrics
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()
                    end_cpu = psutil.cpu_percent(interval=None)
                    
                    metrics = PerformanceMetrics(
                        execution_time=end_time - start_time,
                        memory_usage={
                            'peak_mb': end_memory['peak_mb'],
                            'current_mb': end_memory['current_mb'],
                            'delta_mb': end_memory['current_mb'] - start_memory['current_mb']
                        },
                        cpu_usage=max(end_cpu - start_cpu, 0)
                    )
                    
                    self.profile_data[name] = metrics
                    
                    if metrics.execution_time > 1.0:  # Log significant operations
                        self.logger.info(f"{name}: {metrics.execution_time:.2f}s, "
                                       f"Memory: +{metrics.memory_usage['delta_mb']:.1f}MB, "
                                       f"CPU: {metrics.cpu_usage:.1f}%")
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Function {name} failed: {str(e)}")
                    raise
                    
            return wrapper
        return decorator
    
    def start_timer(self, operation_name: str) -> None:
        """Start timing an operation.
        
        Args:
            operation_name: Name of the operation to time
        """
        if self.enable_profiling:
            self.start_times[operation_name] = {
                'time': time.perf_counter(),
                'memory': self._get_memory_usage(),
                'cpu': psutil.cpu_percent(interval=None)
            }
    
    def end_timer(self, operation_name: str) -> PerformanceMetrics:
        """End timing an operation and return metrics.
        
        Args:
            operation_name: Name of the operation that was timed
            
        Returns:
            PerformanceMetrics containing timing and resource usage data
        """
        if not self.enable_profiling or operation_name not in self.start_times:
            return PerformanceMetrics(0, {}, 0)
        
        start_data = self.start_times.pop(operation_name)
        
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        end_cpu = psutil.cpu_percent(interval=None)
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_data['time'],
            memory_usage={
                'peak_mb': end_memory['peak_mb'],
                'current_mb': end_memory['current_mb'],
                'delta_mb': end_memory['current_mb'] - start_data['memory']['current_mb']
            },
            cpu_usage=max(end_cpu - start_data['cpu'], 0)
        )
        
        self.profile_data[operation_name] = metrics
        
        return metrics
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'current_mb': memory_info.rss / 1024 / 1024,
            'peak_mb': memory_info.peak_wset / 1024 / 1024 if hasattr(memory_info, 'peak_wset') else memory_info.rss / 1024 / 1024
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance analysis
        """
        if not self.profile_data:
            return {}
        
        summary = {
            'total_operations': len(self.profile_data),
            'total_time': sum(m.execution_time for m in self.profile_data.values()),
            'memory_statistics': {},
            'timing_statistics': {},
            'top_operations': {}
        }
        
        # Timing statistics
        times = [m.execution_time for m in self.profile_data.values()]
        if times:
            summary['timing_statistics'] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_time': np.sum(times)
            }
        
        # Memory statistics
        memory_deltas = [m.memory_usage.get('delta_mb', 0) for m in self.profile_data.values()]
        if memory_deltas:
            summary['memory_statistics'] = {
                'mean_delta_mb': np.mean(memory_deltas),
                'std_delta_mb': np.std(memory_deltas),
                'max_delta_mb': np.max(memory_deltas),
                'total_delta_mb': np.sum(memory_deltas)
            }
        
        # Top operations by time
        sorted_ops = sorted(self.profile_data.items(), 
                           key=lambda x: x[1].execution_time, reverse=True)
        
        summary['top_operations'] = {
            'by_time': [(name, metrics.execution_time) for name, metrics in sorted_ops[:10]],
            'by_memory': sorted([(name, metrics.memory_usage.get('delta_mb', 0)) 
                               for name, metrics in self.profile_data.items()], 
                              key=lambda x: x[1], reverse=True)[:10]
        }
        
        return summary
    
    def save_profile_report(self, filepath: str) -> None:
        """Save performance profile report to file.
        
        Args:
            filepath: Path to save the report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'performance_summary': self.get_performance_summary(),
            'detailed_metrics': {
                name: {
                    'execution_time': metrics.execution_time,
                    'memory_usage': metrics.memory_usage,
                    'cpu_usage': metrics.cpu_usage
                }
                for name, metrics in self.profile_data.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved: {filepath}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for performance context."""
        return {
            'cpu_count': mp.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }


class ParallelProcessor:
    """Optimized parallel processing for Monte Carlo simulations.
    
    Provides efficient parallel execution with load balancing,
    memory management, and progress monitoring.
    """
    
    def __init__(self, 
                 num_workers: Optional[int] = None,
                 chunk_size: Optional[int] = None,
                 memory_limit_gb: float = 8.0):
        """Initialize parallel processor.
        
        Args:
            num_workers: Number of worker processes (default: CPU count)
            chunk_size: Size of work chunks (default: auto-calculated)
            memory_limit_gb: Memory limit per worker in GB
        """
        self.logger = logging.getLogger(__name__)
        self.num_workers = num_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        
        # Adjust workers based on memory constraints
        available_memory = psutil.virtual_memory().available / (1024**3)
        max_workers_by_memory = int(available_memory / memory_limit_gb)
        
        if max_workers_by_memory < self.num_workers:
            self.num_workers = max(1, max_workers_by_memory)
            self.logger.warning(f"Reduced workers to {self.num_workers} due to memory constraints")
        
        self.logger.info(f"Initialized parallel processor with {self.num_workers} workers")
    
    def parallel_map(self, 
                    function: Callable,
                    iterable: List[Any],
                    progress_callback: Optional[Callable] = None) -> List[Any]:
        """Execute function on iterable in parallel with optimization.
        
        Args:
            function: Function to execute
            iterable: Iterable of arguments
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results
        """
        if len(iterable) == 0:
            return []
        
        # Calculate optimal chunk size
        if self.chunk_size is None:
            chunk_size = max(1, len(iterable) // (self.num_workers * 4))
        else:
            chunk_size = self.chunk_size
        
        # Create chunks
        chunks = [iterable[i:i + chunk_size] 
                 for i in range(0, len(iterable), chunk_size)]
        
        self.logger.info(f"Processing {len(iterable)} items in {len(chunks)} chunks "
                        f"using {self.num_workers} workers")
        
        def process_chunk(chunk):
            """Process a chunk of work items."""
            results = []
            for item in chunk:
                try:
                    result = function(item)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Item processing failed: {str(e)}")
                    results.append(None)
            return results
        
        # Execute in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        all_results = []
        completed_items = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    completed_items += len(chunk_results)
                    
                    # Progress callback
                    if progress_callback:
                        progress = completed_items / len(iterable)
                        progress_callback(progress)
                    
                except Exception as e:
                    self.logger.error(f"Chunk {chunk_idx} failed: {str(e)}")
                    # Add None results for failed chunk
                    chunk_size_actual = len(chunks[chunk_idx])
                    all_results.extend([None] * chunk_size_actual)
        
        self.logger.info(f"Parallel processing completed: {len(all_results)} results")
        
        return all_results
    
    def estimate_parallel_efficiency(self, 
                                   sequential_time: float,
                                   parallel_time: float) -> float:
        """Estimate parallel processing efficiency.
        
        Args:
            sequential_time: Time for sequential execution
            parallel_time: Time for parallel execution
            
        Returns:
            Parallel efficiency (0-1, higher is better)
        """
        if parallel_time <= 0:
            return 0.0
        
        theoretical_speedup = self.num_workers
        actual_speedup = sequential_time / parallel_time
        efficiency = actual_speedup / theoretical_speedup
        
        return min(1.0, efficiency)


class CacheManager:
    """Intelligent caching system for expensive computations.
    
    Provides LRU caching with memory management and cache statistics.
    """
    
    def __init__(self, max_cache_size: int = 1000):
        """Initialize cache manager.
        
        Args:
            max_cache_size: Maximum number of cached items
        """
        self.logger = logging.getLogger(__name__)
        self.max_cache_size = max_cache_size
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
    def cached_function(self, maxsize: int = 128):
        """Decorator for caching function results.
        
        Args:
            maxsize: Maximum cache size for this function
        """
        def decorator(func):
            cached_func = lru_cache(maxsize=maxsize)(func)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get cache info before call
                cache_info_before = cached_func.cache_info()
                
                # Execute function (potentially cached)
                result = cached_func(*args, **kwargs)
                
                # Update cache statistics
                cache_info_after = cached_func.cache_info()
                
                if cache_info_after.hits > cache_info_before.hits:
                    self.cache_stats['hits'] += 1
                else:
                    self.cache_stats['misses'] += 1
                
                return result
            
            # Add cache management methods
            wrapper.cache_info = cached_func.cache_info
            wrapper.cache_clear = cached_func.cache_clear
            
            return wrapper
        
        return decorator
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Dictionary containing cache performance metrics
        """
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        
        if total_requests == 0:
            hit_rate = 0.0
        else:
            hit_rate = self.cache_stats['hits'] / total_requests
        
        return {
            'hit_rate': hit_rate,
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'total_requests': total_requests,
            'evictions': self.cache_stats['evictions']
        }


class MemoryOptimizer:
    """Memory usage optimization utilities.
    
    Provides tools for monitoring and optimizing memory usage
    during large-scale simulations.
    """
    
    def __init__(self, memory_threshold_gb: float = 6.0):
        """Initialize memory optimizer.
        
        Args:
            memory_threshold_gb: Memory usage threshold for warnings
        """
        self.logger = logging.getLogger(__name__)
        self.memory_threshold_bytes = memory_threshold_gb * 1024**3
        
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        process = psutil.Process()
        system_memory = psutil.virtual_memory()
        
        memory_stats = {
            'process_memory_gb': process.memory_info().rss / (1024**3),
            'system_memory_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_percent': system_memory.percent
        }
        
        if process.memory_info().rss > self.memory_threshold_bytes:
            self.logger.warning(f"High memory usage: {memory_stats['process_memory_gb']:.2f} GB")
        
        return memory_stats
    
    def optimize_array_operations(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize array operations for memory efficiency.
        
        Args:
            arrays: List of NumPy arrays to optimize
            
        Returns:
            List of optimized arrays
        """
        optimized = []
        
        for arr in arrays:
            # Convert to appropriate dtype if possible
            if arr.dtype == np.float64:
                # Check if we can use float32 without significant precision loss
                if np.allclose(arr, arr.astype(np.float32), rtol=1e-6):
                    arr = arr.astype(np.float32)
                    self.logger.debug("Converted array to float32 for memory efficiency")
            
            # Ensure arrays are contiguous for better performance
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            
            optimized.append(arr)
        
        return optimized
    
    def memory_efficient_operations(self, 
                                  operation: Callable,
                                  data: np.ndarray,
                                  chunk_size: Optional[int] = None) -> np.ndarray:
        """Perform operations on large arrays in memory-efficient chunks.
        
        Args:
            operation: Function to apply to array chunks
            data: Input array
            chunk_size: Size of chunks to process
            
        Returns:
            Result of operation applied to entire array
        """
        if chunk_size is None:
            # Estimate chunk size based on available memory
            available_memory = psutil.virtual_memory().available
            element_size = data.dtype.itemsize
            chunk_size = min(len(data), int(available_memory * 0.1 / element_size))
        
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_result = operation(chunk)
            results.append(chunk_result)
        
        return np.concatenate(results) if results else np.array([])


# Global profiler instance
profiler = PerformanceProfiler()

# Convenience decorators
def profile(func_name: Optional[str] = None):
    """Convenience decorator for performance profiling."""
    return profiler.profile_function(func_name)


def timed_operation(operation_name: str):
    """Context manager for timing operations."""
    class TimedOperation:
        def __enter__(self):
            profiler.start_timer(operation_name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            metrics = profiler.end_timer(operation_name)
            return False
    
    return TimedOperation()