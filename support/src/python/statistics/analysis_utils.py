"""
analysis_utils - Statistical analysis utilities.

Provides common statistical metrics for signal analysis.
"""

import numpy as np
from typing import Tuple, NamedTuple, Union
from dataclasses import dataclass


@dataclass
class Statistics:
    """Container for statistical metrics."""
    rmse: float
    mae: float
    vaf: float
    max_err: float


def calculate_rmse(real: np.ndarray, estimated: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error between real and estimated signals.
    
    Args:
        real: Ground truth values
        estimated: Estimated values
        
    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((np.array(real) - np.array(estimated)) ** 2)))


def calculate_mae(real: np.ndarray, estimated: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error between real and estimated signals.
    
    Args:
        real: Ground truth values
        estimated: Estimated values
        
    Returns:
        MAE value
    """
    return float(np.mean(np.abs(np.array(real) - np.array(estimated))))


def calculate_vaf(real: np.ndarray, estimated: np.ndarray) -> float:
    """
    Calculate Variance Accounted For (%) between real and estimated signals.
    
    100% is perfect match.
    
    Args:
        real: Ground truth values
        estimated: Estimated values
        
    Returns:
        VAF percentage
    """
    real = np.array(real)
    estimated = np.array(estimated)
    v_real = np.var(real)
    v_diff = np.var(real - estimated)
    return float(100 * (1 - v_diff / (v_real + 1e-10)))


def calculate_max_error(real: np.ndarray, estimated: np.ndarray) -> float:
    """
    Calculate Maximum Absolute Error.
    
    Args:
        real: Ground truth values
        estimated: Estimated values
        
    Returns:
        Maximum absolute error
    """
    return float(np.max(np.abs(np.array(real) - np.array(estimated))))


def get_statistics(real: np.ndarray, estimated: np.ndarray) -> Statistics:
    """
    Calculate and return all statistics.
    
    Args:
        real: Ground truth values
        estimated: Estimated values
        
    Returns:
        Statistics dataclass with rmse, mae, vaf, max_err
    """
    return Statistics(
        rmse=calculate_rmse(real, estimated),
        mae=calculate_mae(real, estimated),
        vaf=calculate_vaf(real, estimated),
        max_err=calculate_max_error(real, estimated)
    )


def detect_divergence(signal: np.ndarray, threshold: float = 1e6) -> Tuple[bool, str]:
    """
    Check for numerical instability (NaN, Inf, or excessively large values).
    
    Args:
        signal: Signal to check
        threshold: Maximum allowed absolute value
        
    Returns:
        Tuple of (is_divergent, reason)
    """
    signal = np.array(signal)
    
    if np.any(np.isnan(signal)):
        return True, "Contains NaN"
    
    if np.any(np.isinf(signal)):
        return True, "Contains Inf"
    
    max_val = np.max(np.abs(signal[~np.isnan(signal)]))
    if max_val > threshold:
        return True, f"Exceeds threshold ({max_val} > {threshold})"
    
    return False, "Stable"
