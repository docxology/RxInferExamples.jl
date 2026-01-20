"""
validation_utils - Validation framework utilities.

Provides structured validation for experiment results.
"""

from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
import operator


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    messages: List[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.passed
    
    @classmethod
    def success(cls, message: str) -> 'ValidationResult':
        """Create a successful validation result."""
        return cls(passed=True, messages=[message])
    
    @classmethod
    def failure(cls, message: str) -> 'ValidationResult':
        """Create a failed validation result."""
        return cls(passed=False, messages=[message])


COMPARISON_OPS = {
    '>=': operator.ge,
    '<=': operator.le,
    '>': operator.gt,
    '<': operator.lt,
    '==': operator.eq,
}


def validate_threshold(
    value: float,
    threshold: float,
    comparison: str = '>=',
    name: str = 'Value'
) -> ValidationResult:
    """
    Validate a value against a threshold.
    
    Args:
        value: The value to check
        threshold: The threshold to compare against
        comparison: Comparison operator ('>=', '<=', '>', '<', '==')
        name: Name for the validation message
        
    Returns:
        ValidationResult
    """
    if comparison not in COMPARISON_OPS:
        raise ValueError(f"Unknown comparison operator: {comparison}")
    
    op = COMPARISON_OPS[comparison]
    result = op(value, threshold)
    
    status = 'PASS' if result else 'FAIL'
    message = f"{status}: {name} = {value:.4f} {comparison} {threshold}"
    
    return ValidationResult(passed=result, messages=[message])


def validate_metrics(
    stats: Any,
    config: Dict[str, Any]
) -> ValidationResult:
    """
    Validate statistics against configuration thresholds.
    
    Checks validation_min_vaf and validation_max_rmse from config.
    
    Args:
        stats: Statistics object with .vaf and .rmse attributes
        config: Configuration dictionary
        
    Returns:
        ValidationResult
    """
    min_vaf = config.get('validation_min_vaf', float('-inf'))
    max_rmse = config.get('validation_max_rmse', float('inf'))
    
    validations = [
        validate_threshold(stats.vaf, min_vaf, '>=', 'VAF (%)'),
        validate_threshold(stats.rmse, max_rmse, '<=', 'RMSE')
    ]
    
    return validate_all(validations)


def validate_all(validations: List[ValidationResult]) -> ValidationResult:
    """
    Combine multiple validation results.
    
    Overall passes only if all pass.
    
    Args:
        validations: List of ValidationResult objects
        
    Returns:
        Combined ValidationResult
    """
    all_passed = all(v.passed for v in validations)
    all_messages = []
    for v in validations:
        all_messages.extend(v.messages)
    
    return ValidationResult(passed=all_passed, messages=all_messages)
