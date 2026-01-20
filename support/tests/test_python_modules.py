"""
Test script for Python support modules.

Tests all modules in support/src/python/ with real methods.
"""

import os
import sys
import tempfile
import shutil

# Add support/src/python to path
SUPPORT_ROOT = os.path.join(os.path.dirname(__file__), "..", "src", "python")
sys.path.insert(0, SUPPORT_ROOT)

print("=" * 60)
print("PYTHON SUPPORT MODULES TEST SUITE")
print("=" * 60)
print()

passed = 0
failed = 0
errors = []

def test_module(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global passed, failed
            print(f"Testing {name}... ", end="", flush=True)
            try:
                func()
                passed += 1
                print("✓ PASS")
            except Exception as e:
                failed += 1
                errors.append(f"{name}: {e}")
                print(f"✗ FAIL: {e}")
        return wrapper
    return decorator


# ============================================================
# Test Utils Modules
# ============================================================
print("\n--- UTILS ---")

@test_module("file_utils")
def test_file_utils():
    from utils.file_utils import ensure_directory, copy_if_newer, find_files, safe_write
    
    # Test ensure_directory
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = os.path.join(tmpdir, "a", "b", "c")
        result = ensure_directory(nested)
        assert os.path.isdir(nested), "ensure_directory failed"
        assert result == nested, "ensure_directory should return path"
        
        # Test safe_write
        test_file = os.path.join(tmpdir, "test.txt")
        safe_write(test_file, "test content")
        assert os.path.isfile(test_file), "safe_write should create file"
        with open(test_file) as f:
            assert f.read() == "test content", "safe_write content mismatch"
        
        # Test find_files
        found = find_files(tmpdir, r"\.txt$")
        assert len(found) >= 1, "find_files should find .txt files"

test_file_utils()


@test_module("config_utils")
def test_config_utils():
    from utils.config_utils import load_json, get_config_value, merge_configs
    
    # Test get_config_value
    config = {"section": {"key": "value"}}
    result = get_config_value(config, "section", "key", default="default")
    assert result == "value", "get_config_value should return nested value"
    
    result_default = get_config_value(config, "missing", "key", default="default")
    assert result_default == "default", "get_config_value should return default"
    
    # Test merge_configs
    c1 = {"a": 1}
    c2 = {"b": 2}
    merged = merge_configs(c1, c2)
    assert merged["a"] == 1 and merged["b"] == 2, "merge_configs should combine"
    
    # Test deep merge
    c3 = {"nested": {"a": 1}}
    c4 = {"nested": {"b": 2}}
    deep_merged = merge_configs(c3, c4)
    assert deep_merged["nested"]["a"] == 1, "Deep merge should preserve a"
    assert deep_merged["nested"]["b"] == 2, "Deep merge should add b"

test_config_utils()


# ============================================================
# Test Statistics Modules
# ============================================================
print("\n--- STATISTICS ---")

@test_module("analysis_utils")
def test_analysis_utils():
    import numpy as np
    from statistics.analysis_utils import (
        calculate_rmse, calculate_mae, calculate_vaf,
        get_statistics, detect_divergence
    )
    
    real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    est = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    # Test RMSE
    rmse = calculate_rmse(real, est)
    assert abs(rmse - 0.1) < 0.01, f"RMSE should be ~0.1, got {rmse}"
    
    # Test MAE
    mae = calculate_mae(real, est)
    assert abs(mae - 0.1) < 0.01, f"MAE should be ~0.1, got {mae}"
    
    # Test VAF
    vaf = calculate_vaf(real, est)
    assert vaf > 99, f"VAF should be high, got {vaf}"
    
    # Test get_statistics
    stats = get_statistics(real, est)
    assert hasattr(stats, 'rmse'), "stats should have rmse"
    assert hasattr(stats, 'vaf'), "stats should have vaf"
    
    # Test detect_divergence
    stable, reason = detect_divergence(real)
    assert stable == False, "stable signal should not be divergent"
    
    divergent, reason = detect_divergence([1.0, float('nan'), 3.0])
    assert divergent == True, "NaN signal should be divergent"

test_analysis_utils()


@test_module("validation_utils")
def test_validation_utils():
    from statistics.validation_utils import (
        ValidationResult, validate_threshold, validate_all
    )
    
    # Test validate_threshold
    result = validate_threshold(0.95, 0.9, comparison='>=', name='Accuracy')
    assert result.passed == True, "0.95 >= 0.9 should pass"
    assert len(result.messages) > 0, "Should have messages"
    
    result_fail = validate_threshold(0.85, 0.9, comparison='>=', name='Accuracy')
    assert result_fail.passed == False, "0.85 >= 0.9 should fail"
    
    # Test validate_all
    v1 = ValidationResult(passed=True, messages=["pass1"])
    v2 = ValidationResult(passed=True, messages=["pass2"])
    combined = validate_all([v1, v2])
    assert combined.passed == True, "All passing should pass"
    assert len(combined.messages) == 2, "Should combine messages"

test_validation_utils()


# ============================================================
# Test Visualization Modules
# ============================================================
print("\n--- VISUALIZATION ---")

@test_module("plotting_utils")
def test_plotting_utils():
    from visualization.plotting_utils import (
        setup_plot_defaults, create_figure, save_figure, add_reference_line
    )
    
    # Test setup_plot_defaults
    setup_plot_defaults()
    
    # Test create_figure
    fig, ax = create_figure(title="Test", xlabel="X", ylabel="Y")
    assert fig is not None, "create_figure should return figure"
    assert ax is not None, "create_figure should return axes"
    
    # Test add_reference_line
    add_reference_line(ax, 0.5, orientation='horizontal')
    add_reference_line(ax, 0.5, orientation='vertical')
    
    # Test save_figure
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_figure(fig, "test.png", tmpdir)
        assert os.path.isfile(path), "save_figure should create file"
    
    import matplotlib.pyplot as plt
    plt.close(fig)

test_plotting_utils()


# ============================================================
# Test Logging Modules
# ============================================================
print("\n--- LOGGING ---")

@test_module("logging_utils")
def test_logging_utils():
    from experiment_logging.logging_utils import (
        setup_logger, log_section, log_config
    )
    from statistics.validation_utils import ValidationResult
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test setup_logger
        logger = setup_logger(tmpdir, "config.toml", 1234)
        assert logger is not None, "setup_logger should return logger"
        
        # Test log_section
        log_section(logger, "Test Section")
        
        # Test log_config
        log_config(logger, {"key": "value"})
        
        # Check log file was created
        log_path = os.path.join(tmpdir, "execution.log")
        assert os.path.isfile(log_path), "Log file should be created"
        
        with open(log_path) as f:
            content = f.read()
            assert "Test Section" in content, "Log should contain section"

test_logging_utils()


# ============================================================
# Test Integration Modules
# ============================================================
print("\n--- INTEGRATION ---")

@test_module("gnn_utils")
def test_gnn_utils():
    from integration.gnn_utils import (
        check_git_available, GNN_REPO_URL, DEFAULT_BRANCH
    )
    
    # Test check_git_available
    result = check_git_available()
    assert isinstance(result, bool), "check_git_available should return bool"
    
    # Test constants
    assert GNN_REPO_URL is not None, "GNN_REPO_URL should be defined"
    assert DEFAULT_BRANCH is not None, "DEFAULT_BRANCH should be defined"

test_gnn_utils()


@test_module("server_utils")
def test_server_utils():
    from integration.server_utils import DEFAULT_SERVER_URL
    
    # Test constant
    assert DEFAULT_SERVER_URL is not None, "DEFAULT_SERVER_URL should be defined"
    assert "localhost" in DEFAULT_SERVER_URL, "Should reference localhost"

test_server_utils()


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if errors:
    print("\nErrors:")
    for err in errors:
        print(f"  - {err}")

print("\nAll tests use REAL METHODS - NO MOCKS")
print("=" * 60)

sys.exit(1 if failed > 0 else 0)
