module Analysis

using Statistics
using LinearAlgebra

export calculate_rmse, calculate_mae, calculate_vaf, calculate_max_error, detect_divergence, check_convergence, generate_statistics_report, get_statistics

"""
    calculate_rmse(real, estimated)

Calculate Root Mean Square Error between real and estimated signals.
"""
function calculate_rmse(real, estimated)
    return sqrt(mean((real .- estimated).^2))
end

"""
    calculate_mae(real, estimated)

Calculate Mean Absolute Error between real and estimated signals.
"""
function calculate_mae(real, estimated)
    return mean(abs.(real .- estimated))
end

"""
    check_convergence(free_energy_history; threshold=1e-3, window=5)

Check if the Free Energy has converged by looking at the change over the last `window` iterations.
"""
function check_convergence(free_energy_history; threshold=1e-3, window=5)
    if length(free_energy_history) < window
        return false, abs(free_energy_history[end] - free_energy_history[1])
    end
    
    recent_changes = diff(free_energy_history[end-window+1:end])
    max_change = maximum(abs.(recent_changes))
    
    return max_change < threshold, max_change
end

"""
    calculate_vaf(real, estimated)

Calculate Variance Accounted For (%) between real and estimated signals.
100% is perfect match.
"""
function calculate_vaf(real, estimated)
    v_real = var(real)
    v_diff = var(real .- estimated)
    return 100 * (1 - v_diff / (v_real + eps()))
end

"""
    calculate_max_error(real, estimated)

Calculate Maximum Absolute Error.
"""
function calculate_max_error(real, estimated)
    return maximum(abs.(real .- estimated))
end

"""
    detect_divergence(signal; threshold=1e6)

Check for numerical instability (NaN, Inf, or excessively large values).
Returns (is_divergent::Bool, reason::String).
"""
function detect_divergence(signal; threshold=1e6)
    if any(isnan, signal)
        return true, "Contains NaN"
    end
    if any(isinf, signal)
        return true, "Contains Inf"
    end
    max_val = maximum(abs.(filter(!isnan, signal))) # Filter NaN just in case, though checked above
    if max_val > threshold
        return true, "Exceeds threshold ($max_val > $threshold)"
    end
    return false, "Stable"
end

"""
    get_statistics(real, estimated)

Calculate and return a NamedTuple of statistics: (rmse, mae, vaf, max_err).
"""
function get_statistics(real, estimated)
    rmse = calculate_rmse(real, estimated)
    mae = calculate_mae(real, estimated)
    vaf = calculate_vaf(real, estimated)
    max_err = calculate_max_error(real, estimated)
    return (; rmse, mae, vaf, max_err)
end

"""
    generate_statistics_report(real, estimated, name)

Generate a string report of statistical metrics.
"""
function generate_statistics_report(real, estimated, name)
    rmse, mae = get_statistics(real, estimated)
    
    return """
    --- Statistics for $name ---
    RMSE: $rmse
    MAE:  $mae
    ---------------------------
    """
end

"""
    struct ValidationResult
        passed::Bool
        messages::Vector{String}
    end

Container for validation results.
"""
struct ValidationResult
    passed::Bool
    messages::Vector{String}
end

"""
    validate_metrics(stats, config)

Validate statistics against configuration thresholds.
checks `validation_min_vaf` and `validation_max_rmse`.
"""
function validate_metrics(stats, config)
    min_vaf = get(config, "validation_min_vaf", -Inf)
    max_rmse = get(config, "validation_max_rmse", Inf)
    
    passed = true
    messages = String[]
    
    # Check VAF
    if stats.vaf < min_vaf
        push!(messages, "FAIL: VAF $(round(stats.vaf, digits=2))% < Threshold $min_vaf%")
        passed = false
    else
        push!(messages, "PASS: VAF $(round(stats.vaf, digits=2))% >= $min_vaf%")
    end
    
    # Check RMSE
    if stats.rmse > max_rmse
        push!(messages, "FAIL: RMSE $(round(stats.rmse, digits=4)) > Threshold $max_rmse")
        passed = false
    else
        push!(messages, "PASS: RMSE $(round(stats.rmse, digits=4)) <= $max_rmse")
    end
    
    return ValidationResult(passed, messages)
end

export ValidationResult, validate_metrics

end # module
