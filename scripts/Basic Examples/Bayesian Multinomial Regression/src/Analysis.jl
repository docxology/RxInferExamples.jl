"""
Analysis - Statistical analysis for multinomial regression results.

Provides analysis, reporting, and summary functions.
"""
module Analysis

using Statistics
using LinearAlgebra
using Printf
using JSON

export compute_mse, analyze_classification_result, analyze_regression_result
export generate_analysis_report, generate_json_summary

"""
    compute_mse(estimated, true_values) -> Float64

Compute mean squared error.
"""
function compute_mse(estimated::AbstractVector, true_values::AbstractVector)
    return mean((estimated .- true_values).^2)
end

"""
    analyze_classification_result(result, true_probs, config) -> NamedTuple

Analyze multinomial classification inference result.
"""
function analyze_classification_result(estimated_probs::AbstractVector, 
                                       true_probs::AbstractVector,
                                       free_energy::AbstractVector)
    mse = compute_mse(estimated_probs, true_probs)
    max_error = maximum(abs.(estimated_probs .- true_probs))
    correlation = cor(estimated_probs, true_probs)
    
    # Convergence analysis
    fe_final = free_energy[end]
    fe_change = length(free_energy) > 1 ? abs(free_energy[end] - free_energy[end-1]) : 0.0
    
    return (
        mse = mse,
        max_error = max_error,
        correlation = correlation,
        free_energy_final = fe_final,
        free_energy_change = fe_change,
        n_iterations = length(free_energy)
    )
end

"""
    analyze_regression_result(estimated_beta, true_beta, free_energy) -> NamedTuple

Analyze multinomial regression inference result.
"""
function analyze_regression_result(estimated_mean::AbstractVector,
                                   estimated_cov::AbstractMatrix,
                                   true_beta::AbstractVector,
                                   free_energy::AbstractVector)
    mse = compute_mse(estimated_mean, true_beta)
    max_error = maximum(abs.(estimated_mean .- true_beta))
    
    # Uncertainty analysis
    std_beta = sqrt.(diag(estimated_cov))
    mean_std = mean(std_beta)
    
    # Check if true values within 2σ
    within_2sigma = sum(abs.(estimated_mean .- true_beta) .< 2*std_beta) / length(true_beta)
    
    return (
        mse = mse,
        max_error = max_error,
        mean_std = mean_std,
        coverage_2sigma = within_2sigma,
        free_energy_final = free_energy[end],
        n_iterations = length(free_energy)
    )
end

"""
    generate_analysis_report(cls_analysis, reg_analysis, validation_results, output_path)

Generate comprehensive text report.
"""
function generate_analysis_report(cls_analysis, reg_analysis, 
                                  validation_results::Vector,
                                  output_path::String)
    open(output_path, "w") do io
        println(io, "="^70)
        println(io, "BAYESIAN MULTINOMIAL REGRESSION ANALYSIS REPORT")
        println(io, "="^70)
        println(io)
        
        # Classification results
        println(io, "CLASSIFICATION EXPERIMENT")
        println(io, "-"^40)
        println(io, @sprintf("  Probability MSE: %.6f", cls_analysis.mse))
        println(io, @sprintf("  Max Error: %.4f", cls_analysis.max_error))
        println(io, @sprintf("  Correlation: %.4f", cls_analysis.correlation))
        println(io, @sprintf("  Final Free Energy: %.2f", cls_analysis.free_energy_final))
        println(io, @sprintf("  Iterations: %d", cls_analysis.n_iterations))
        println(io)
        
        # Regression results
        println(io, "REGRESSION EXPERIMENT")
        println(io, "-"^40)
        println(io, @sprintf("  Coefficient MSE: %.6f", reg_analysis.mse))
        println(io, @sprintf("  Max Error: %.4f", reg_analysis.max_error))
        println(io, @sprintf("  Mean Std: %.4f", reg_analysis.mean_std))
        println(io, @sprintf("  2σ Coverage: %.1f%%", reg_analysis.coverage_2sigma * 100))
        println(io, @sprintf("  Final Free Energy: %.2f", reg_analysis.free_energy_final))
        println(io)
        
        # Validation
        println(io, "VALIDATION RESULTS")
        println(io, "-"^40)
        n_passed = count(r -> r.passed, validation_results)
        n_total = length(validation_results)
        println(io, @sprintf("  Passed: %d/%d", n_passed, n_total))
        for result in validation_results
            status = result.passed ? "[PASS]" : "[FAIL]"
            println(io, "  $status $(result.message)")
        end
        println(io)
        
        println(io, "="^70)
    end
    
    @info "Report saved: $output_path"
end

"""
    generate_json_summary(cls_analysis, reg_analysis, validation, config, output_path)

Generate machine-readable JSON summary.
"""
function generate_json_summary(cls_analysis, reg_analysis,
                               validation_results::Vector,
                               config::Dict,
                               output_path::String)
    summary = Dict(
        "config" => Dict(
            "seed" => config["general"]["seed"],
            "classification_samples" => config["classification"]["nsamples"],
            "regression_samples" => config["regression"]["nsamples"]
        ),
        "classification" => Dict(
            "mse" => cls_analysis.mse,
            "max_error" => cls_analysis.max_error,
            "correlation" => cls_analysis.correlation,
            "iterations" => cls_analysis.n_iterations
        ),
        "regression" => Dict(
            "mse" => reg_analysis.mse,
            "max_error" => reg_analysis.max_error,
            "coverage_2sigma" => reg_analysis.coverage_2sigma
        ),
        "validation" => Dict(
            "passed" => count(r -> r.passed, validation_results),
            "total" => length(validation_results),
            "all_passed" => all(r -> r.passed, validation_results)
        )
    )
    
    open(output_path, "w") do io
        JSON.print(io, summary, 2)
    end
    
    @info "JSON summary saved: $output_path"
end

end # module
