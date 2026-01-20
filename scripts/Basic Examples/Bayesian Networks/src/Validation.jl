"""
Validation - Validation framework for Bayesian Networks experiments.

Provides structured validation of inference and learning results.
"""
module Validation

using Statistics
using Printf

export ValidationResult, validate_cpt_accuracy, validate_posterior
export validate_all, generate_validation_report

"""
    struct ValidationResult

Container for validation results with pass/fail status.
"""
struct ValidationResult
    passed::Bool
    metric_name::String
    actual_value::Float64
    threshold::Float64
    message::String
end

function ValidationResult(passed::Bool, message::String)
    ValidationResult(passed, "", 0.0, 0.0, message)
end

"""
    validate_cpt_accuracy(learned_cpt, true_cpt, threshold; name="CPT")

Validate that learned CPT matches true CPT within threshold.
"""
function validate_cpt_accuracy(learned_cpt::AbstractArray, true_cpt::AbstractArray, 
                                threshold::Float64; name::String="CPT")
    max_error = maximum(abs.(learned_cpt .- true_cpt))
    mean_error = mean(abs.(learned_cpt .- true_cpt))
    
    passed = max_error <= threshold
    message = @sprintf("%s: max_error=%.4f, mean_error=%.4f, threshold=%.4f → %s",
        name, max_error, mean_error, threshold, passed ? "PASS" : "FAIL")
    
    return ValidationResult(passed, name, max_error, threshold, message)
end

"""
    validate_posterior(posterior, expected_state::Int, threshold; name="Posterior")

Validate that posterior has sufficient confidence in expected state.
"""
function validate_posterior(posterior, expected_state::Int, threshold::Float64; name::String="Posterior")
    probs = posterior.p
    actual_confidence = probs[expected_state]
    
    passed = actual_confidence >= threshold
    message = @sprintf("%s: P(state=%d)=%.4f, threshold=%.4f → %s",
        name, expected_state, actual_confidence, threshold, passed ? "PASS" : "FAIL")
    
    return ValidationResult(passed, name, actual_confidence, threshold, message)
end

"""
    validate_all(results::Vector{ValidationResult}) -> ValidationResult

Combine multiple validation results. Overall passes only if all pass.
"""
function validate_all(results::Vector{ValidationResult})
    all_passed = all(r -> r.passed, results)
    n_passed = count(r -> r.passed, results)
    n_total = length(results)
    
    message = @sprintf("Validation Summary: %d/%d passed → %s", 
        n_passed, n_total, all_passed ? "OVERALL PASS" : "OVERALL FAIL")
    
    return ValidationResult(all_passed, message)
end

"""
    generate_validation_report(results::Vector{ValidationResult}, output_path::String)

Generate a validation report as a text file.
"""
function generate_validation_report(results::Vector{ValidationResult}, output_path::String)
    overall = validate_all(results)
    
    open(output_path, "w") do io
        println(io, "="^60)
        println(io, "VALIDATION REPORT")
        println(io, "="^60)
        println(io)
        
        for result in results
            status = result.passed ? "[PASS]" : "[FAIL]"
            println(io, "$status $(result.message)")
        end
        
        println(io)
        println(io, "-"^60)
        status = overall.passed ? "[PASS]" : "[FAIL]"
        println(io, "$status $(overall.message)")
        println(io, "="^60)
    end
    
    @info "Validation report saved: $output_path"
    return overall
end

"""
    compute_cpt_errors(learned_cpt, true_cpt) -> NamedTuple

Compute detailed error statistics between learned and true CPT.
"""
function compute_cpt_errors(learned_cpt::AbstractArray, true_cpt::AbstractArray)
    errors = abs.(learned_cpt .- true_cpt)
    return (
        max_error = maximum(errors),
        mean_error = mean(errors),
        min_error = minimum(errors),
        std_error = std(errors),
        rmse = sqrt(mean(errors.^2))
    )
end

export compute_cpt_errors

end # module
