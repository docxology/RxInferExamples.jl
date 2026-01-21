"""
Validation - Validation framework for multinomial regression.

Provides structured validation with PASS/FAIL status.
"""
module Validation

using Statistics
using Printf

export ValidationResult, validate_classification, validate_regression
export validate_all, generate_validation_report

"""
    struct ValidationResult

Container for validation results.
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
    validate_classification(mse, threshold; name="Classification") -> ValidationResult

Validate classification MSE is below threshold.
"""
function validate_classification(mse::Float64, threshold::Float64; 
                                 name::String="Probability MSE")
    passed = mse <= threshold
    message = @sprintf("%s: %.6f %s %.6f → %s",
        name, mse, passed ? "≤" : ">", threshold, passed ? "PASS" : "FAIL")
    
    return ValidationResult(passed, name, mse, threshold, message)
end

"""
    validate_regression(mse, threshold; name="Regression") -> ValidationResult

Validate regression coefficient MSE is below threshold.
"""
function validate_regression(mse::Float64, threshold::Float64;
                            name::String="Coefficient MSE")
    passed = mse <= threshold
    message = @sprintf("%s: %.6f %s %.6f → %s",
        name, mse, passed ? "≤" : ">", threshold, passed ? "PASS" : "FAIL")
    
    return ValidationResult(passed, name, mse, threshold, message)
end

"""
    validate_all(results::Vector{ValidationResult}) -> ValidationResult

Combine multiple validation results.
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
    generate_validation_report(results, output_path)

Generate validation report file.
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

end # module
