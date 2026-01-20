"""
ValidationUtils - Validation framework utilities.

Provides structured validation for experiment results.

# Exports
- `ValidationResult`: Struct for validation outcomes
- `validate_threshold`: Check value against threshold
- `validate_all`: Run multiple validations
"""
module ValidationUtils

export ValidationResult, validate_threshold, validate_all

"""
    struct ValidationResult

Container for validation results.

# Fields
- `passed::Bool`: Whether validation passed
- `messages::Vector{String}`: Validation messages
"""
struct ValidationResult
    passed::Bool
    messages::Vector{String}
end

"""
    ValidationResult(passed::Bool, message::String)

Create a ValidationResult with a single message.
"""
ValidationResult(passed::Bool, message::String) = ValidationResult(passed, [message])

"""
    passed(result::ValidationResult) -> Bool

Check if validation passed.
"""
passed(result::ValidationResult) = result.passed

"""
    validate_threshold(value, threshold; comparison=:>=, name="Value") -> ValidationResult

Validate a value against a threshold.

# Arguments
- `value`: The value to check
- `threshold`: The threshold to compare against
- `comparison`: Comparison operator (:>=, :<=, :>, :<, :==)
- `name`: Name for the validation message
"""
function validate_threshold(value, threshold; comparison=:>=, name::String="Value")::ValidationResult
    result = if comparison == :>=
        value >= threshold
    elseif comparison == :<=
        value <= threshold
    elseif comparison == :>
        value > threshold
    elseif comparison == :<
        value < threshold
    elseif comparison == :(==)
        value == threshold
    else
        error("Unknown comparison operator: $comparison")
    end
    
    status = result ? "PASS" : "FAIL"
    message = "$status: $name = $(round(value, digits=4)) $comparison $threshold"
    
    return ValidationResult(result, message)
end

"""
    validate_all(validations::Vector{ValidationResult}) -> ValidationResult

Combine multiple validation results. Overall passes only if all pass.
"""
function validate_all(validations::Vector{ValidationResult})::ValidationResult
    all_passed = all(v -> v.passed, validations)
    all_messages = reduce(vcat, [v.messages for v in validations])
    return ValidationResult(all_passed, all_messages)
end

end # module
