"""
Analysis - Statistical analysis for Sprinkler model results.

Provides analysis functions for convergence, CPT learning, and reporting.
"""
module Analysis

using Statistics
using Printf
using JSON

export analyze_learning_result, compute_all_cpt_errors
export generate_analysis_report, generate_json_summary
export get_true_cpts_from_config

"""
    get_true_cpts_from_config(config::Dict) -> Dict

Extract true CPT matrices from configuration.
"""
function get_true_cpts_from_config(config::Dict)
    model = config["model"]
    
    # P(Rain | Cloudy)
    p_rain = model["p_rain_given_cloudy"]
    cpt_rain = [1.0 - p_rain[1]  1.0 - p_rain[2];
                p_rain[1]        p_rain[2]]
    
    # P(Sprinkler | Cloudy)
    p_sprinkler = model["p_sprinkler_given_cloudy"]
    cpt_sprinkler = [1.0 - p_sprinkler[1]  1.0 - p_sprinkler[2];
                     p_sprinkler[1]        p_sprinkler[2]]
    
    return Dict(
        "P(Rain|Cloudy)" => cpt_rain,
        "P(Sprinkler|Cloudy)" => cpt_sprinkler
    )
end

"""
    analyze_learning_result(result, config::Dict) -> NamedTuple

Analyze learning results and compute comprehensive metrics.
"""
function analyze_learning_result(result, config::Dict)
    posteriors = result.posteriors
    
    # Extract learned CPTs
    cpt_cr = mean(last(posteriors[:cpt_cloud_rain]))
    cpt_cs = mean(last(posteriors[:cpt_cloud_sprinkler]))
    cpt_srw = mean(last(posteriors[:cpt_sprinkler_rain_wet_grass]))
    
    # Get true CPTs
    true_cpts = get_true_cpts_from_config(config)
    
    # Compute errors
    model = config["model"]
    p_rain = model["p_rain_given_cloudy"]
    p_sprinkler = model["p_sprinkler_given_cloudy"]
    
    # Error analysis
    rain_errors = [
        abs(cpt_cr[2,1] - p_rain[1]),  # P(Rain=T|Cloudy=F)
        abs(cpt_cr[2,2] - p_rain[2])   # P(Rain=T|Cloudy=T)
    ]
    
    sprinkler_errors = [
        abs(cpt_cs[2,1] - p_sprinkler[1]),  # P(Sprinkler=T|Cloudy=F)
        abs(cpt_cs[2,2] - p_sprinkler[2])   # P(Sprinkler=T|Cloudy=T)
    ]
    
    return (
        learned_cpts = Dict(
            "cpt_cloud_rain" => cpt_cr,
            "cpt_cloud_sprinkler" => cpt_cs,
            "cpt_sprinkler_rain_wet_grass" => cpt_srw
        ),
        true_cpts = true_cpts,
        errors = Dict(
            "rain_max_error" => maximum(rain_errors),
            "rain_mean_error" => mean(rain_errors),
            "sprinkler_max_error" => maximum(sprinkler_errors),
            "sprinkler_mean_error" => mean(sprinkler_errors)
        ),
        learned_values = Dict(
            "P(Rain=T|Cloudy=F)" => cpt_cr[2,1],
            "P(Rain=T|Cloudy=T)" => cpt_cr[2,2],
            "P(Sprinkler=T|Cloudy=F)" => cpt_cs[2,1],
            "P(Sprinkler=T|Cloudy=T)" => cpt_cs[2,2]
        ),
        true_values = Dict(
            "P(Rain=T|Cloudy=F)" => p_rain[1],
            "P(Rain=T|Cloudy=T)" => p_rain[2],
            "P(Sprinkler=T|Cloudy=F)" => p_sprinkler[1],
            "P(Sprinkler=T|Cloudy=T)" => p_sprinkler[2]
        )
    )
end

"""
    generate_analysis_report(analysis_result, validation_results, output_path::String)

Generate comprehensive analysis report as text file.
"""
function generate_analysis_report(analysis, validation_results::Vector, output_path::String)
    open(output_path, "w") do io
        println(io, "="^70)
        println(io, "BAYESIAN NETWORKS ANALYSIS REPORT")
        println(io, "="^70)
        println(io)
        
        # Section 1: Learned CPT Values
        println(io, "LEARNED CPT VALUES")
        println(io, "-"^40)
        for (key, val) in analysis.learned_values
            true_val = analysis.true_values[key]
            error = abs(val - true_val)
            println(io, @sprintf("  %s: %.4f (true: %.4f, error: %.4f)", key, val, true_val, error))
        end
        println(io)
        
        # Section 2: Error Summary
        println(io, "ERROR SUMMARY")
        println(io, "-"^40)
        for (key, val) in analysis.errors
            println(io, @sprintf("  %s: %.4f", key, val))
        end
        println(io)
        
        # Section 3: Validation Results
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
    
    @info "Analysis report saved: $output_path"
end

"""
    generate_json_summary(analysis, validation_results, config, output_path::String)

Generate machine-readable JSON summary of results.
"""
function generate_json_summary(analysis, validation_results::Vector, config::Dict, output_path::String)
    summary = Dict(
        "config" => Dict(
            "seed" => config["general"]["seed"],
            "n_samples" => config["learning"]["n_samples"],
            "iterations" => config["learning"]["learning_iterations"]
        ),
        "learned_values" => analysis.learned_values,
        "true_values" => analysis.true_values,
        "errors" => analysis.errors,
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
