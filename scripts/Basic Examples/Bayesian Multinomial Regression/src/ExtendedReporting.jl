"""
ExtendedReporting - Comprehensive reporting for regression analysis.

Generates detailed text and JSON reports with all available statistics.
"""
module ExtendedReporting

using Printf
using JSON
using Statistics
using LinearAlgebra
using Dates

export generate_comprehensive_report, generate_detailed_json
export generate_latex_table, generate_markdown_report
export ReportConfig

# ============================================================
# Report Configuration
# ============================================================

"""
    struct ReportConfig

Configuration for report generation.
"""
struct ReportConfig
    include_raw_data::Bool
    decimal_places::Int
    include_timestamps::Bool
    include_system_info::Bool
end

ReportConfig() = ReportConfig(false, 6, true, true)

function ReportConfig(config::Dict)
    rep = get(config, "reporting", Dict())
    ReportConfig(
        get(rep, "include_raw_data", false),
        get(rep, "decimal_places", 6),
        get(rep, "include_timestamps", true),
        get(rep, "include_system_info", true)
    )
end

# ============================================================
# Comprehensive Text Report
# ============================================================

"""
    generate_comprehensive_report(stats, config, output_path)

Generate comprehensive text report with all statistics.
"""
function generate_comprehensive_report(;
        cls_stats=nothing,
        reg_stats=nothing,
        validation_results=nothing,
        config::Dict=Dict(),
        output_path::String)
    
    open(output_path, "w") do io
        # Header
        println(io, "="^80)
        println(io, center_text("COMPREHENSIVE BAYESIAN REGRESSION ANALYSIS REPORT", 80))
        println(io, "="^80)
        println(io)
        
        # Timestamp
        println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "="^80)
        println(io)
        
        # Configuration Summary
        println(io, section_header("CONFIGURATION"))
        println(io, format_config(config))
        println(io)
        
        # Classification Results
        if cls_stats !== nothing
            println(io, section_header("CLASSIFICATION EXPERIMENT"))
            println(io, format_classification_stats(cls_stats))
            println(io)
        end
        
        # Regression Results
        if reg_stats !== nothing
            println(io, section_header("REGRESSION EXPERIMENT"))
            println(io, format_regression_stats(reg_stats))
            println(io)
        end
        
        # Validation Results
        if validation_results !== nothing
            println(io, section_header("VALIDATION SUMMARY"))
            println(io, format_validation(validation_results))
            println(io)
        end
        
        # Statistical Interpretation Guide
        println(io, section_header("INTERPRETATION GUIDE"))
        println(io, interpretation_guide())
        println(io)
        
        println(io, "="^80)
        println(io, center_text("END OF REPORT", 80))
        println(io, "="^80)
    end
    
    @info "Comprehensive report saved: $output_path"
end

function center_text(text::String, width::Int)
    pad = max(0, (width - length(text)) ÷ 2)
    return " "^pad * text
end

function section_header(title::String)
    line = "-"^60
    return "$line\n$(uppercase(title))\n$line"
end

function format_config(config::Dict)
    result = ""
    
    if haskey(config, "general")
        result *= "General:\n"
        result *= @sprintf("  Seed: %d\n", get(config["general"], "seed", 0))
        result *= @sprintf("  Output Directory: %s\n", get(config["general"], "output_dir", "output"))
    end
    
    if haskey(config, "classification")
        cls = config["classification"]
        result *= "\nClassification:\n"
        result *= @sprintf("  Trials per observation (N): %d\n", get(cls, "N", 0))
        result *= @sprintf("  Categories (k): %d\n", get(cls, "k", 0))
        result *= @sprintf("  Sample size: %d\n", get(cls, "nsamples", 0))
        result *= @sprintf("  Iterations: %d\n", get(cls, "iterations", 0))
    end
    
    if haskey(config, "regression")
        reg = config["regression"]
        result *= "\nRegression:\n"
        result *= @sprintf("  Trials per observation (N): %d\n", get(reg, "N", 0))
        result *= @sprintf("  Feature dimension (k): %d\n", get(reg, "k", 0))
        result *= @sprintf("  Sample size: %d\n", get(reg, "nsamples", 0))
        result *= @sprintf("  Iterations: %d\n", get(reg, "iterations", 0))
    end
    
    return result
end

function format_classification_stats(stats)
    result = """
Error Metrics:
  Mean Squared Error (MSE):       $(fmt(stats.mse))
  Root MSE (RMSE):                $(fmt(sqrt(stats.mse)))
  Maximum Absolute Error:         $(fmt(stats.max_error))

Correlation Metrics:
  Pearson Correlation:            $(fmt(stats.correlation))

Convergence:
  Final Free Energy:              $(fmt(stats.free_energy_final))
  Final Change (|ΔFE|):           $(fmt(stats.free_energy_change))
  Total Iterations:               $(stats.n_iterations)
"""
    return result
end

function format_regression_stats(stats)
    result = """
Error Metrics:
  Mean Squared Error (MSE):       $(fmt(stats.mse))
  Root MSE (RMSE):                $(fmt(sqrt(stats.mse)))
  Maximum Absolute Error:         $(fmt(stats.max_error))

Uncertainty Quantification:
  Mean Posterior Std:             $(fmt(stats.mean_std))
  95% Credible Interval Coverage: $(fmt(stats.coverage_2sigma * 100))%

Convergence:
  Final Free Energy:              $(fmt(stats.free_energy_final))
  Total Iterations:               $(stats.n_iterations)
"""
    return result
end

function format_validation(results::Vector)
    n_passed = count(r -> r.passed, results)
    n_total = length(results)
    
    result = @sprintf("Overall: %d/%d PASSED\n\n", n_passed, n_total)
    
    result *= "Individual Tests:\n"
    for r in results
        status = r.passed ? "[PASS]" : "[FAIL]"
        result *= "  $status $(r.message)\n"
    end
    
    return result
end

function interpretation_guide()
    return """
MSE (Mean Squared Error):
  < 0.001  : Excellent fit
  0.001-0.01: Good fit
  0.01-0.1 : Acceptable fit
  > 0.1    : Poor fit

Correlation:
  > 0.99   : Nearly perfect
  0.9-0.99 : Excellent
  0.7-0.9  : Good
  < 0.7    : Weak

Coverage (95% CI should contain ~95% of true values):
  90-100%  : Well-calibrated
  < 90%    : Overconfident (intervals too narrow)
  > 100%   : Conservative (intervals too wide)

Convergence:
  |ΔFE| < 1e-6 : Converged
  |ΔFE| < 1e-3 : Near convergence
  |ΔFE| > 1e-3 : May need more iterations
"""
end

fmt(x::Real; dp::Int=6) = @sprintf("%.*f", dp, x)
fmt(x::Int) = string(x)
fmt(x::Nothing) = "N/A"

# ============================================================
# Detailed JSON Report
# ============================================================

"""
    generate_detailed_json(stats; output_path)

Generate detailed JSON with all statistics.
"""
function generate_detailed_json(;
        cls_result=nothing,
        reg_result=nothing,
        cls_stats=nothing,
        reg_stats=nothing,
        validation_results=nothing,
        config::Dict=Dict(),
        output_path::String)
    
    report = Dict{String, Any}()
    
    # Metadata
    report["metadata"] = Dict(
        "generated_at" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "report_type" => "comprehensive_regression_analysis"
    )
    
    # Configuration
    report["configuration"] = config
    
    # Classification
    if cls_stats !== nothing
        report["classification"] = Dict(
            "error_metrics" => Dict(
                "mse" => cls_stats.mse,
                "rmse" => sqrt(cls_stats.mse),
                "max_error" => cls_stats.max_error
            ),
            "correlation" => Dict(
                "pearson" => cls_stats.correlation
            ),
            "convergence" => Dict(
                "free_energy_final" => cls_stats.free_energy_final,
                "free_energy_change" => cls_stats.free_energy_change,
                "iterations" => cls_stats.n_iterations
            )
        )
    end
    
    # Regression
    if reg_stats !== nothing
        report["regression"] = Dict(
            "error_metrics" => Dict(
                "mse" => reg_stats.mse,
                "rmse" => sqrt(reg_stats.mse),
                "max_error" => reg_stats.max_error
            ),
            "uncertainty" => Dict(
                "mean_posterior_std" => reg_stats.mean_std,
                "coverage_95" => reg_stats.coverage_2sigma
            ),
            "convergence" => Dict(
                "free_energy_final" => reg_stats.free_energy_final,
                "iterations" => reg_stats.n_iterations
            )
        )
    end
    
    # Validation
    if validation_results !== nothing
        report["validation"] = Dict(
            "passed" => count(r -> r.passed, validation_results),
            "total" => length(validation_results),
            "all_passed" => all(r -> r.passed, validation_results),
            "tests" => [Dict(
                "name" => r.metric_name,
                "passed" => r.passed,
                "actual" => r.actual_value,
                "threshold" => r.threshold,
                "message" => r.message
            ) for r in validation_results]
        )
    end
    
    open(output_path, "w") do io
        JSON.print(io, report, 2)
    end
    
    @info "Detailed JSON saved: $output_path"
end

# ============================================================
# Markdown Report
# ============================================================

"""
    generate_markdown_report(stats; output_path)

Generate formatted Markdown report.
"""
function generate_markdown_report(;
        cls_stats=nothing,
        reg_stats=nothing,
        validation_results=nothing,
        config::Dict=Dict(),
        figures_dir::String="",
        output_path::String)
    
    open(output_path, "w") do io
        # Title
        println(io, "# Bayesian Multinomial Regression Analysis Report")
        println(io)
        println(io, "_Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))_")
        println(io)
        
        # Table of Contents
        println(io, "## Table of Contents")
        println(io, "1. [Configuration](#configuration)")
        println(io, "2. [Classification Results](#classification-results)")
        println(io, "3. [Regression Results](#regression-results)")
        println(io, "4. [Validation](#validation)")
        println(io, "5. [Visualizations](#visualizations)")
        println(io)
        
        # Configuration
        println(io, "## Configuration")
        println(io)
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        if haskey(config, "general")
            println(io, "| Seed | $(get(config["general"], "seed", "N/A")) |")
        end
        if haskey(config, "classification")
            cls = config["classification"]
            println(io, "| Classification N | $(get(cls, "N", "N/A")) |")
            println(io, "| Classification k | $(get(cls, "k", "N/A")) |")
            println(io, "| Classification samples | $(get(cls, "nsamples", "N/A")) |")
        end
        if haskey(config, "regression")
            reg = config["regression"]
            println(io, "| Regression N | $(get(reg, "N", "N/A")) |")
            println(io, "| Regression k | $(get(reg, "k", "N/A")) |")
            println(io, "| Regression samples | $(get(reg, "nsamples", "N/A")) |")
        end
        println(io)
        
        # Classification Results
        if cls_stats !== nothing
            println(io, "## Classification Results")
            println(io)
            println(io, "### Error Metrics")
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")
            println(io, "| MSE | $(fmt(cls_stats.mse)) |")
            println(io, "| RMSE | $(fmt(sqrt(cls_stats.mse))) |")
            println(io, "| Max Error | $(fmt(cls_stats.max_error)) |")
            println(io, "| Correlation | $(fmt(cls_stats.correlation)) |")
            println(io)
        end
        
        # Regression Results
        if reg_stats !== nothing
            println(io, "## Regression Results")
            println(io)
            println(io, "### Error Metrics")
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")
            println(io, "| MSE | $(fmt(reg_stats.mse)) |")
            println(io, "| RMSE | $(fmt(sqrt(reg_stats.mse))) |")
            println(io, "| Max Error | $(fmt(reg_stats.max_error)) |")
            println(io, "| Mean Std | $(fmt(reg_stats.mean_std)) |")
            println(io, "| 95% Coverage | $(fmt(reg_stats.coverage_2sigma * 100))% |")
            println(io)
        end
        
        # Validation
        if validation_results !== nothing
            println(io, "## Validation")
            println(io)
            n_passed = count(r -> r.passed, validation_results)
            n_total = length(validation_results)
            println(io, "**Overall: $n_passed/$n_total PASSED**")
            println(io)
            println(io, "| Test | Status | Details |")
            println(io, "|------|--------|---------|")
            for r in validation_results
                status = r.passed ? "✅ PASS" : "❌ FAIL"
                println(io, "| $(r.metric_name) | $status | $(r.message) |")
            end
            println(io)
        end
        
        # Visualizations
        if !isempty(figures_dir)
            println(io, "## Visualizations")
            println(io)
            # List PNG files
            if isdir(figures_dir)
                for f in readdir(figures_dir)
                    if endswith(f, ".png")
                        # Use relative path
                        println(io, "### $(replace(f, ".png" => ""))")
                        println(io, "![$(f)]($(joinpath(figures_dir, f)))")
                        println(io)
                    end
                end
            end
        end
    end
    
    @info "Markdown report saved: $output_path"
end

# ============================================================
# LaTeX Table Generation
# ============================================================

"""
    generate_latex_table(stats; output_path)

Generate LaTeX table for publication.
"""
function generate_latex_table(data::Dict{String, Any};
                             caption::String="Analysis Results",
                             label::String="tab:results",
                             output_path::String)
    
    open(output_path, "w") do io
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{$caption}")
        println(io, "\\label{$label}")
        println(io, "\\begin{tabular}{lcc}")
        println(io, "\\hline")
        println(io, "\\textbf{Metric} & \\textbf{Classification} & \\textbf{Regression} \\\\")
        println(io, "\\hline")
        
        # MSE
        cls_mse = get(get(data, "classification", Dict()), "mse", "---")
        reg_mse = get(get(data, "regression", Dict()), "mse", "---")
        println(io, "MSE & $(fmt_latex(cls_mse)) & $(fmt_latex(reg_mse)) \\\\")
        
        # Add more rows as needed
        
        println(io, "\\hline")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
    end
    
    @info "LaTeX table saved: $output_path"
end

fmt_latex(x::Real) = @sprintf("\$%.4f\$", x)
fmt_latex(x::String) = x
fmt_latex(x::Nothing) = "---"

end # module
