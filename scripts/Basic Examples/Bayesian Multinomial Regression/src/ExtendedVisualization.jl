"""
ExtendedVisualization - Comprehensive visualization for regression and multivariate data.

Provides extensive plotting functions for Bayesian regression analysis.
"""
module ExtendedVisualization

using Plots
using Statistics
using LinearAlgebra
using Printf
using Distributions  # Use proper Distributions.jl

export plot_coefficient_forest, plot_posterior_density, plot_residuals_panel
export plot_correlation_heatmap, plot_qq_normal, plot_calibration
export plot_convergence_diagnostics, plot_uncertainty_decomposition
export plot_prediction_intervals, plot_error_distribution
export plot_parameter_trace, plot_autocorrelation
export plot_comprehensive_dashboard, plot_multivariate_panel
export PlotStyle

# ============================================================
# Plot Styling
# ============================================================

"""
    struct PlotStyle

Configuration for consistent plot styling.
"""
struct PlotStyle
    dpi::Int
    colorscheme::Symbol
    primary_color::Symbol
    secondary_color::Symbol
    error_color::Symbol
    success_color::Symbol
    font_size::Int
    line_width::Float64
    marker_size::Int
    grid_alpha::Float64
end

PlotStyle() = PlotStyle(150, :viridis, :steelblue, :coral, :firebrick, :forestgreen,
                        10, 2.0, 5, 0.3)

function PlotStyle(config::Dict)
    viz = get(config, "visualization", Dict())
    PlotStyle(
        get(viz, "dpi", 150),
        Symbol(get(viz, "colorscheme", "viridis")),
        Symbol(get(viz, "primary_color", "steelblue")),
        Symbol(get(viz, "secondary_color", "coral")),
        Symbol(get(viz, "error_color", "firebrick")),
        Symbol(get(viz, "success_color", "forestgreen")),
        get(viz, "font_size", 10),
        get(viz, "line_width", 2.0),
        get(viz, "marker_size", 5),
        get(viz, "grid_alpha", 0.3)
    )
end

# ============================================================
# Coefficient Visualizations
# ============================================================

"""
    plot_coefficient_forest(names, means, stds, true_values; kwargs...)

Forest plot of coefficients with credible intervals.
"""
function plot_coefficient_forest(names::Vector{String}, 
                                 means::AbstractVector,
                                 stds::AbstractVector,
                                 true_values::AbstractVector;
                                 levels=[0.5, 0.95],
                                 output_path::Union{Nothing,String}=nothing,
                                 style::PlotStyle=PlotStyle())
    n = length(names)
    y_positions = n:-1:1
    
    p = plot(size=(600, 50 + 30*n), legend=:topright, grid=true,
             xlabel="Value", ylabel="", yticks=(y_positions, names),
             title="Coefficient Forest Plot")
    
    # Plot intervals (wider first)
    z_scores = Dict(0.5 => 0.6745, 0.95 => 1.96, 0.99 => 2.576)
    colors = [:lightblue, :steelblue]
    
    for (idx, level) in enumerate(sort(levels, rev=true))
        z = get(z_scores, level, 1.96)
        for i in 1:n
            plot!(p, [means[i] - z*stds[i], means[i] + z*stds[i]], [y_positions[i], y_positions[i]],
                  linewidth=6-2*idx, color=colors[min(idx, length(colors))], label=idx==1 && i==1 ? "$(Int(level*100))% CI" : "")
        end
    end
    
    # Plot means
    scatter!(p, means, y_positions, markersize=6, color=:white, 
             markerstrokecolor=:black, markerstrokewidth=2, label="Estimate")
    
    # Plot true values
    scatter!(p, true_values, y_positions, markershape=:diamond, 
             markersize=6, color=:red, label="True")
    
    # Add zero line
    vline!(p, [0], linestyle=:dash, color=:gray, alpha=0.5, label="")
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(p, output_path)
        @info "Saved: $output_path"
    end
    
    return p
end

"""
    plot_posterior_density(samples; true_value=nothing, kwargs...)

Plot posterior density with optional true value.
"""
function plot_posterior_density(samples::AbstractVector;
                               true_value::Union{Nothing,Float64}=nothing,
                               name::String="Parameter",
                               output_path::Union{Nothing,String}=nothing)
    p = density(samples, fill=true, alpha=0.3, label="Posterior",
                title="Posterior Density: $name", xlabel=name, ylabel="Density")
    
    # Add mean and median
    vline!(p, [mean(samples)], linewidth=2, label="Mean", color=:blue)
    vline!(p, [median(samples)], linewidth=2, linestyle=:dash, label="Median", color=:green)
    
    if true_value !== nothing
        vline!(p, [true_value], linewidth=2, color=:red, label="True")
    end
    
    # Add 95% CI
    ci_low, ci_high = quantile(samples, [0.025, 0.975])
    vspan!(p, [ci_low, ci_high], alpha=0.1, color=:blue, label="95% CI")
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(p, output_path)
        @info "Saved: $output_path"
    end
    
    return p
end

# ============================================================
# Residual Visualizations
# ============================================================

"""
    plot_residuals_panel(residuals, fitted; output_path=nothing)

4-panel residual diagnostic plot.
"""
function plot_residuals_panel(residuals::AbstractVector, 
                              fitted::AbstractVector;
                              output_path::Union{Nothing,String}=nothing)
    n = length(residuals)
    d = Normal(0.0, 1.0)
    
    # 1. Residuals vs Fitted
    p1 = scatter(fitted, residuals, alpha=0.5, label="",
                 xlabel="Fitted Values", ylabel="Residuals",
                 title="Residuals vs Fitted")
    hline!(p1, [0], linestyle=:dash, color=:red, label="")
    sorted_idx = sortperm(fitted)
    plot!(p1, fitted[sorted_idx], residuals[sorted_idx], alpha=0.3, linewidth=2, color=:blue, label="")
    
    # 2. Q-Q Plot
    sorted_resid = sort((residuals .- mean(residuals)) ./ std(residuals))
    theoretical = [Distributions.quantile(d, (i - 0.5)/n) for i in 1:n]
    p2 = scatter(theoretical, sorted_resid, alpha=0.5, label="",
                 xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
                 title="Normal Q-Q Plot")
    plot!(p2, [-3, 3], [-3, 3], linestyle=:dash, color=:red, label="")
    
    # 3. Scale-Location
    sqrt_resid = sqrt.(abs.(residuals .- mean(residuals)) ./ std(residuals))
    p3 = scatter(fitted, sqrt_resid, alpha=0.5, label="",
                 xlabel="Fitted Values", ylabel="√|Standardized Residuals|",
                 title="Scale-Location")
    
    # 4. Residuals Histogram
    p4 = histogram(residuals, bins=30, normalize=:pdf, alpha=0.7, label="Residuals",
                   xlabel="Residuals", ylabel="Density", title="Residual Distribution")
    x_range = range(minimum(residuals), maximum(residuals), length=100)
    d_fit = Normal(mean(residuals), std(residuals))
    y_vals = [Distributions.pdf(d_fit, x) for x in x_range]
    plot!(p4, collect(x_range), y_vals, linewidth=2, color=:red, label="Normal")
    
    final_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700))
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(final_plot, output_path)
        @info "Saved: $output_path"
    end
    
    return final_plot
end

"""
    plot_qq_normal(values; output_path=nothing)

Q-Q plot against standard normal.
"""
function plot_qq_normal(values::AbstractVector;
                       title::String="Normal Q-Q Plot",
                       output_path::Union{Nothing,String}=nothing)
    n = length(values)
    d = Normal(0.0, 1.0)
    standardized = (values .- mean(values)) ./ std(values)
    sorted_vals = sort(standardized)
    theoretical = [Distributions.quantile(d, (i - 0.5)/n) for i in 1:n]
    
    p = scatter(theoretical, sorted_vals, alpha=0.6, label="Data",
                xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
                title=title)
    
    lims = extrema([theoretical; sorted_vals])
    plot!(p, [lims[1], lims[2]], [lims[1], lims[2]], 
          linestyle=:dash, color=:red, linewidth=2, label="1:1 Line")
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(p, output_path)
        @info "Saved: $output_path"
    end
    
    return p
end

# ============================================================
# Correlation and Covariance Visualizations
# ============================================================

"""
    plot_correlation_heatmap(corr_matrix, names; output_path=nothing)

Heatmap of correlation matrix with annotations.
"""
function plot_correlation_heatmap(corr::AbstractMatrix,
                                 names::Vector{String};
                                 output_path::Union{Nothing,String}=nothing)
    n = size(corr, 1)
    
    p = heatmap(corr, c=:RdBu, clims=(-1, 1),
                xticks=(1:n, names), yticks=(1:n, names),
                xrotation=45, title="Correlation Matrix",
                aspect_ratio=:equal, size=(500, 500))
    
    # Add correlation values
    for i in 1:n, j in 1:n
        val = corr[i, j]
        color = abs(val) > 0.5 ? :white : :black
        annotate!(p, j, i, text(@sprintf("%.2f", val), 8, color))
    end
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(p, output_path)
        @info "Saved: $output_path"
    end
    
    return p
end

# ============================================================
# Convergence Visualizations
# ============================================================

"""
    plot_convergence_diagnostics(free_energy; output_path=nothing)

Multi-panel convergence diagnostic plot.
"""
function plot_convergence_diagnostics(fe::AbstractVector;
                                      output_path::Union{Nothing,String}=nothing)
    n = length(fe)
    
    p1 = plot(1:n, fe, linewidth=2, label="",
              xlabel="Iteration", ylabel="Free Energy",
              title="Free Energy Trace")
    
    diffs = abs.(diff(fe))
    diffs = max.(diffs, 1e-15)
    p2 = plot(2:n, log10.(diffs), linewidth=2, label="",
              xlabel="Iteration", ylabel="log₁₀|ΔFE|",
              title="Convergence Rate (log scale)")
    
    cumulative = cumsum(diffs)
    p3 = plot(2:n, cumulative, linewidth=2, label="",
              xlabel="Iteration", ylabel="Cumulative |ΔFE|",
              title="Cumulative Change")
    
    window = min(5, n÷4)
    if window >= 2
        ma = [mean(diffs[max(1,i-window+1):i]) for i in 1:length(diffs)]
        p4 = plot(2:n, ma, linewidth=2, label="",
                  xlabel="Iteration", ylabel="Moving Avg |ΔFE|",
                  title="Smoothed Convergence (window=$window)")
    else
        p4 = plot(title="Insufficient data")
    end
    
    final_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700))
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(final_plot, output_path)
        @info "Saved: $output_path"
    end
    
    return final_plot
end

# ============================================================
# Error Distribution Visualizations
# ============================================================

"""
    plot_error_distribution(errors; output_path=nothing)

Multi-panel error distribution analysis.
"""
function plot_error_distribution(errors::AbstractVector;
                                output_path::Union{Nothing,String}=nothing)
    μ = mean(errors)
    σ = std(errors)
    
    # 1. Histogram with stats
    p1 = histogram(errors, bins=30, normalize=:pdf, alpha=0.7, label="Errors",
                   xlabel="Error", ylabel="Density", title="Error Distribution")
    x_range = collect(range(minimum(errors), maximum(errors), length=100))
    d_fit = Normal(μ, σ)
    y_vals = [Distributions.pdf(d_fit, x) for x in x_range]
    plot!(p1, x_range, y_vals, linewidth=2, color=:red, label="Normal Fit")
    
    # 2. Boxplot
    p2 = boxplot(["Errors"], errors, label="", title="Error Boxplot",
                 ylabel="Error Value")
    
    # 3. ECDF
    sorted_errors = sort(errors)
    ecdf_vals = (1:length(errors)) ./ length(errors)
    p3 = plot(sorted_errors, ecdf_vals, linewidth=2, label="Empirical CDF",
              xlabel="Error", ylabel="Cumulative Probability",
              title="Empirical CDF")
    y_cdf = [Distributions.cdf(d_fit, x) for x in x_range]
    plot!(p3, x_range, y_cdf, linestyle=:dash, color=:red, label="Normal CDF")
    
    # 4. Absolute error over index
    p4 = plot(1:length(errors), abs.(errors), linewidth=1, alpha=0.7, label="",
              xlabel="Observation Index", ylabel="|Error|",
              title="Absolute Error by Index")
    hline!(p4, [mean(abs.(errors))], linestyle=:dash, color=:red, label="Mean |Error|")
    
    final_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700))
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(final_plot, output_path)
        @info "Saved: $output_path"
    end
    
    return final_plot
end

# ============================================================
# Comprehensive Dashboard
# ============================================================

"""
    plot_comprehensive_dashboard(est_mean, est_std, true_values, free_energy; output_path=nothing)

6-panel comprehensive analysis dashboard.
"""
function plot_comprehensive_dashboard(est_mean::AbstractVector,
                                      est_std::AbstractVector,
                                      true_values::AbstractVector,
                                      free_energy::AbstractVector;
                                      output_path::Union{Nothing,String}=nothing)
    errors = est_mean .- true_values
    n = length(est_mean)
    d = Normal(0.0, 1.0)
    
    # 1. Coefficient comparison
    p1 = scatter(1:n, true_values, label="True", markersize=8, markershape=:diamond,
                 xlabel="Coefficient Index", ylabel="Value", title="Coefficient Comparison")
    scatter!(p1, 1:n, est_mean, yerror=2*est_std, label="Estimated (±2σ)", markersize=6)
    
    # 2. True vs Estimated scatter
    p2 = scatter(true_values, est_mean, alpha=0.7, label="",
                 xlabel="True Value", ylabel="Estimated Value",
                 title="True vs Estimated", aspect_ratio=:equal)
    lims = extrema([true_values; est_mean])
    plot!(p2, [lims[1], lims[2]], [lims[1], lims[2]], linestyle=:dash, color=:red, label="1:1")
    
    # 3. Error histogram
    p3 = histogram(errors, bins=20, normalize=:pdf, alpha=0.7, label="",
                   xlabel="Error", ylabel="Density", title="Error Distribution")
    
    # 4. Q-Q Plot
    sorted_errors = sort((errors .- mean(errors)) ./ std(errors))
    theoretical = [Distributions.quantile(d, (i - 0.5)/n) for i in 1:n]
    p4 = scatter(theoretical, sorted_errors, alpha=0.7, label="",
                 xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
                 title="Normal Q-Q Plot")
    plot!(p4, [-3, 3], [-3, 3], linestyle=:dash, color=:red, label="")
    
    # 5. Free energy
    p5 = plot(1:length(free_energy), free_energy, linewidth=2, label="",
              xlabel="Iteration", ylabel="Free Energy", title="Convergence")
    
    # 6. Coverage check
    z_levels = [0.5, 0.9, 0.95]
    z_scores = [0.6745, 1.645, 1.96]
    coverages = [mean(abs.(errors) .< z * est_std) for z in z_scores]
    p6 = bar(string.(Int.(z_levels .* 100)) .* "%", coverages .* 100, label="",
             xlabel="Nominal Level", ylabel="Actual Coverage (%)",
             title="Calibration Check")
    hline!(p6, z_levels .* 100, linestyle=:dash, color=:red, label="Expected")
    
    final_plot = plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1200, 700))
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(final_plot, output_path)
        @info "Saved: $output_path"
    end
    
    return final_plot
end

end # module
