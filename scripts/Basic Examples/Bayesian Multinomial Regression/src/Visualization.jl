"""
Visualization - Plotting functions for multinomial regression.

Provides visualization for free energy, probability comparison, and simplex densities.
"""
module Visualization

using Plots
using Printf
using Statistics
using LinearAlgebra
using Distributions

export PlotConfig, plot_free_energy, plot_probability_comparison
export plot_simplex_densities, plot_beta_comparison, create_summary_dashboard

include("SimplexUtils.jl")
using .SimplexUtils: compute_simplex_density

"""
    struct PlotConfig

Configuration for plot styling.
"""
struct PlotConfig
    dpi::Int
    colorscheme::Symbol
    sigma_range::Float64
    simplex_resolution::Int
    simplex_variance::Float64
    simplex_correlation::Float64
    show_values::Bool
end

function PlotConfig(config::Dict)
    viz = get(config, "visualization", Dict())
    PlotConfig(
        get(viz, "dpi", 150),
        Symbol(get(viz, "colorscheme", "viridis")),
        get(viz, "sigma_range", 4.0),
        get(viz, "simplex_resolution", 500),
        get(viz, "simplex_variance", 1.0),
        get(viz, "simplex_correlation", 0.9),
        get(viz, "show_values", true)
    )
end

"""
    plot_free_energy(fe_history; output_path=nothing)

Plot free energy convergence over iterations.
"""
function plot_free_energy(fe_history::AbstractVector; 
                         title_suffix::String="",
                         output_path::Union{Nothing,String}=nothing)
    p = plot(1:length(fe_history), fe_history,
        title="Free Energy Convergence $title_suffix",
        xlabel="Iteration",
        ylabel="Free Energy",
        linewidth=2,
        marker=:circle,
        markersize=3,
        legend=false,
        grid=true,
        size=(800, 400))
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(p, output_path)
        @info "Saved: $output_path"
    end
    
    return p
end

"""
    plot_probability_comparison(estimated, true_probs; output_path=nothing)

Plot comparison between estimated and true probabilities.
"""
function plot_probability_comparison(estimated::AbstractVector, true_probs::AbstractVector;
                                    output_path::Union{Nothing,String}=nothing)
    n = length(estimated)
    
    p1 = bar(1:n, true_probs, 
        label="True",
        alpha=0.7,
        title="Probability Comparison",
        xlabel="Category",
        ylabel="Probability")
    bar!(1:n, estimated, 
        label="Estimated",
        alpha=0.5)
    
    # Also plot scatter comparison
    p2 = scatter(true_probs, estimated,
        xlabel="True Probability",
        ylabel="Estimated Probability",
        title="True vs Estimated",
        legend=false,
        aspect_ratio=:equal)
    plot!([0, 1], [0, 1], linestyle=:dash, color=:red, label="Identity")
    
    final_plot = plot(p1, p2, layout=(1, 2), size=(1000, 400))
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(final_plot, output_path)
        @info "Saved: $output_path"
    end
    
    return final_plot
end

"""
    plot_beta_comparison(estimated_mean, estimated_cov, true_beta; output_path=nothing)

Plot comparison of estimated vs true regression coefficients with uncertainty.
"""
function plot_beta_comparison(estimated_mean::AbstractVector, estimated_std::AbstractVector,
                             true_beta::AbstractVector; output_path::Union{Nothing,String}=nothing)
    n = length(true_beta)
    
    p = scatter(1:n, true_beta,
        label="True β",
        markersize=8,
        markershape=:diamond,
        title="Regression Coefficients",
        xlabel="Coefficient Index",
        ylabel="Value")
    
    scatter!(1:n, estimated_mean,
        label="Estimated β",
        markersize=6,
        yerror=2*estimated_std)  # 95% CI
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(p, output_path)
        @info "Saved: $output_path"
    end
    
    return p
end

"""
    plot_simplex_densities(config::PlotConfig; output_path=nothing)

Plot transformed Gaussian densities on the simplex.
"""
function plot_simplex_densities(config::PlotConfig; output_path::Union{Nothing,String}=nothing)
    σ² = config.simplex_variance
    ρ = config.simplex_correlation
    
    # Create covariance matrices
    Σ_corr = [σ² ρ*σ²; ρ*σ² σ²]
    Σ_anticorr = [σ² -ρ*σ²; -ρ*σ² σ²]
    Σ_uncorr = [σ² 0.0; 0.0 σ²]
    
    # Gaussian space plots
    ψ_range = range(-config.sigma_range*sqrt(σ²), config.sigma_range*sqrt(σ²), length=100)
    
    p1 = contour(ψ_range, ψ_range, 
        (x,y) -> pdf(MvNormal(zeros(2), Σ_corr), [x,y]),
        title="Correlated Prior", xlabel="ψ₁", ylabel="ψ₂")
    
    p2 = contour(ψ_range, ψ_range,
        (x,y) -> pdf(MvNormal(zeros(2), Σ_anticorr), [x,y]),
        title="Anti-correlated Prior", xlabel="ψ₁", ylabel="ψ₂")
    
    p3 = contour(ψ_range, ψ_range,
        (x,y) -> pdf(MvNormal(zeros(2), Σ_uncorr), [x,y]),
        title="Uncorrelated Prior", xlabel="ψ₁", ylabel="ψ₂")
    
    # Simplex space plots
    n_pts = min(config.simplex_resolution, 200)  # Limit for speed
    x = range(0, 1, length=n_pts)
    y = range(0, 1, length=n_pts)
    
    # Simplex with correlated prior
    p4 = contour(x, y, (x,y) -> compute_simplex_density(x, y, Σ_corr),
        title="Correlated Simplex")
    plot!(p4, [0,1,0,0], [0,0,1,0], color=:black, label="")
    
    p5 = contour(x, y, (x,y) -> compute_simplex_density(x, y, Σ_anticorr),
        title="Anti-correlated Simplex")
    plot!(p5, [0,1,0,0], [0,0,1,0], color=:black, label="")
    
    p6 = contour(x, y, (x,y) -> compute_simplex_density(x, y, Σ_uncorr),
        title="Uncorrelated Simplex")
    plot!(p6, [0,1,0,0], [0,0,1,0], color=:black, label="")
    
    final_plot = plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(900, 600))
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(final_plot, output_path)
        @info "Saved: $output_path"
    end
    
    return final_plot
end

end # module
