"""
Visualization - Enhanced visualization for Bayesian Networks.

Provides advanced plotting functions with configurable themes.
"""
module Visualization

using Plots
using Printf
using Statistics

export PlotConfig, plot_posterior_bars_enhanced, plot_cpt_heatmaps
export plot_cpt_comparison, plot_convergence, create_dashboard
export plot_network_structure, setup_plot_theme

"""
    struct PlotConfig

Configuration container for plot styling.
"""
struct PlotConfig
    colorscheme::Symbol
    dpi::Int
    font_size::Int
    title_font_size::Int
    show_values::Bool
    bar_color::String
    bar_edge_color::String
    posterior_size::Tuple{Int,Int}
    heatmap_size::Tuple{Int,Int}
    dashboard_size::Tuple{Int,Int}
end

"""
    PlotConfig(config::Dict)

Create PlotConfig from configuration dictionary.
"""
function PlotConfig(config::Dict)
    viz = get(config, "visualization", Dict())
    PlotConfig(
        Symbol(get(viz, "colorscheme", "viridis")),
        get(viz, "dpi", 150),
        get(viz, "font_size", 10),
        get(viz, "title_font_size", 12),
        get(viz, "show_values", true),
        get(viz, "bar_color", "#4C72B0"),
        get(viz, "bar_edge_color", "#2E4057"),
        tuple(get(viz, "posterior_plot_size", [350, 300])...),
        tuple(get(viz, "cpt_heatmap_size", [350, 350])...),
        tuple(get(viz, "dashboard_size", [1400, 900])...)
    )
end

"""
    setup_plot_theme(config::PlotConfig)

Apply plot theme from configuration.
"""
function setup_plot_theme(config::PlotConfig)
    default(
        fontfamily = "sans-serif",
        titlefontsize = config.title_font_size,
        tickfontsize = config.font_size,
        guidefontsize = config.font_size,
        legendfontsize = config.font_size - 1,
        dpi = config.dpi
    )
end

"""
    plot_posterior_bars_enhanced(posteriors, variables, labels, title_suffix, config; output_path=nothing)

Enhanced posterior bar plot with value annotations.
"""
function plot_posterior_bars_enhanced(posteriors, variables::Vector{Symbol}, labels::Dict,
                                     title_suffix::String, config::PlotConfig;
                                     output_path::Union{Nothing,String}=nothing)
    plots_list = []
    
    for var in variables
        if haskey(posteriors, var)
            posterior = last(posteriors[var])
            var_labels = get(labels, var, ["State 1", "State 2"])
            probs = posterior.p
            
            p = bar(probs,
                xticks=(1:length(var_labels), var_labels),
                ylabel="Probability",
                title=@sprintf("P(%s | %s)", string(var), title_suffix),
                titlefontsize=config.title_font_size,
                legend=false,
                ylims=(0, 1.15),
                color=config.bar_color,
                linecolor=config.bar_edge_color,
                linewidth=1.5)
            
            # Add value annotations
            if config.show_values
                for (i, prob) in enumerate(probs)
                    annotate!(p, i, prob + 0.05, text(@sprintf("%.2f", prob), 8, :center))
                end
            end
            
            push!(plots_list, p)
        end
    end
    
    n = length(plots_list)
    w, h = config.posterior_size
    final_plot = plot(plots_list..., layout=(1, n), size=(w*n, h), margin=5Plots.mm)
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(final_plot, output_path)
        @info "Saved posterior plot: $output_path"
    end
    
    return final_plot
end

"""
    plot_cpt_heatmaps(cpt_dict::Dict, config::PlotConfig; output_path=nothing)

Plot multiple CPT heatmaps with consistent styling.
"""
function plot_cpt_heatmaps(cpts::Dict, config::PlotConfig; 
                           output_path::Union{Nothing,String}=nothing)
    plots_list = []
    
    for (name, cpt) in cpts
        if ndims(cpt) == 2
            p = heatmap(cpt,
                title=name,
                xlabel="Parent State",
                ylabel="Child State",
                xticks=(1:size(cpt,2), ["False", "True"]),
                yticks=(1:size(cpt,1), ["False", "True"]),
                c=config.colorscheme,
                clims=(0, 1),
                aspect_ratio=:equal,
                xrotation=45,
                margin=10Plots.mm)
            
            # Add value annotations
            if config.show_values
                for i in 1:size(cpt,1), j in 1:size(cpt,2)
                    annotate!(p, j, i, text(@sprintf("%.2f", cpt[i,j]), 9, :white, :center))
                end
            end
            
            push!(plots_list, p)
        end
    end
    
    n = length(plots_list)
    w, h = config.heatmap_size
    final_plot = plot(plots_list..., layout=(1, n), size=(w*n, h))
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(final_plot, output_path)
        @info "Saved CPT heatmaps: $output_path"
    end
    
    return final_plot
end

"""
    plot_cpt_comparison(learned::Dict, true_cpt::Dict, config::PlotConfig; output_path=nothing)

Side-by-side comparison of learned vs true CPTs.
"""
function plot_cpt_comparison(learned::Dict, true_cpts::Dict, config::PlotConfig;
                            output_path::Union{Nothing,String}=nothing)
    plots_list = []
    
    for (name, learned_cpt) in learned
        if haskey(true_cpts, name) && ndims(learned_cpt) == 2
            true_cpt = true_cpts[name]
            error_cpt = abs.(learned_cpt .- true_cpt)
            
            # Learned
            p1 = heatmap(learned_cpt, title="Learned: $name", c=config.colorscheme,
                clims=(0,1), aspect_ratio=:equal, xticks=(1:2, ["F","T"]), yticks=(1:2, ["F","T"]))
            
            # True
            p2 = heatmap(true_cpt, title="True: $name", c=config.colorscheme,
                clims=(0,1), aspect_ratio=:equal, xticks=(1:2, ["F","T"]), yticks=(1:2, ["F","T"]))
            
            # Error
            p3 = heatmap(error_cpt, title="Error: $name", c=:reds,
                clims=(0,0.5), aspect_ratio=:equal, xticks=(1:2, ["F","T"]), yticks=(1:2, ["F","T"]))
            
            push!(plots_list, p1, p2, p3)
        end
    end
    
    n = length(plots_list)
    if n > 0
        rows = div(n, 3)
        final_plot = plot(plots_list..., layout=(rows, 3), size=(900, 300*rows))
        
        if output_path !== nothing
            mkpath(dirname(output_path))
            savefig(final_plot, output_path)
            @info "Saved CPT comparison: $output_path"
        end
        
        return final_plot
    end
    return nothing
end

"""
    plot_convergence(free_energy_history::Vector, config::PlotConfig; output_path=nothing)

Plot variational free energy convergence over iterations.
"""
function plot_convergence(fe_history::Vector, config::PlotConfig;
                         output_path::Union{Nothing,String}=nothing)
    p = plot(1:length(fe_history), fe_history,
        xlabel="Iteration",
        ylabel="Free Energy",
        title="Variational Inference Convergence",
        legend=false,
        linewidth=2,
        marker=:circle,
        markersize=4,
        size=(800, 400))
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(p, output_path)
        @info "Saved convergence plot: $output_path"
    end
    
    return p
end

"""
    create_dashboard(posteriors_dict, cpts_dict, validation_results, config; output_path=nothing)

Create a comprehensive dashboard combining multiple visualizations.
"""
function create_dashboard(posterior_plots::Vector, cpt_plot, validation_summary::String,
                         config::PlotConfig; output_path::Union{Nothing,String}=nothing)
    # Create text annotation for validation
    val_plot = plot(
        [],
        legend=false,
        axis=false,
        grid=false,
        title="Validation Summary",
        annotations=(0.5, 0.5, text(validation_summary, 10, :center))
    )
    
    # Combine all plots
    n_posteriors = length(posterior_plots)
    all_plots = vcat(posterior_plots, [cpt_plot, val_plot])
    
    final_plot = plot(all_plots..., 
        layout=@layout([grid(1, n_posteriors); a{0.6h}; b{0.2h}]),
        size=config.dashboard_size)
    
    if output_path !== nothing
        mkpath(dirname(output_path))
        savefig(final_plot, output_path)
        @info "Saved dashboard: $output_path"
    end
    
    return final_plot
end

# Default variable labels
const DEFAULT_LABELS = Dict(
    :clouded => ["Not Cloudy", "Cloudy"],
    :rain => ["No Rain", "Rain"],
    :sprinkler => ["Off", "On"],
    :wet_grass => ["Dry", "Wet"]
)

export DEFAULT_LABELS

end # module
