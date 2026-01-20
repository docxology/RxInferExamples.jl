module Visualization

using Plots
using RxInfer
using Distributions

export plot_rotating_ssm, plot_identification_signals, plot_identification_results, plot_smoothing_results, save_plot, plot_free_energy_history, plot_error_analysis

"""
    save_plot(p, filename, output_dir)

Save a plot to the specified directory and return the absolute path.
"""
function save_plot(p, filename, output_dir)
    mkpath(output_dir)
    path = joinpath(output_dir, filename)
    savefig(p, path)
    abs_path = abspath(path)
    println("Saved plot to: $abs_path")
    return abs_path
end

"""
    plot_free_energy_history(history)

Plot the Free Energy convergence history.
"""
function plot_free_energy_history(history)
    p = plot(title = "Free Energy Convergence", xlabel = "Iteration", ylabel = "Free Energy")
    plot!(p, history, label = "Bethe Free Energy", linewidth = 2, marker = :circle)
    return p
end

"""
    plot_error_analysis(real, estimated, name)

Plot residuals and error distribution.
"""
function plot_error_analysis(real, estimated, name)
    residuals = real .- estimated
    
    p1 = plot(title = "Residuals: $name", xlabel = "Time", ylabel = "Error")
    plot!(p1, residuals, label = "Residuals", color = :red)
    
    p2 = histogram(residuals, title = "Error Distribution", xlabel = "Error", ylabel = "Frequency", label = false, bins = 20)
    
    return plot(p1, p2, layout = (2, 1), size = (800, 600))
end

"""
    plot_rotating_ssm(x, y, xmarginals)

Plot results for the rotating SSM example.
"""
function plot_rotating_ssm(x, y, xmarginals)
    px = plot(title = "Rotating SSM")
    
    # Plot true hidden signal
    px = plot!(px, getindex.(x, 1), label = "Hidden Signal (dim-1)", color = :orange)
    px = scatter!(px, getindex.(y, 1), label = false, markersize = 2, color = :orange)
    px = plot!(px, getindex.(x, 2), label = "Hidden Signal (dim-2)", color = :green)
    px = scatter!(px, getindex.(y, 2), label = false, markersize = 2, color = :green)
    
    # Plot estimated signal
    if !isnothing(xmarginals)
        px = plot!(px, getindex.(mean.(xmarginals), 1), ribbon = getindex.(var.(xmarginals), 1) .|> sqrt, fillalpha = 0.5, label = "Estimated Signal (dim-1)", color = :teal)
        px = plot!(px, getindex.(mean.(xmarginals), 2), ribbon = getindex.(var.(xmarginals), 2) .|> sqrt, fillalpha = 0.5, label = "Estimated Signal (dim-2)", color = :violet)
    end

    return px
end

"""
    plot_identification_signals(real_x, real_w, real_y, f_name)

Plot the underlying generated signals for identification problems.
"""
function plot_identification_signals(real_x, real_w, real_y, f_name)
    pl = plot(title = "Underlying signals")
    pl = plot!(pl, real_x, label = "x")
    pl = plot!(pl, real_w, label = "w")

    pr = plot(title = "Combined y = $f_name")
    pr = scatter!(pr, real_y, ms = 3, color = :red, label = "y")

    return plot(pl, pr, size = (800, 300))
end

"""
    plot_identification_results(real_x, real_w, real_y, xmarginals, wmarginals, smarginals)

Plot the inference results for identification problems.
"""
function plot_identification_results(real_x, real_w, real_y, xmarginals, wmarginals, smarginals)
    px1 = plot(legend = :bottomleft, title = "Estimated hidden signals")
    px2 = plot(legend = :bottomright, title = "Estimated combined signals")

    px1 = plot!(px1, real_x, label = "Real hidden X")
    px1 = plot!(px1, mean.(xmarginals), ribbon = var.(xmarginals), label = "Estimated X")

    px1 = plot!(px1, real_w, label = "Real hidden W")
    px1 = plot!(px1, mean.(wmarginals), ribbon = var.(wmarginals), label = "Estimated W")

    px2 = scatter!(px2, real_y, label = "Observations", ms = 2, alpha = 0.5, color = :red)
    px2 = plot!(px2, mean.(smarginals), ribbon = std.(smarginals), label = "Combined estimated signal", color = :green)

    return plot(px1, px2, size = (800, 300))
end

"""
    plot_smoothing_results(real_signal, missing_indices, x_posteriors)

Plot the results for the smoothing (missing data) example.
"""
function plot_smoothing_results(real_signal, missing_indices, x_posteriors)
    p = plot(real_signal, label = "Noisy signal", legend = :bottomright, title = "Smoothing (Missing Data)")
    scatter!(p, missing_indices, real_signal[missing_indices], ms = 2, opacity = 0.75, label = "Missing region")
    plot!(p, mean.(x_posteriors), ribbon = var.(x_posteriors), label = "Estimated hidden state")
    return p
end

end # module
