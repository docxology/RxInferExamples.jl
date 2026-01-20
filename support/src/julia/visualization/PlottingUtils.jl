"""
PlottingUtils - Plotting helper utilities.

Provides common plotting operations and styling.

# Exports
- `save_plot`: Save a plot to file with directory creation
- `create_figure`: Create a new figure with standard styling
- `add_reference_line`: Add horizontal/vertical reference lines
"""
module PlottingUtils

using Plots

export save_plot, create_figure, add_reference_line, setup_plot_defaults

"""
    setup_plot_defaults()

Set up default plotting parameters for consistent styling.
"""
function setup_plot_defaults()
    default(
        fontfamily = "Computer Modern",
        titlefontsize = 12,
        guidefontsize = 10,
        tickfontsize = 8,
        legendfontsize = 8,
        margin = 5Plots.mm,
        linewidth = 2
    )
end

"""
    save_plot(plot, filename::AbstractString, directory::AbstractString; dpi::Int=150) -> String

Save a plot to file, creating directory if needed.
Returns the full path to the saved file.
"""
function save_plot(plot, filename::AbstractString, directory::AbstractString; dpi::Int=150)::String
    mkpath(directory)
    path = joinpath(directory, filename)
    savefig(plot, path)
    return path
end

"""
    create_figure(; title::String="", xlabel::String="", ylabel::String="", size=(800, 600)) -> Plot

Create a new figure with standard styling.
"""
function create_figure(; title::String="", xlabel::String="", ylabel::String="", size=(800, 600))
    return plot(
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        size = size,
        legend = :topright,
        grid = true,
        gridalpha = 0.3
    )
end

"""
    add_reference_line!(p, value; orientation=:horizontal, color=:red, linestyle=:dash, label="")

Add a reference line to an existing plot.

# Arguments
- `p`: Plot object
- `value`: Value for the reference line
- `orientation`: :horizontal or :vertical
- `color`: Line color
- `linestyle`: Line style
- `label`: Legend label
"""
function add_reference_line!(p, value; orientation::Symbol=:horizontal, color=:red, linestyle=:dash, label::String="")
    if orientation == :horizontal
        hline!(p, [value], color=color, linestyle=linestyle, label=label, linewidth=1.5)
    elseif orientation == :vertical
        vline!(p, [value], color=color, linestyle=linestyle, label=label, linewidth=1.5)
    else
        error("orientation must be :horizontal or :vertical")
    end
    return p
end

end # module
