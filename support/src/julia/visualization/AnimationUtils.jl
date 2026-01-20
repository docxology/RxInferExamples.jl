"""
AnimationUtils - Animation creation utilities.

Provides utilities for creating animations from plot sequences.

# Exports
- `create_animation`: Create animation from frames
- `save_animation`: Save animation to file
"""
module AnimationUtils

using Plots

export create_animation, save_animation

"""
    create_animation(frames::Vector; fps::Int=10) -> Animation

Create an animation from a vector of plot frames.

# Arguments
- `frames`: Vector of plot objects
- `fps`: Frames per second
"""
function create_animation(frames::Vector; fps::Int=10)
    anim = @animate for frame in frames
        plot(frame)
    end
    return anim
end

"""
    save_animation(anim::Animation, filename::AbstractString, directory::AbstractString; fps::Int=10) -> String

Save animation to a GIF file.
Returns the full path to the saved file.
"""
function save_animation(anim::Animation, filename::AbstractString, directory::AbstractString; fps::Int=10)::String
    mkpath(directory)
    path = joinpath(directory, filename)
    gif(anim, path, fps=fps)
    return path
end

"""
    animate_series(data::AbstractVector, plot_fn::Function; fps::Int=10, show_history::Bool=true)

Create an animation by plotting data incrementally.

# Arguments
- `data`: Vector of data points
- `plot_fn`: Function that takes (data_slice, index) and returns a plot
- `fps`: Frames per second
- `show_history`: If true, show all previous data points; if false, only current
"""
function animate_series(data::AbstractVector, plot_fn::Function; fps::Int=10, show_history::Bool=true)
    anim = @animate for i in 1:length(data)
        if show_history
            slice = data[1:i]
        else
            slice = [data[i]]
        end
        plot_fn(slice, i)
    end
    
    return anim
end

export animate_series

end # module
