# Trajectory Visualization

using Plots

"""
    TrajectoryRecorder

Records agent trajectories for visualization.
"""
mutable struct TrajectoryRecorder
    positions::Vector{Tuple{Int,Int}}
    actions::Vector{Int}
    observations::Vector{Tuple{Int,Int}}
    beliefs::Vector{Any}
    policies::Vector{Any}
    step_times::Vector{Float64}
end

TrajectoryRecorder() = TrajectoryRecorder([], [], [], [], [], [])

"""
    clear!(recorder)

Clear all recorded data.
"""
function clear!(recorder::TrajectoryRecorder)
    empty!(recorder.positions)
    empty!(recorder.actions)
    empty!(recorder.observations)
    empty!(recorder.beliefs)
    empty!(recorder.policies)
    empty!(recorder.step_times)
end

"""
    record_step!(recorder, agent, action_idx, obs, step_time)

Record a single step of agent behavior.
"""
function record_step!(recorder::TrajectoryRecorder, agent, action_idx::Int, 
                      obs::Tuple{Int,Int}, step_time::Float64)
    push!(recorder.positions, obs)
    push!(recorder.actions, action_idx)
    push!(recorder.observations, obs)
    push!(recorder.beliefs, deepcopy(agent.p_s))
    push!(recorder.policies, deepcopy(agent.current_policy))
    push!(recorder.step_times, step_time)
end

"""
    plot_trajectory(positions::Vector{Tuple{Int,Int}}; save_path=nothing)

Plot agent trajectory on the grid world.
"""
function plot_trajectory(positions::Vector{Tuple{Int,Int}}; save_path=nothing)
    grid_size = grid_params.grid_size
    
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    
    # Create grid background
    p = heatmap(1:grid_size, 1:grid_size, zeros(grid_size, grid_size)',
        color=:grays,
        clims=(0, 1),
        colorbar=false,
        xlabel="X Position",
        ylabel="Y Position",
        title="Agent Trajectory ($(length(positions)) steps)",
        aspect_ratio=:equal
    )
    
    # Add wind indicators
    for x in 1:grid_size
        wind_strength = grid_params.wind[x]
        if wind_strength > 0
            for y in 1:grid_size
                annotate!(p, x, y, text("↑"^wind_strength, 8, :gray))
            end
        end
    end
    
    # Plot trajectory with color gradient
    for i in 1:(length(positions)-1)
        alpha = 0.3 + 0.7 * (i / length(positions))
        plot!(p, [xs[i], xs[i+1]], [ys[i], ys[i+1]],
            linewidth=3, color=RGBA(0.2, 0.4, 0.8, alpha),
            label=nothing)
    end
    
    # Mark start, positions, and goal
    scatter!(p, [xs[1]], [ys[1]], 
        marker=:circle, markersize=15, color=:green, label="Start")
    scatter!(p, xs[2:end-1], ys[2:end-1],
        marker=:circle, markersize=8, color=:blue, label="Path")
    scatter!(p, [xs[end]], [ys[end]],
        marker=:diamond, markersize=12, color=:purple, label="End")
    scatter!(p, [grid_params.goal[1]], [grid_params.goal[2]],
        marker=:star, markersize=18, color=:red, label="Goal")
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved trajectory plot to $save_path")
    end
    
    return p
end

"""
    plot_action_sequence(actions::Vector{Int}; save_path=nothing)

Plot the sequence of actions taken as scatter plot.
"""
function plot_action_sequence(actions::Vector{Int}; save_path=nothing)
    action_names = [ACTION_NAMES[a] for a in actions]
    colors = [a == 1 ? :blue : a == 2 ? :green : a == 3 ? :orange : :red for a in actions]
    
    p = scatter(1:length(actions), actions,
        xlabel="Time Step",
        ylabel="Action",
        title="Action Sequence",
        yticks=(1:4, ["Up", "Right", "Down", "Left"]),
        color=colors,
        markersize=10,
        markerstrokewidth=2,
        legend=false,
        ylims=(0.5, 4.5)
    )
    
    # Connect with lines
    plot!(p, 1:length(actions), actions, 
        linewidth=1, linestyle=:dash, color=:gray, alpha=0.5, label=nothing)
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved action sequence plot to $save_path")
    end
    
    return p
end

"""
    plot_experiment_summary(recorder::TrajectoryRecorder; save_path=nothing)

Create a comprehensive summary plot of a single experiment.
"""
function plot_experiment_summary(recorder::TrajectoryRecorder; save_path=nothing)
    p1 = plot_trajectory(recorder.positions)
    p2 = plot_action_sequence(recorder.actions)
    
    # Position over time
    xs = [p[1] for p in recorder.positions]
    ys = [p[2] for p in recorder.positions]
    
    p3 = plot(1:length(xs), xs, label="X", linewidth=2)
    plot!(p3, 1:length(ys), ys, label="Y", linewidth=2)
    hline!(p3, [grid_params.goal[1]], linestyle=:dash, color=:gray, label="Goal X")
    hline!(p3, [grid_params.goal[2]], linestyle=:dot, color=:gray, label="Goal Y")
    xlabel!(p3, "Time Step")
    ylabel!(p3, "Position")
    title!(p3, "Position Components Over Time")
    
    # Distance to goal
    distances = [abs(px - grid_params.goal[1]) + abs(py - grid_params.goal[2]) 
                 for (px, py) in recorder.positions]
    p4 = plot(1:length(distances), distances,
        xlabel="Time Step",
        ylabel="Manhattan Distance",
        title="Distance to Goal",
        linewidth=2,
        color=:purple,
        legend=false
    )
    hline!(p4, [0], linestyle=:dash, color=:green, label="Goal")
    
    combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
    
    if !isnothing(save_path)
        savefig(combined, save_path)
        rxlog("info", "Saved experiment summary to $save_path")
    end
    
    return combined
end

export TrajectoryRecorder, clear!, record_step!
export plot_trajectory, plot_action_sequence, plot_experiment_summary
