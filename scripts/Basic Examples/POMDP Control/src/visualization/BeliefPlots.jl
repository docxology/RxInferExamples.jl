# Belief State Visualization

using Plots

"""
    plot_state_belief(agent; save_path=nothing)

Visualize the current state belief as a heatmap over the grid.
"""
function plot_state_belief(agent; save_path=nothing)
    probs = params(agent.p_s)[1]  # Get probability vector
    grid_size = grid_params.grid_size
    
    # Reshape to grid
    belief_grid = reshape(probs, grid_size, grid_size)
    
    p = heatmap(1:grid_size, 1:grid_size, belief_grid',
        xlabel="X Position",
        ylabel="Y Position",
        title="State Belief: P(s)",
        color=:YlOrRd,
        aspect_ratio=:equal,
        clims=(0, 1)
    )
    
    # Mark start and goal
    scatter!(p, [grid_params.start[1]], [grid_params.start[2]],
        marker=:circle, markersize=12, color=:blue, label="Start")
    scatter!(p, [grid_params.goal[1]], [grid_params.goal[2]],
        marker=:star, markersize=15, color=:green, label="Goal")
    
    # Mark current position if available
    try
        pos = get_current_position(agent)
        scatter!(p, [pos[1]], [pos[2]],
            marker=:diamond, markersize=10, color=:purple, label="Current")
    catch
    end
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved state belief plot to $save_path")
    end
    
    return p
end

"""
    plot_policy(agent; save_path=nothing)

Visualize the current policy as action probabilities.
"""
function plot_policy(agent; save_path=nothing)
    policy = agent.current_policy
    n_steps = length(policy)
    
    # Extract action probabilities for each time step
    action_probs = zeros(4, n_steps)
    for t in 1:n_steps
        probs = params(policy[t])[1]
        action_probs[:, t] = probs
    end
    
    action_labels = ["Up", "Right", "Down", "Left"]
    action_colors = [:blue, :green, :orange, :red]
    
    # Create stacked area-style bar plot
    p = plot(xlabel="Planning Step", ylabel="Action Probability",
        title="Current Policy", xlims=(0.5, n_steps + 0.5), ylims=(0, 1),
        legend=:outertopright)
    
    for (i, (label, color)) in enumerate(zip(action_labels, action_colors))
        bar!(p, 1:n_steps, action_probs[i, :], 
            label=label, color=color, alpha=0.7, bar_width=0.6)
    end
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved policy plot to $save_path")
    end
    
    return p
end

"""
    plot_goal_prior(agent; save_path=nothing)

Visualize the goal prior as a heatmap over the grid.
"""
function plot_goal_prior(agent; save_path=nothing)
    probs = params(agent.goal)[1]
    grid_size = grid_params.grid_size
    
    # Reshape to grid
    goal_grid = reshape(probs, grid_size, grid_size)
    
    p = heatmap(1:grid_size, 1:grid_size, goal_grid',
        xlabel="X Position",
        ylabel="Y Position",
        title="Goal Prior: P(s_goal)",
        color=:Greens,
        aspect_ratio=:equal,
        clims=(0, 1)
    )
    
    # Mark goal
    scatter!(p, [grid_params.goal[1]], [grid_params.goal[2]],
        marker=:star, markersize=15, markerstrokecolor=:red, 
        markerstrokewidth=2, color=:yellow, label="Goal")
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved goal prior plot to $save_path")
    end
    
    return p
end

"""
    plot_belief_entropy(belief_history::Vector; save_path=nothing)

Plot entropy of state beliefs over time.
"""
function plot_belief_entropy(belief_history::Vector; save_path=nothing)
    entropies = Float64[]
    
    for belief in belief_history
        probs = params(belief)[1]
        # Compute entropy
        H = -sum(p > 0 ? p * log(p) : 0.0 for p in probs)
        push!(entropies, H)
    end
    
    p = plot(1:length(entropies), entropies,
        xlabel="Time Step",
        ylabel="Entropy (nats)",
        title="State Belief Entropy Over Time",
        linewidth=2,
        marker=:circle,
        color=:purple,
        legend=false
    )
    
    # Add reference lines
    max_entropy = log(grid_params.grid_size^2)
    hline!(p, [max_entropy], linestyle=:dash, color=:gray, 
        label="Max Entropy (uniform)")
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved belief entropy plot to $save_path")
    end
    
    return p
end

export plot_state_belief, plot_policy, plot_goal_prior, plot_belief_entropy
