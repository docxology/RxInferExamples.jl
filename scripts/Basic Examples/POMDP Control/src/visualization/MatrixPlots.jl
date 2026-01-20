# Matrix Visualization for A (observation) and B (transition) matrices

using Plots

"""
    plot_A_matrix(agent; save_path=nothing)

Visualize the observation likelihood matrix A = P(observation|state).
Rows: observations, Columns: states
"""
function plot_A_matrix(agent; save_path=nothing)
    A = mean(agent.p_A)  # Expected observation matrix
    n_states = grid_params.grid_size^2
    
    p = heatmap(1:n_states, 1:n_states, A,
        xlabel="Hidden State Index",
        ylabel="Observation Index",
        title="Observation Matrix A: P(o|s)",
        color=:viridis,
        aspect_ratio=:equal,
        clims=(0, 1)
    )
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved A matrix plot to $save_path")
    end
    
    return p
end

"""
    plot_B_matrix(agent, action_idx::Int; save_path=nothing)

Visualize the transition matrix B for a specific action.
B[:,:,action] = P(s'|s, action)
"""
function plot_B_matrix(agent, action_idx::Int; save_path=nothing)
    B = mean(agent.p_B)  # Expected transition matrix
    B_action = B[:, :, action_idx]
    n_states = grid_params.grid_size^2
    action_name = ACTION_NAMES[action_idx]
    
    p = heatmap(1:n_states, 1:n_states, B_action,
        xlabel="Current State Index",
        ylabel="Next State Index",
        title="Transition Matrix B[$(action_name)]: P(s'|s, $(action_name))",
        color=:plasma,
        aspect_ratio=:equal,
        clims=(0, 1)
    )
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved B[$action_name] matrix plot to $save_path")
    end
    
    return p
end

"""
    plot_all_B_matrices(agent; save_dir=nothing)

Visualize all 4 action-conditioned transition matrices.
"""
function plot_all_B_matrices(agent; save_dir=nothing)
    plots = []
    for action_idx in 1:4
        p = plot_B_matrix(agent, action_idx)
        push!(plots, p)
    end
    
    combined = plot(plots..., layout=(2, 2), size=(1000, 1000))
    
    if !isnothing(save_dir)
        save_path = joinpath(save_dir, "B_matrices_all.png")
        savefig(combined, save_path)
        rxlog("info", "Saved combined B matrices to $save_path")
    end
    
    return combined
end

"""
    plot_A_matrix_as_grid(agent; save_path=nothing)

Visualize observation matrix as grid of states with observation probabilities.
"""
function plot_A_matrix_as_grid(agent; save_path=nothing)
    A = mean(agent.p_A)
    grid_size = grid_params.grid_size
    
    # Extract diagonal (identity learning)
    diag_strength = diag(A)
    grid_diag = reshape(diag_strength, grid_size, grid_size)
    
    p = heatmap(1:grid_size, 1:grid_size, grid_diag',
        xlabel="X Position",
        ylabel="Y Position",
        title="Observation Certainty (A diagonal)",
        color=:YlGnBu,
        aspect_ratio=:equal,
        clims=(0, 1)
    )
    
    # Mark goal
    scatter!(p, [grid_params.goal[1]], [grid_params.goal[2]],
        marker=:star, markersize=15, color=:red, label="Goal")
    
    if !isnothing(save_path)
        savefig(p, save_path)
    end
    
    return p
end

"""
    plot_B_matrix_as_arrows(agent, action_idx::Int; save_path=nothing)

Visualize transition probabilities as arrow field on grid.
"""
function plot_B_matrix_as_arrows(agent, action_idx::Int; save_path=nothing)
    B = mean(agent.p_B)
    B_action = B[:, :, action_idx]
    grid_size = grid_params.grid_size
    action_name = ACTION_NAMES[action_idx]
    
    # Compute expected transitions
    xs, ys, us, vs = Float64[], Float64[], Float64[], Float64[]
    
    for state_idx in 1:(grid_size^2)
        x, y = index_to_grid_location(state_idx)
        
        # Expected next position
        next_probs = B_action[:, state_idx]
        expected_x, expected_y = 0.0, 0.0
        
        for next_idx in 1:(grid_size^2)
            nx, ny = index_to_grid_location(next_idx)
            expected_x += next_probs[next_idx] * nx
            expected_y += next_probs[next_idx] * ny
        end
        
        push!(xs, Float64(x))
        push!(ys, Float64(y))
        push!(us, expected_x - x)
        push!(vs, expected_y - y)
    end
    
    p = quiver(xs, ys, quiver=(us, vs),
        xlabel="X Position",
        ylabel="Y Position",
        title="Learned Transitions: $(action_name)",
        xlims=(0, grid_size + 1),
        ylims=(0, grid_size + 1),
        aspect_ratio=:equal,
        arrow=arrow(:closed, 0.2)
    )
    
    # Mark goal
    scatter!(p, [grid_params.goal[1]], [grid_params.goal[2]],
        marker=:star, markersize=15, color=:red, label="Goal")
    
    if !isnothing(save_path)
        savefig(p, save_path)
    end
    
    return p
end

export plot_A_matrix, plot_B_matrix, plot_all_B_matrices
export plot_A_matrix_as_grid, plot_B_matrix_as_arrows
