# Animation Generation for POMDP Control
# Comprehensive animations for all variables through time

using Plots

# ============================================================================
# TRAJECTORY ANIMATIONS
# ============================================================================

"""
    animate_trajectory(positions::Vector{Tuple{Int,Int}}; 
                       fps=2, save_path=nothing)

Create an animated GIF of agent trajectory.
"""
function animate_trajectory(positions::Vector{Tuple{Int,Int}}; 
                            fps::Int=2, save_path=nothing)
    grid_size = grid_params.grid_size
    
    anim = @animate for t in 1:length(positions)
        # Create grid background
        p = heatmap(1:grid_size, 1:grid_size, zeros(grid_size, grid_size)',
            color=:grays,
            clims=(0, 1),
            colorbar=false,
            xlabel="X Position",
            ylabel="Y Position",
            title="Agent Trajectory: Step $t / $(length(positions))",
            aspect_ratio=:equal,
            xlims=(0.5, grid_size + 0.5),
            ylims=(0.5, grid_size + 0.5)
        )
        
        # Add wind indicators
        for x in 1:grid_size
            wind_strength = grid_params.wind[x]
            if wind_strength > 0
                for y in 1:grid_size
                    annotate!(p, x, y, text("↑"^wind_strength, 6, :lightgray))
                end
            end
        end
        
        # Plot trajectory so far
        xs = [pos[1] for pos in positions[1:t]]
        ys = [pos[2] for pos in positions[1:t]]
        
        if t > 1
            plot!(p, xs, ys, linewidth=3, color=:blue, label=nothing, alpha=0.5)
        end
        
        # Mark start and goal
        scatter!(p, [grid_params.start[1]], [grid_params.start[2]],
            marker=:circle, markersize=12, color=:green, label="Start")
        scatter!(p, [grid_params.goal[1]], [grid_params.goal[2]],
            marker=:star, markersize=15, color=:red, label="Goal")
        
        # Current position
        scatter!(p, [positions[t][1]], [positions[t][2]],
            marker=:diamond, markersize=18, color=:purple, label="Agent")
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved trajectory animation to $save_path")
    end
    
    return anim
end

"""
    animate_belief_evolution(beliefs::Vector, positions::Vector{Tuple{Int,Int}};
                             fps=2, save_path=nothing)

Animate state belief evolution alongside trajectory.
"""
function animate_belief_evolution(beliefs::Vector, positions::Vector{Tuple{Int,Int}};
                                  fps::Int=2, save_path=nothing)
    grid_size = grid_params.grid_size
    
    anim = @animate for t in 1:length(beliefs)
        # Get belief probabilities
        probs = params(beliefs[t])[1]
        belief_grid = reshape(probs, grid_size, grid_size)
        
        # Belief heatmap
        p1 = heatmap(1:grid_size, 1:grid_size, belief_grid',
            xlabel="X",
            ylabel="Y",
            title="State Belief (Step $t)",
            color=:YlOrRd,
            aspect_ratio=:equal,
            clims=(0, 1)
        )
        
        scatter!(p1, [positions[t][1]], [positions[t][2]],
            marker=:diamond, markersize=10, color=:blue, label="Agent")
        scatter!(p1, [grid_params.goal[1]], [grid_params.goal[2]],
            marker=:star, markersize=12, color=:green, label="Goal")
        
        # Trajectory
        xs = [pos[1] for pos in positions[1:t]]
        ys = [pos[2] for pos in positions[1:t]]
        
        p2 = heatmap(1:grid_size, 1:grid_size, zeros(grid_size, grid_size)',
            color=:grays,
            clims=(0, 1),
            colorbar=false,
            xlabel="X",
            ylabel="Y",
            title="Trajectory",
            aspect_ratio=:equal
        )
        
        if t > 1
            plot!(p2, xs, ys, linewidth=3, color=:blue, label=nothing)
        end
        
        scatter!(p2, [xs[end]], [ys[end]],
            marker=:diamond, markersize=12, color=:purple, label="Agent")
        scatter!(p2, [grid_params.goal[1]], [grid_params.goal[2]],
            marker=:star, markersize=12, color=:red, label="Goal")
        
        plot(p1, p2, layout=(1, 2), size=(900, 400))
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved belief evolution animation to $save_path")
    end
    
    return anim
end

# ============================================================================
# LEARNING PROGRESS ANIMATIONS
# ============================================================================

"""
    animate_learning_progress(history::LearningHistory; 
                              fps=5, save_path=nothing)

Animate learning curve progression over experiments.
"""
function animate_learning_progress(history::LearningHistory; 
                                   fps::Int=5, save_path=nothing)
    n = length(history.experiment_ids)
    
    anim = @animate for t in 1:n
        p = plot(history.experiment_ids[1:t], 
                 history.cumulative_success_rate[1:t] .* 100,
            xlabel="Experiment Number",
            ylabel="Success Rate (%)",
            title="Learning Progress (Experiment $t / $n)",
            linewidth=2,
            color=:blue,
            legend=false,
            xlims=(1, n),
            ylims=(0, 100)
        )
        
        # Reference line
        hline!(p, [50], linestyle=:dash, color=:gray)
        
        # Current success rate annotation
        rate = history.cumulative_success_rate[t] * 100
        annotate!(p, t, rate + 5, text("$(round(rate, digits=1))%", 8, :blue))
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved learning progress animation to $save_path")
    end
    
    return anim
end

# ============================================================================
# MATRIX EVOLUTION ANIMATIONS
# ============================================================================

"""
    animate_matrix_evolution(A_history::Vector, B_history::Vector;
                             fps=3, save_path=nothing)

Animate how A and B matrices evolve during learning.
"""
function animate_matrix_evolution(A_history::Vector, B_history::Vector;
                                  fps::Int=3, save_path=nothing)
    n = length(A_history)
    n_states = grid_params.grid_size^2
    
    anim = @animate for t in 1:n
        A = mean(A_history[t])
        B = mean(B_history[t])
        
        p1 = heatmap(1:n_states, 1:n_states, A,
            title="A Matrix (Exp $t)",
            color=:viridis,
            aspect_ratio=:equal,
            clims=(0, 0.5),
            colorbar=false
        )
        
        # Just show B for action "right" as example
        B_right = B[:, :, ACTION_RIGHT]
        p2 = heatmap(1:n_states, 1:n_states, B_right,
            title="B[right] Matrix (Exp $t)",
            color=:plasma,
            aspect_ratio=:equal,
            clims=(0, 0.5),
            colorbar=false
        )
        
        plot(p1, p2, layout=(1, 2), size=(800, 400))
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved matrix evolution animation to $save_path")
    end
    
    return anim
end

# ============================================================================
# POLICY EVOLUTION ANIMATIONS
# ============================================================================

"""
    animate_policy_evolution(policies::Vector; fps=3, save_path=nothing)

Animate how policy probabilities evolve over steps.
"""
function animate_policy_evolution(policies::Vector{Vector{Float64}}; 
                                  fps::Int=3, save_path=nothing)
    n = length(policies)
    
    anim = @animate for t in 1:n
        probs = policies[t]
        
        p = bar(["Up", "Right", "Down", "Left"], probs .* 100,
            xlabel="Action",
            ylabel="Probability (%)",
            title="Policy Distribution (Step $t / $n)",
            color=[:blue, :green, :orange, :red],
            legend=false,
            ylims=(0, 100)
        )
        
        # Annotate with values
        for (i, prob) in enumerate(probs)
            annotate!(p, i, prob * 100 + 3, text("$(round(prob*100, digits=1))%", 8))
        end
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved policy evolution animation to $save_path")
    end
    
    return anim
end

# ============================================================================
# ENTROPY EVOLUTION ANIMATIONS
# ============================================================================

"""
    animate_entropy_evolution(belief_entropies::Vector{Float64}, 
                              policy_entropies::Vector{Float64};
                              fps=5, save_path=nothing)

Animate how entropies change through time.
"""
function animate_entropy_evolution(belief_entropies::Vector{Float64}, 
                                   policy_entropies::Vector{Float64};
                                   fps::Int=5, save_path=nothing)
    n = length(belief_entropies)
    max_entropy = max(maximum(belief_entropies), maximum(policy_entropies), 2.0)
    
    anim = @animate for t in 1:n
        p = plot(1:t, belief_entropies[1:t],
            xlabel="Step",
            ylabel="Entropy (nats)",
            title="Entropy Evolution (Step $t / $n)",
            label="Belief",
            linewidth=2,
            color=:blue,
            xlims=(1, n),
            ylims=(0, max_entropy * 1.1)
        )
        
        plot!(p, 1:t, policy_entropies[1:t],
            label="Policy",
            linewidth=2,
            color=:orange
        )
        
        # Current values
        scatter!(p, [t], [belief_entropies[t]], color=:blue, label=nothing, markersize=6)
        scatter!(p, [t], [policy_entropies[t]], color=:orange, label=nothing, markersize=6)
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved entropy evolution animation to $save_path")
    end
    
    return anim
end

# ============================================================================
# ACTION DISTRIBUTION THROUGH EXPERIMENTS
# ============================================================================

"""
    animate_action_distribution(action_counts_history::Vector{Vector{Int}};
                                fps=5, save_path=nothing)

Animate how action distribution changes across experiments.
"""
function animate_action_distribution(action_counts_history::Vector{Vector{Int}};
                                     fps::Int=5, save_path=nothing)
    n = length(action_counts_history)
    
    anim = @animate for t in 1:n
        counts = action_counts_history[t]
        total = sum(counts)
        freqs = total > 0 ? counts ./ total .* 100 : zeros(4)
        
        p = bar(["Up", "Right", "Down", "Left"], freqs,
            xlabel="Action",
            ylabel="Frequency (%)",
            title="Action Distribution (Exp 1-$t)",
            color=[:blue, :green, :orange, :red],
            legend=false,
            ylims=(0, 60)
        )
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved action distribution animation to $save_path")
    end
    
    return anim
end

# ============================================================================
# COMPREHENSIVE DASHBOARD ANIMATION
# ============================================================================

"""
    animate_dashboard(history::LearningHistory, recorder::TrajectoryRecorder;
                      fps=5, save_path=nothing)

Animate a comprehensive dashboard with multiple metrics.
"""
function animate_dashboard(history::LearningHistory, recorder::TrajectoryRecorder;
                           fps::Int=5, save_path=nothing)
    n = min(length(history.experiment_ids), 50)  # Limit for animation
    
    anim = @animate for t in 1:n
        # Learning curve
        p1 = plot(history.experiment_ids[1:t], 
                  history.cumulative_success_rate[1:t] .* 100,
            xlabel="Experiment",
            ylabel="Success (%)",
            title="Learning Curve",
            linewidth=2,
            color=:blue,
            legend=false,
            xlims=(1, n),
            ylims=(0, 100)
        )
        hline!(p1, [50], linestyle=:dash, color=:gray)
        
        # A matrix entropy
        p2 = plot(history.experiment_ids[1:t], 
                  history.A_entropy[1:t],
            xlabel="Experiment",
            ylabel="Entropy",
            title="A Matrix Entropy",
            linewidth=2,
            color=:green,
            legend=false,
            xlims=(1, n)
        )
        
        # B matrix entropy
        p3 = plot(history.experiment_ids[1:t], 
                  history.B_entropy[1:t],
            xlabel="Experiment",
            ylabel="Entropy",
            title="B Matrix Entropy",
            linewidth=2,
            color=:orange,
            legend=false,
            xlims=(1, n)
        )
        
        # Steps histogram (cumulative)
        steps = history.steps_taken[1:t]
        p4 = histogram(steps,
            xlabel="Steps",
            ylabel="Count",
            title="Steps Distribution",
            bins=1:5,
            color=:purple,
            legend=false
        )
        
        plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700))
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved dashboard animation to $save_path")
    end
    
    return anim
end

# ============================================================================
# A/B MATRIX GRID EVOLUTION (as grid, not heatmap)
# ============================================================================

"""
    animate_A_grid_evolution(A_history::Vector; fps=3, save_path=nothing)

Animate A matrix evolution as a grid view.
"""
function animate_A_grid_evolution(A_history::Vector; fps::Int=3, save_path=nothing)
    grid_size = grid_params.grid_size
    
    anim = @animate for t in 1:length(A_history)
        A = mean(A_history[t])
        
        # For each state, show estimated position distribution
        plots_arr = []
        for s in 1:min(9, grid_size^2)  # Show first 9 states
            s_probs = A[:, s]
            s_grid = reshape(s_probs, grid_size, grid_size)
            p = heatmap(s_grid', 
                title="State $s", 
                color=:YlOrRd, 
                clims=(0, 1),
                colorbar=false,
                axis=false)
            push!(plots_arr, p)
        end
        
        plot(plots_arr..., layout=(3, 3), size=(600, 600), 
             plot_title="A Matrix Grid (Exp $t / $(length(A_history)))")
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved A grid evolution animation to $save_path")
    end
    
    return anim
end

"""
    animate_B_action_evolution(B_history::Vector, action_idx::Int; 
                               fps=3, save_path=nothing)

Animate B matrix evolution for a specific action as grid view.
"""
function animate_B_action_evolution(B_history::Vector, action_idx::Int; 
                                    fps::Int=3, save_path=nothing)
    grid_size = grid_params.grid_size
    action_name = ACTION_NAMES[action_idx]
    
    anim = @animate for t in 1:length(B_history)
        B = mean(B_history[t])
        B_action = B[:, :, action_idx]
        
        # Show transition grid for sample states
        plots_arr = []
        for s in 1:min(9, grid_size^2)
            s_probs = B_action[:, s]
            s_grid = reshape(s_probs, grid_size, grid_size)
            p = heatmap(s_grid', 
                title="From S$s", 
                color=:plasma, 
                clims=(0, 1),
                colorbar=false,
                axis=false)
            push!(plots_arr, p)
        end
        
        plot(plots_arr..., layout=(3, 3), size=(600, 600), 
             plot_title="B[$action_name] Transitions (Exp $t)")
    end
    
    if !isnothing(save_path)
        gif(anim, save_path, fps=fps)
        rxlog("info", "Saved B[$action_name] evolution animation to $save_path")
    end
    
    return anim
end

export animate_trajectory, animate_belief_evolution
export animate_learning_progress, animate_matrix_evolution
export animate_policy_evolution, animate_entropy_evolution
export animate_action_distribution, animate_dashboard
export animate_A_grid_evolution, animate_B_action_evolution
