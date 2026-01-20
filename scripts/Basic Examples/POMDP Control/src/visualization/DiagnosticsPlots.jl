# Diagnostics Visualization for Learning and Inference

using Plots

"""
    LearningHistory

Records learning statistics across experiments.
"""
mutable struct LearningHistory
    experiment_ids::Vector{Int}
    success::Vector{Bool}
    steps_taken::Vector{Int}
    A_entropy::Vector{Float64}
    B_entropy::Vector{Float64}
    cumulative_success_rate::Vector{Float64}
end

LearningHistory() = LearningHistory([], [], [], [], [], [])

"""
    record_experiment!(history, exp_id, success, steps, agent)

Record results of a single experiment.
"""
function record_experiment!(history::LearningHistory, exp_id::Int, 
                            success::Bool, steps::Int, agent)
    push!(history.experiment_ids, exp_id)
    push!(history.success, success)
    push!(history.steps_taken, steps)
    
    # Compute matrix entropies
    A = mean(agent.p_A)
    B = mean(agent.p_B)
    
    A_ent = compute_matrix_entropy(A)
    B_ent = compute_matrix_entropy(B)
    
    push!(history.A_entropy, A_ent)
    push!(history.B_entropy, B_ent)
    
    # Cumulative success rate
    rate = sum(history.success) / length(history.success)
    push!(history.cumulative_success_rate, rate)
end

"""
    compute_matrix_entropy(M)

Compute average entropy of a stochastic matrix (row-wise).
"""
function compute_matrix_entropy(M::AbstractMatrix)
    n_rows = size(M, 1)
    total_ent = 0.0
    for i in 1:n_rows
        row = M[i, :]
        ent = -sum(p > 0 ? p * log(p) : 0.0 for p in row)
        total_ent += ent
    end
    return total_ent / n_rows
end

"""
    compute_matrix_entropy(M::AbstractArray{T,3}) where T

Compute average entropy of a 3D transition matrix (averaged over actions).
"""
function compute_matrix_entropy(M::AbstractArray{T,3}) where T
    n_actions = size(M, 3)
    total_ent = 0.0
    for a in 1:n_actions
        total_ent += compute_matrix_entropy(M[:, :, a])
    end
    return total_ent / n_actions
end

"""
    plot_learning_curve(history::LearningHistory; save_path=nothing)

Plot cumulative success rate over experiments.
"""
function plot_learning_curve(history::LearningHistory; save_path=nothing)
    p = plot(history.experiment_ids, history.cumulative_success_rate .* 100,
        xlabel="Experiment Number",
        ylabel="Success Rate (%)",
        title="Learning Curve",
        linewidth=2,
        color=:blue,
        legend=false,
        ylims=(0, 100)
    )
    
    # Add 50% reference line
    hline!(p, [50], linestyle=:dash, color=:gray, label="50%")
    
    # Shade success region
    hspan!(p, [50, 100], alpha=0.1, color=:green, label=nothing)
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved learning curve to $save_path")
    end
    
    return p
end

"""
    plot_matrix_entropy_evolution(history::LearningHistory; save_path=nothing)

Plot how matrix entropies evolve during learning.
"""
function plot_matrix_entropy_evolution(history::LearningHistory; save_path=nothing)
    p = plot(history.experiment_ids, history.A_entropy,
        xlabel="Experiment Number",
        ylabel="Average Entropy (nats)",
        title="Matrix Entropy Evolution",
        label="A (observation)",
        linewidth=2,
        color=:blue
    )
    
    plot!(p, history.experiment_ids, history.B_entropy,
        label="B (transition)",
        linewidth=2,
        color=:red
    )
    
    # Add max entropy reference
    n_states = grid_params.grid_size^2
    max_ent = log(n_states)
    hline!(p, [max_ent], linestyle=:dash, color=:gray, label="Max entropy")
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved entropy evolution to $save_path")
    end
    
    return p
end

"""
    plot_steps_histogram(history::LearningHistory; save_path=nothing)

Histogram of steps taken per experiment.
"""
function plot_steps_histogram(history::LearningHistory; save_path=nothing)
    p = histogram(history.steps_taken,
        xlabel="Steps Taken",
        ylabel="Frequency",
        title="Steps Distribution",
        bins=1:maximum(history.steps_taken)+1,
        color=:steelblue,
        legend=false
    )
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved steps histogram to $save_path")
    end
    
    return p
end

"""
    plot_success_by_experiment(history::LearningHistory; save_path=nothing)

Binary success/failure plot for each experiment.
"""
function plot_success_by_experiment(history::LearningHistory; save_path=nothing)
    colors = [s ? :green : :red for s in history.success]
    
    p = scatter(history.experiment_ids, ones(length(history.success)),
        xlabel="Experiment Number",
        ylabel="",
        title="Success/Failure by Experiment",
        marker=:square,
        markersize=6,
        color=colors,
        yticks=[],
        legend=false
    )
    
    if !isnothing(save_path)
        savefig(p, save_path)
        rxlog("info", "Saved success by experiment to $save_path")
    end
    
    return p
end

"""
    plot_diagnostics_dashboard(history::LearningHistory; save_path=nothing)

Create comprehensive diagnostics dashboard.
"""
function plot_diagnostics_dashboard(history::LearningHistory; save_path=nothing)
    p1 = plot_learning_curve(history)
    p2 = plot_matrix_entropy_evolution(history)
    p3 = plot_steps_histogram(history)
    p4 = plot_success_by_experiment(history)
    
    combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
    
    if !isnothing(save_path)
        savefig(combined, save_path)
        rxlog("info", "Saved diagnostics dashboard to $save_path")
    end
    
    return combined
end

export LearningHistory, record_experiment!, compute_matrix_entropy
export plot_learning_curve, plot_matrix_entropy_evolution
export plot_steps_histogram, plot_success_by_experiment, plot_diagnostics_dashboard
