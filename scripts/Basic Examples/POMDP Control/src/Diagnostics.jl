# Diagnostics Module for POMDP Control
# Real-time statistical tracking and analysis

using Statistics
using Distributions

# ============================================================================
# DIAGNOSTICS TRACKER
# ============================================================================

"""
    DiagnosticsTracker

Comprehensive tracking of all metrics during POMDP experiments.

Fields are organized by temporal scope:
- Step-level: per-step metrics within an experiment
- Experiment-level: per-experiment summaries
- Replicate-level: per-replicate aggregates
- Matrix-level: model parameter evolution
"""
mutable struct DiagnosticsTracker
    # ─────────────────────────────────────────────────────────────────────────
    # Step-level metrics (reset each experiment)
    # ─────────────────────────────────────────────────────────────────────────
    step_times::Vector{Float64}           # Inference time per step (ms)
    action_history::Vector{Int}           # Actions taken
    observation_history::Vector{Tuple{Int,Int}}  # Positions observed
    belief_entropies::Vector{Float64}     # State belief entropy
    policy_entropies::Vector{Float64}     # Policy entropy
    policy_probs::Vector{Vector{Float64}} # Full policy probabilities
    
    # ─────────────────────────────────────────────────────────────────────────
    # Experiment-level metrics (reset each replicate)
    # ─────────────────────────────────────────────────────────────────────────
    experiment_success::Vector{Bool}      # Success per experiment
    experiment_steps::Vector{Int}         # Steps taken per experiment
    experiment_final_pos::Vector{Tuple{Int,Int}}  # Final positions
    experiment_times::Vector{Float64}     # Total time per experiment (ms)
    cumulative_success::Vector{Float64}   # Running success rate
    
    # ─────────────────────────────────────────────────────────────────────────
    # Matrix evolution metrics (across experiments)
    # ─────────────────────────────────────────────────────────────────────────
    A_entropies::Vector{Float64}          # A matrix entropy over time
    B_entropies::Vector{Float64}          # B matrix entropy over time
    A_diagonality::Vector{Float64}        # A matrix diagonality score
    B_transition_confidence::Vector{Float64}  # B matrix confidence
    
    # ─────────────────────────────────────────────────────────────────────────
    # Replicate-level aggregates
    # ─────────────────────────────────────────────────────────────────────────
    replicate_success_rates::Vector{Float64}
    replicate_convergence_exp::Vector{Int}
    replicate_mean_steps::Vector{Float64}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Current context
    # ─────────────────────────────────────────────────────────────────────────
    current_replicate::Int
    current_experiment::Int
    current_step::Int
end

"""Create a new DiagnosticsTracker."""
function DiagnosticsTracker()
    return DiagnosticsTracker(
        # Step-level
        Float64[], Int[], Tuple{Int,Int}[], Float64[], Float64[], Vector{Float64}[],
        # Experiment-level
        Bool[], Int[], Tuple{Int,Int}[], Float64[], Float64[],
        # Matrix evolution
        Float64[], Float64[], Float64[], Float64[],
        # Replicate-level
        Float64[], Int[], Float64[],
        # Context
        0, 0, 0
    )
end

# ============================================================================
# ENTROPY CALCULATIONS
# ============================================================================

"""Calculate entropy of a probability distribution."""
function distribution_entropy(probs::Vector{Float64})
    h = 0.0
    for p in probs
        if p > 1e-10
            h -= p * log(p)
        end
    end
    return h
end

"""Calculate entropy of a Categorical distribution."""
function distribution_entropy(dist::Categorical)
    return distribution_entropy(params(dist)[1])
end

"""Calculate average entropy of a matrix (DirichletCollection)."""
function matrix_entropy(matrix_dist)
    m = mean(matrix_dist)
    total_entropy = 0.0
    n = size(m, 2)
    for col in 1:n
        total_entropy += distribution_entropy(m[:, col])
    end
    return total_entropy / n
end

"""Calculate diagonality score of A matrix (how identity-like)."""
function matrix_diagonality(matrix_dist)
    m = mean(matrix_dist)
    n = min(size(m)...)
    diag_sum = sum(m[i,i] for i in 1:n)
    return diag_sum / n
end

"""Calculate transition confidence (how peaked B is)."""
function transition_confidence(B_dist)
    m = mean(B_dist)
    n_states, _, n_actions = size(m)
    max_probs = 0.0
    count = 0
    for a in 1:n_actions, s in 1:n_states
        max_probs += maximum(m[:, s, a])
        count += 1
    end
    return max_probs / count
end

# ============================================================================
# RECORDING FUNCTIONS
# ============================================================================

"""Reset step-level metrics for new experiment."""
function reset_step_metrics!(tracker::DiagnosticsTracker)
    empty!(tracker.step_times)
    empty!(tracker.action_history)
    empty!(tracker.observation_history)
    empty!(tracker.belief_entropies)
    empty!(tracker.policy_entropies)
    empty!(tracker.policy_probs)
end

"""Reset experiment-level metrics for new replicate."""
function reset_experiment_metrics!(tracker::DiagnosticsTracker)
    empty!(tracker.experiment_success)
    empty!(tracker.experiment_steps)
    empty!(tracker.experiment_final_pos)
    empty!(tracker.experiment_times)
    empty!(tracker.cumulative_success)
    empty!(tracker.A_entropies)
    empty!(tracker.B_entropies)
    empty!(tracker.A_diagonality)
    empty!(tracker.B_transition_confidence)
end

"""Record step metrics."""
function record_step!(tracker::DiagnosticsTracker;
    step_time::Float64=0.0,
    action::Int=0,
    observation::Tuple{Int,Int}=(0,0),
    belief_entropy::Float64=0.0,
    policy_entropy::Float64=0.0,
    policy_probs::Vector{Float64}=Float64[]
)
    tracker.current_step += 1
    push!(tracker.step_times, step_time)
    push!(tracker.action_history, action)
    push!(tracker.observation_history, observation)
    push!(tracker.belief_entropies, belief_entropy)
    push!(tracker.policy_entropies, policy_entropy)
    if !isempty(policy_probs)
        push!(tracker.policy_probs, copy(policy_probs))
    end
end

"""Record experiment summary."""
function record_experiment!(tracker::DiagnosticsTracker;
    success::Bool,
    steps::Int,
    final_pos::Tuple{Int,Int},
    exp_time::Float64,
    A_dist=nothing,
    B_dist=nothing
)
    tracker.current_experiment += 1
    push!(tracker.experiment_success, success)
    push!(tracker.experiment_steps, steps)
    push!(tracker.experiment_final_pos, final_pos)
    push!(tracker.experiment_times, exp_time)
    
    # Update cumulative success rate
    cum_rate = mean(tracker.experiment_success)
    push!(tracker.cumulative_success, cum_rate)
    
    # Matrix evolution
    if !isnothing(A_dist)
        push!(tracker.A_entropies, matrix_entropy(A_dist))
        push!(tracker.A_diagonality, matrix_diagonality(A_dist))
    end
    if !isnothing(B_dist)
        push!(tracker.B_entropies, matrix_entropy(B_dist))
        push!(tracker.B_transition_confidence, transition_confidence(B_dist))
    end
end

"""Record replicate summary."""
function record_replicate!(tracker::DiagnosticsTracker)
    tracker.current_replicate += 1
    
    success_rate = mean(tracker.experiment_success)
    push!(tracker.replicate_success_rates, success_rate)
    
    # Find convergence point (first experiment where cumulative > 50%)
    conv = findfirst(r -> r > 0.5, tracker.cumulative_success)
    push!(tracker.replicate_convergence_exp, conv !== nothing ? conv : length(tracker.experiment_success))
    
    push!(tracker.replicate_mean_steps, mean(tracker.experiment_steps))
end

# ============================================================================
# STATISTICAL SUMMARIES
# ============================================================================

"""Calculate comprehensive statistics for the tracker."""
function compute_statistics(tracker::DiagnosticsTracker)
    n_reps = length(tracker.replicate_success_rates)
    
    stats = Dict{String, Any}(
        "n_replicates" => n_reps,
        "n_experiments_per_rep" => length(tracker.experiment_success),
        
        # Success rate statistics
        "mean_success_rate" => n_reps > 0 ? mean(tracker.replicate_success_rates) : 0.0,
        "std_success_rate" => n_reps > 1 ? std(tracker.replicate_success_rates) : 0.0,
        "min_success_rate" => n_reps > 0 ? minimum(tracker.replicate_success_rates) : 0.0,
        "max_success_rate" => n_reps > 0 ? maximum(tracker.replicate_success_rates) : 0.0,
        
        # Convergence statistics
        "mean_convergence" => n_reps > 0 ? mean(tracker.replicate_convergence_exp) : 0.0,
        "std_convergence" => n_reps > 1 ? std(tracker.replicate_convergence_exp) : 0.0,
        
        # Step statistics
        "mean_steps" => n_reps > 0 ? mean(tracker.replicate_mean_steps) : 0.0,
        "std_steps" => n_reps > 1 ? std(tracker.replicate_mean_steps) : 0.0,
        
        # Matrix evolution (from last replicate)
        "final_A_entropy" => !isempty(tracker.A_entropies) ? tracker.A_entropies[end] : 0.0,
        "final_B_entropy" => !isempty(tracker.B_entropies) ? tracker.B_entropies[end] : 0.0,
        "final_A_diagonality" => !isempty(tracker.A_diagonality) ? tracker.A_diagonality[end] : 0.0,
        "final_B_confidence" => !isempty(tracker.B_transition_confidence) ? tracker.B_transition_confidence[end] : 0.0
    )
    
    # 95% CI
    if n_reps > 1
        sem = stats["std_success_rate"] / sqrt(n_reps)
        stats["ci_95_success_rate"] = 1.96 * sem
    else
        stats["ci_95_success_rate"] = 0.0
    end
    
    return stats
end

"""Print formatted statistics summary."""
function print_statistics(tracker::DiagnosticsTracker)
    stats = compute_statistics(tracker)
    
    println("┌" * "─"^68 * "┐")
    println("│  DIAGNOSTICS SUMMARY" * " "^47 * "│")
    println("├" * "─"^68 * "┤")
    println("│  Success Rate: $(round(stats["mean_success_rate"]*100, digits=1))% ± $(round(stats["ci_95_success_rate"]*100, digits=1))%" * " "^30 * "│")
    println("│  Convergence: exp $(round(stats["mean_convergence"], digits=1))" * " "^45 * "│")
    println("│  Mean Steps: $(round(stats["mean_steps"], digits=2))" * " "^48 * "│")
    println("├" * "─"^68 * "┤")
    println("│  A entropy: $(round(stats["final_A_entropy"], digits=3))  diagonality: $(round(stats["final_A_diagonality"], digits=3))" * " "^20 * "│")
    println("│  B entropy: $(round(stats["final_B_entropy"], digits=3))  confidence: $(round(stats["final_B_confidence"], digits=3))" * " "^21 * "│")
    println("└" * "─"^68 * "┘")
end

export DiagnosticsTracker
export reset_step_metrics!, reset_experiment_metrics!
export record_step!, record_experiment!, record_replicate!
export distribution_entropy, matrix_entropy, matrix_diagonality, transition_confidence
export compute_statistics, print_statistics
