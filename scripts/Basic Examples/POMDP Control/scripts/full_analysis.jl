# POMDP Control - Full Comprehensive Analysis
# Includes within-trial and across-trial statistical analysis
# Called from run.jl: julia --project=. run.jl analysis
#
# See: docs/README.md for documentation

const SCRIPT_DIR = @__DIR__
const PROJECT_DIR = dirname(SCRIPT_DIR)

using Pkg
Pkg.activate(PROJECT_DIR)
Pkg.instantiate()

include(joinpath(PROJECT_DIR, "src/PODMPControlApp.jl"))
using .PODMPControlApp
using Distributions
using Statistics
using JSON
using Dates

"""Format bytes into human-readable string."""
function format_bytes(bytes::Integer)
    if bytes < 1024
        return "$(bytes) B"
    elseif bytes < 1024^2
        return "$(round(bytes/1024, digits=1)) KB"
    else
        return "$(round(bytes/1024^2, digits=2)) MB"
    end
end

"""Count files and total size in directory."""
function dir_stats(path::String)
    if !isdir(path)
        return (0, 0)
    end
    files = filter(f -> isfile(joinpath(path, f)), readdir(path))
    total_size = sum(filesize(joinpath(path, f)) for f in files; init=0)
    return (length(files), total_size)
end

println("═"^70)
println("   POMDP CONTROL - COMPREHENSIVE ANALYSIS")
println("   $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
println("═"^70)
println()

# Ensure output directories exist
output_dir = ensure_output_dirs()
# Convert to clean path
output_dir = abspath(joinpath(PROJECT_DIR, "output"))
println("Output Directory: $output_dir")
println()

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================
println("┌" * "─"^68 * "┐")
println("│  CONFIGURATION" * " "^53 * "│")
println("├" * "─"^68 * "┤")
println("│  Environment:" * " "^54 * "│")
println("│    Grid: $(grid_params.grid_size)×$(grid_params.grid_size)  Wind: $(collect(grid_params.wind))" * " "^(32 - length(string(collect(grid_params.wind)))) * "│")
println("│    Start: $(grid_params.start)  →  Goal: $(grid_params.goal)" * " "^35 * "│")
println("├" * "─"^68 * "┤")
println("│  Experiment:" * " "^55 * "│")
println("│    Planning horizon (T): $(experiment_params.planning_horizon)" * " "^41 * "│")
println("│    VMP iterations: $(experiment_params.n_iterations)" * " "^45 * "│")
println("│    Experiments per replicate: $(experiment_params.n_experiments)" * " "^(35 - length(string(experiment_params.n_experiments))) * "│")
println("│    Number of replicates: $(experiment_params.n_replicates)" * " "^(41 - length(string(experiment_params.n_replicates))) * "│")
println("├" * "─"^68 * "┤")
println("│  Priors:" * " "^59 * "│")
println("│    A diagonal bias: $(prior_params.A_diagonal_bias)" * " "^(45 - length(string(prior_params.A_diagonal_bias))) * "│")
println("│    B concentration: $(prior_params.B_concentration)" * " "^(45 - length(string(prior_params.B_concentration))) * "│")
println("└" * "─"^68 * "┘")
println()

# Storage for across-trial analysis
all_histories = LearningHistory[]
all_trajectories = Vector{TrajectoryRecorder}[]
all_final_rates = Float64[]
all_convergence_exp = Int[]
all_mean_steps = Float64[]
# Matrix evolution for animations (from last replicate)
last_A_history = []
last_B_history = []

n_replicates = max(experiment_params.n_replicates, 1)
n_experiments = experiment_params.n_experiments
T = experiment_params.planning_horizon

println("═"^70)
println("   RUNNING $n_replicates REPLICATE(S) × $n_experiments EXPERIMENTS")
println("═"^70)
println()

for rep in 1:n_replicates
    rep_start = time()
    println("┌" * "─"^68 * "┐")
    println("│  REPLICATE $rep / $n_replicates" * " "^(55 - length(string(rep)) - length(string(n_replicates))) * "│")
    println("└" * "─"^68 * "┘")
    
    # Create fresh environment and agent
    env, rx_agent = create_environment()
    agent = PODMPAgent(env, rx_agent)
    
    history = LearningHistory()
    trajectories = TrajectoryRecorder[]
    steps_per_exp = Int[]
    
    # Track matrix evolution for animations
    A_history = []
    B_history = []
    
    for exp in 1:n_experiments
        reset!(agent)
        recorder = TrajectoryRecorder()
        
        success = false
        steps_taken = 0
        
        for t in 1:T
            step_start = time()
            action_idx = mode(first(agent.current_policy))
            goal_reached = step!(agent, T - t)
            step_time = time() - step_start
            
            obs = get_current_position(agent)
            record_step!(recorder, agent, action_idx, obs, step_time)
            steps_taken = t
            
            if goal_reached
                success = true
                break
            end
        end
        
        record_experiment!(history, exp, success, steps_taken, agent)
        push!(trajectories, recorder)
        push!(steps_per_exp, steps_taken)
        
        # Record matrices for evolution animation (every 10 experiments to limit size)
        if exp % 10 == 0 || exp == 1
            push!(A_history, deepcopy(agent.p_A))
            push!(B_history, deepcopy(agent.p_B))
        end
        
        # Progress every 20 experiments
        if exp % 20 == 0 || exp == n_experiments
            rate = history.cumulative_success_rate[end] * 100
            mean_steps = mean(steps_per_exp[max(1, exp-19):exp])
            println("  Exp $(lpad(exp, 4)): $(lpad(round(rate, digits=1), 5))% success, mean steps: $(round(mean_steps, digits=2))")
        end
    end
    
    # Replicate summary
    rep_time = time() - rep_start
    final_rate = history.cumulative_success_rate[end]
    conv_exp = findfirst(r -> r > 0.5, history.cumulative_success_rate)
    mean_steps_all = mean(steps_per_exp)
    
    push!(all_histories, history)
    push!(all_trajectories, trajectories)
    push!(all_final_rates, final_rate)
    push!(all_convergence_exp, conv_exp !== nothing ? conv_exp : n_experiments)
    push!(all_mean_steps, mean_steps_all)
    
    # Store matrix history from last replicate for animations
    global last_A_history = A_history
    global last_B_history = B_history
    
    println()
    println("  ──────────────────────────────────────────────────────────")
    println("  Replicate $rep Summary:")
    println("    Final success rate: $(round(final_rate * 100, digits=1))%")
    println("    Convergence (>50%): experiment $(conv_exp !== nothing ? conv_exp : "not reached")")
    println("    Mean steps to goal: $(round(mean_steps_all, digits=2))")
    println("    Run time: $(round(rep_time, digits=1))s")
    println()
end

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
println("═"^70)
println("   STATISTICAL ANALYSIS")
println("═"^70)
println()

# Across-replicate statistics
mean_rate = mean(all_final_rates) * 100
std_rate = n_replicates > 1 ? std(all_final_rates) * 100 : 0.0
sem_rate = n_replicates > 1 ? std_rate / sqrt(n_replicates) : 0.0
ci_95 = 1.96 * sem_rate

mean_conv = mean(all_convergence_exp)
std_conv = n_replicates > 1 ? std(all_convergence_exp) : 0.0

mean_steps = mean(all_mean_steps)
std_steps = n_replicates > 1 ? std(all_mean_steps) : 0.0

println("┌" * "─"^68 * "┐")
println("│  ACROSS-REPLICATE STATISTICS (n=$n_replicates)" * " "^(37 - length(string(n_replicates))) * "│")
println("├" * "─"^68 * "┤")
println("│  Success Rate:" * " "^53 * "│")
println("│    Mean: $(round(mean_rate, digits=1))% ± $(round(ci_95, digits=1))% (95% CI)" * " "^(30 - length(string(round(mean_rate, digits=1))) - length(string(round(ci_95, digits=1)))) * "│")
if n_replicates > 1
    println("│    Std:  $(round(std_rate, digits=1))%  SEM: $(round(sem_rate, digits=2))%" * " "^(33 - length(string(round(std_rate, digits=1))) - length(string(round(sem_rate, digits=2)))) * "│")
    println("│    Range: $(round(minimum(all_final_rates)*100, digits=1))% - $(round(maximum(all_final_rates)*100, digits=1))%" * " "^(32 - length(string(round(minimum(all_final_rates)*100, digits=1))) - length(string(round(maximum(all_final_rates)*100, digits=1)))) * "│")
end
println("├" * "─"^68 * "┤")
println("│  Convergence (>50% success):" * " "^38 * "│")
println("│    Mean experiment: $(round(mean_conv, digits=1)) ± $(round(std_conv, digits=1))" * " "^(32 - length(string(round(mean_conv, digits=1))) - length(string(round(std_conv, digits=1)))) * "│")
println("├" * "─"^68 * "┤")
println("│  Steps to Goal:" * " "^52 * "│")
println("│    Mean: $(round(mean_steps, digits=2)) ± $(round(std_steps, digits=2))" * " "^(40 - length(string(round(mean_steps, digits=2))) - length(string(round(std_steps, digits=2)))) * "│")
println("└" * "─"^68 * "┘")
println()

# Within-trial learning dynamics
if n_replicates > 1
    println("┌" * "─"^68 * "┐")
    println("│  WITHIN-TRIAL LEARNING DYNAMICS" * " "^35 * "│")
    println("├" * "─"^68 * "┤")
    
    # Aggregate learning curves
    for milestone in [25, 50, 75, 100]
        if milestone <= n_experiments
            rates_at_milestone = [h.cumulative_success_rate[milestone] * 100 for h in all_histories]
            mean_m = mean(rates_at_milestone)
            std_m = std(rates_at_milestone)
            println("│    Exp $(lpad(milestone, 3)): $(lpad(round(mean_m, digits=1), 5))% ± $(round(std_m, digits=1))%" * " "^(32 - length(string(round(std_m, digits=1)))) * "│")
        end
    end
    println("└" * "─"^68 * "┘")
    println()
end

# ============================================================================
# GENERATING VISUALIZATIONS
# ============================================================================
println("═"^70)
println("   GENERATING OUTPUTS")
println("═"^70)
println()

figures_dir = joinpath(output_dir, "figures")
matrices_dir = joinpath(figures_dir, "matrices")
beliefs_dir = joinpath(figures_dir, "beliefs")
trajectories_dir = joinpath(figures_dir, "trajectories")
diagnostics_dir = joinpath(figures_dir, "diagnostics")
animations_dir = joinpath(output_dir, "animations")
data_dir = joinpath(output_dir, "data")

# Use first replicate for detailed visualization
history = all_histories[1]
trajectories_list = all_trajectories[1]
recorder = trajectories_list[end]

# Final agent for visualization
env, rx_agent = create_environment()
agent = PODMPAgent(env, rx_agent)
for _ in 1:n_experiments
    run_experiment!(agent, T)
end

# Matrix visualizations
print("  Matrices...")
plot_A_matrix(agent, save_path=joinpath(matrices_dir, "A_matrix.png"))
plot_all_B_matrices(agent, save_dir=matrices_dir)
plot_A_matrix_as_grid(agent, save_path=joinpath(matrices_dir, "A_grid.png"))
for action_idx in 1:4
    plot_B_matrix_as_arrows(agent, action_idx, 
        save_path=joinpath(matrices_dir, "B_arrows_$(ACTION_NAMES[action_idx]).png"))
end
n_files, total_size = dir_stats(matrices_dir)
println(" ✓ ($n_files files, $(format_bytes(total_size)))")

# Belief visualizations
print("  Beliefs...")
plot_state_belief(agent, save_path=joinpath(beliefs_dir, "state_belief_final.png"))
plot_policy(agent, save_path=joinpath(beliefs_dir, "policy_final.png"))
plot_goal_prior(agent, save_path=joinpath(beliefs_dir, "goal_prior.png"))
n_files, total_size = dir_stats(beliefs_dir)
println(" ✓ ($n_files files, $(format_bytes(total_size)))")

# Trajectory visualizations
print("  Trajectories...")
if !isempty(recorder.positions)
    plot_trajectory(recorder.positions, save_path=joinpath(trajectories_dir, "trajectory_final.png"))
    plot_action_sequence(recorder.actions, save_path=joinpath(trajectories_dir, "actions_final.png"))
    plot_experiment_summary(recorder, save_path=joinpath(trajectories_dir, "experiment_summary.png"))
end
n_files, total_size = dir_stats(trajectories_dir)
println(" ✓ ($n_files files, $(format_bytes(total_size)))")

# Diagnostics visualizations
print("  Diagnostics...")
plot_learning_curve(history, save_path=joinpath(diagnostics_dir, "learning_curve.png"))
plot_matrix_entropy_evolution(history, save_path=joinpath(diagnostics_dir, "entropy_evolution.png"))
plot_steps_histogram(history, save_path=joinpath(diagnostics_dir, "steps_histogram.png"))
plot_success_by_experiment(history, save_path=joinpath(diagnostics_dir, "success_by_experiment.png"))
plot_diagnostics_dashboard(history, save_path=joinpath(diagnostics_dir, "diagnostics_dashboard.png"))
n_files, total_size = dir_stats(diagnostics_dir)
println(" ✓ ($n_files files, $(format_bytes(total_size)))")

# Animations
if viz_params.generate_animations
    println("  Animations:")
    local anim_count = 0
    
    # 1. Trajectory animation
    if !isempty(recorder.positions)
        print("    Trajectory...")
        animate_trajectory(recorder.positions, 
            fps=viz_params.animation_fps,
            save_path=joinpath(animations_dir, "trajectory.gif"))
        println(" ✓")
        anim_count += 1
    end
    
    # 2. Belief evolution animation
    if !isempty(recorder.beliefs)
        print("    Belief evolution...")
        animate_belief_evolution(recorder.beliefs, recorder.positions,
            fps=viz_params.animation_fps,
            save_path=joinpath(animations_dir, "belief_evolution.gif"))
        println(" ✓")
        anim_count += 1
    end
    
    # 3. Learning progress animation
    print("    Learning progress...")
    animate_learning_progress(history, 
        fps=5,
        save_path=joinpath(animations_dir, "learning_progress.gif"))
    println(" ✓")
    anim_count += 1
    
    # 4. Matrix evolution animation (A & B)
    if !isempty(last_A_history) && !isempty(last_B_history)
        print("    Matrix evolution...")
        animate_matrix_evolution(last_A_history, last_B_history,
            fps=2,
            save_path=joinpath(animations_dir, "matrix_evolution.gif"))
        println(" ✓")
        anim_count += 1
    end
    
    # 5. Policy evolution animation
    if !isempty(recorder.policies)
        print("    Policy evolution...")
        policy_probs = [params(first(p))[1] for p in recorder.policies]
        animate_policy_evolution(policy_probs,
            fps=viz_params.animation_fps,
            save_path=joinpath(animations_dir, "policy_evolution.gif"))
        println(" ✓")
        anim_count += 1
    end
    
    # 6. Entropy evolution animation
    if !isempty(recorder.beliefs) && !isempty(recorder.policies)
        print("    Entropy evolution...")
        belief_entropies = [distribution_entropy(b) for b in recorder.beliefs]
        policy_entropies = [distribution_entropy(first(p)) for p in recorder.policies]
        animate_entropy_evolution(belief_entropies, policy_entropies,
            fps=viz_params.animation_fps,
            save_path=joinpath(animations_dir, "entropy_evolution.gif"))
        println(" ✓")
        anim_count += 1
    end
    
    # 7. Dashboard animation
    print("    Dashboard animation...")
    animate_dashboard(history, recorder,
        fps=3,
        save_path=joinpath(animations_dir, "dashboard.gif"))
    println(" ✓")
    anim_count += 1
    
    # 8. A matrix grid evolution
    if !isempty(last_A_history)
        print("    A matrix grid evolution...")
        animate_A_grid_evolution(last_A_history,
            fps=2,
            save_path=joinpath(animations_dir, "A_grid_evolution.gif"))
        println(" ✓")
        anim_count += 1
    end
    
    # 9. B matrix action evolution (for each action)
    if !isempty(last_B_history)
        for action_idx in 1:4
            print("    B[$( ACTION_NAMES[action_idx])] evolution...")
            animate_B_action_evolution(last_B_history, action_idx,
                fps=2,
                save_path=joinpath(animations_dir, "B_$(ACTION_NAMES[action_idx])_evolution.gif"))
            println(" ✓")
            anim_count += 1
        end
    end
    
    n_files, total_size = dir_stats(animations_dir)
    println("  Total: $anim_count animations ($(format_bytes(total_size)))")
end

# Data export
print("  Data export...")
export_all_diagnostics(agent, history, recorder, output_dir=output_dir)

# Across-trial summary
summary_data = Dict(
    "timestamp" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
    "config" => Dict(
        "grid_size" => grid_params.grid_size,
        "wind" => collect(grid_params.wind),
        "goal" => collect(grid_params.goal),
        "start" => collect(grid_params.start),
        "planning_horizon" => T,
        "n_iterations" => experiment_params.n_iterations,
        "n_experiments" => n_experiments,
        "n_replicates" => n_replicates,
        "A_diagonal_bias" => prior_params.A_diagonal_bias,
        "B_concentration" => prior_params.B_concentration
    ),
    "results" => Dict(
        "final_success_rates" => all_final_rates,
        "mean_success_rate" => mean_rate / 100,
        "std_success_rate" => std_rate / 100,
        "ci_95_success_rate" => ci_95 / 100,
        "convergence_experiments" => all_convergence_exp,
        "mean_convergence" => mean_conv,
        "mean_steps_to_goal" => all_mean_steps
    )
)

open(joinpath(data_dir, "analysis_summary.json"), "w") do io
    JSON.print(io, summary_data, 2)
end
n_files, total_size = dir_stats(data_dir)
println(" ✓ ($n_files files, $(format_bytes(total_size)))")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
println()
println("═"^70)
println("   ANALYSIS COMPLETE")
println("═"^70)
println()

# Count total output files
function count_output_files(dir)
    files_count = 0
    bytes_count = 0
    if isdir(dir)
        for (root, dirs, files) in walkdir(dir)
            for f in files
                files_count += 1
                bytes_count += filesize(joinpath(root, f))
            end
        end
    end
    return files_count, bytes_count
end

out_files, out_bytes = count_output_files(output_dir)

println("┌" * "─"^68 * "┐")
println("│  SUMMARY" * " "^59 * "│")
println("├" * "─"^68 * "┤")
println("│  Total experiments: $(n_replicates * n_experiments)" * " "^(45 - length(string(n_replicates * n_experiments))) * "│")
println("│  Mean success rate: $(round(mean_rate, digits=1))% ± $(round(ci_95, digits=1))%" * " "^(32 - length(string(round(mean_rate, digits=1))) - length(string(round(ci_95, digits=1)))) * "│")
println("│  Convergence: experiment $(round(Int, mean_conv))" * " "^(42 - length(string(round(Int, mean_conv)))) * "│")
println("├" * "─"^68 * "┤")
println("│  Output: $out_files files ($(format_bytes(out_bytes)))" * " "^(42 - length(string(out_files)) - length(format_bytes(out_bytes))) * "│")
println("│  Location: output/" * " "^48 * "│")
println("└" * "─"^68 * "┘")
println()
println("✓ Full analysis complete!")
