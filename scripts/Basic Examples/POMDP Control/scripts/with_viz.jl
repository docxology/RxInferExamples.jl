# POMDP Control with Comprehensive Visualization
# Called from run.jl: julia --project=. run.jl viz

const SCRIPT_DIR = @__DIR__
const PROJECT_DIR = dirname(SCRIPT_DIR)

using Pkg
Pkg.activate(PROJECT_DIR)
Pkg.instantiate()

include(joinpath(PROJECT_DIR, "src/PODMPControlApp.jl"))
using .PODMPControlApp
using Distributions

println("="^60)
println("POMDP Control with Comprehensive Visualization")
println("="^60)
println()

# Ensure output directories exist
output_dir = ensure_output_dirs()
println("✓ Output directory: $output_dir")

# Create environment and agent
env, rx_agent = create_environment()
agent = PODMPAgent(env, rx_agent)

# Initialize trackers
history = LearningHistory()
recorder = TrajectoryRecorder()
A_history = []
B_history = []

# Run experiments with tracking
n_experiments = 50
T = experiment_params.planning_horizon

println("\nRunning $n_experiments experiments with visualization tracking...")
println()

for exp in 1:n_experiments
    # Reset agent for new experiment
    reset!(agent)
    clear!(recorder)
    
    # Track matrices at start
    push!(A_history, deepcopy(agent.p_A))
    push!(B_history, deepcopy(agent.p_B))
    
    # Run experiment with step timing
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
    
    # Record experiment results
    record_experiment!(history, exp, success, steps_taken, agent)
    
    if exp % 10 == 0
        rate = history.cumulative_success_rate[end] * 100
        println("  Experiment $exp: $(round(rate, digits=1))% cumulative success")
    end
end

println()
println("="^60)
println("Generating Visualizations")
println("="^60)
println()

# Generate all visualizations
figures_dir = joinpath(output_dir, "figures")
matrices_dir = joinpath(figures_dir, "matrices")
beliefs_dir = joinpath(figures_dir, "beliefs")
trajectories_dir = joinpath(figures_dir, "trajectories")
diagnostics_dir = joinpath(figures_dir, "diagnostics")
animations_dir = joinpath(output_dir, "animations")
data_dir = joinpath(output_dir, "data")

# Matrix visualizations
println("  Generating matrix visualizations...")
plot_A_matrix(agent, save_path=joinpath(matrices_dir, "A_matrix.png"))
plot_all_B_matrices(agent, save_dir=matrices_dir)
plot_A_matrix_as_grid(agent, save_path=joinpath(matrices_dir, "A_grid.png"))
for action_idx in 1:4
    plot_B_matrix_as_arrows(agent, action_idx, 
        save_path=joinpath(matrices_dir, "B_arrows_$(ACTION_NAMES[action_idx]).png"))
end
println("  ✓ Matrix plots saved")

# Belief visualizations
println("  Generating belief visualizations...")
plot_state_belief(agent, save_path=joinpath(beliefs_dir, "state_belief_final.png"))
plot_policy(agent, save_path=joinpath(beliefs_dir, "policy_final.png"))
plot_goal_prior(agent, save_path=joinpath(beliefs_dir, "goal_prior.png"))
println("  ✓ Belief plots saved")

# Trajectory visualizations
println("  Generating trajectory visualizations...")
plot_trajectory(recorder.positions, save_path=joinpath(trajectories_dir, "trajectory_final.png"))
plot_action_sequence(recorder.actions, save_path=joinpath(trajectories_dir, "actions_final.png"))
plot_experiment_summary(recorder, save_path=joinpath(trajectories_dir, "experiment_summary.png"))
println("  ✓ Trajectory plots saved")

# Diagnostics visualizations
println("  Generating diagnostics dashboard...")
plot_learning_curve(history, save_path=joinpath(diagnostics_dir, "learning_curve.png"))
plot_matrix_entropy_evolution(history, save_path=joinpath(diagnostics_dir, "entropy_evolution.png"))
plot_diagnostics_dashboard(history, save_path=joinpath(diagnostics_dir, "diagnostics_dashboard.png"))
println("  ✓ Diagnostics plots saved")

# Animations
println("  Generating animations...")
animate_trajectory(recorder.positions, save_path=joinpath(animations_dir, "trajectory.gif"))
animate_belief_evolution(recorder.beliefs, recorder.positions, 
    save_path=joinpath(animations_dir, "belief_evolution.gif"))
animate_learning_progress(history, save_path=joinpath(animations_dir, "learning_progress.gif"))
println("  ✓ Animations saved")

# Data export
println("  Exporting data...")
export_all_diagnostics(agent, history, recorder, output_dir=output_dir)
println("  ✓ Data exported")

# Summary
println()
println("="^60)
println("RESULTS SUMMARY")
println("="^60)
println()
println("  Total experiments: $n_experiments")
println("  Final success rate: $(round(history.cumulative_success_rate[end] * 100, digits=1))%")
println()
println("  Output saved to: $output_dir")
println()
println("  Figures:")
println("    - matrices/     : A and B matrix heatmaps and arrows")
println("    - beliefs/      : State beliefs and policies")
println("    - trajectories/ : Agent paths and action sequences")  
println("    - diagnostics/  : Learning curves and dashboards")
println()
println("  Animations:")
println("    - trajectory.gif")
println("    - belief_evolution.gif")
println("    - learning_progress.gif")
println()
println("  Data:")
println("    - agent_state_*.json")
println("    - learning_history_*.json")
println("    - trajectory_*.json")
println("    - model_spec_*.json")
println()
println("✓ Visualization complete!")
