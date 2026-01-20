"""
    PODMPControlApp

Main module for POMDP Control in WindyGridWorld using RxInfer.jl.

Demonstrates:
- POMDP inference-as-planning with discrete state spaces
- Online learning of A and B matrices
- Integration with RxEnvironments for reactive control
- Comprehensive visualization and diagnostics
- Configurable via config.toml

See: docs/README.md for documentation
"""
module PODMPControlApp

using RxInfer
using Distributions
using LinearAlgebra
using Random
using Dates
using RxEnvironments
using Plots
using ProgressMeter
using JSON
using TOML

# Include submodules in dependency order
# Config must come first (loads parameters from config.toml)
include("Config.jl")
include("Constants.jl")
include("Logging.jl")    # Structured logging (replaces Utils.jl logging)
include("Utils.jl")      # Kept for utility functions
include("Diagnostics.jl") # Real-time statistical tracking
include("World.jl")
include("Model.jl")
include("Agent.jl")

# Include visualization submodule
include("visualization/Visualization.jl")

"""
    run_experiments(n_experiments, T; show_progress=true) -> Float64

Run n_experiments of T steps each. Returns success rate.
"""
function run_experiments(
    n_experiments::Int=experiment_params.n_experiments,
    T::Int=experiment_params.planning_horizon;
    show_progress::Bool=true
)
    rxlog("info", "Starting $n_experiments experiments with horizon $T")
    
    # Create environment and agent
    env, rx_agent = create_environment()
    agent = PODMPAgent(env, rx_agent)
    
    successes = Bool[]
    
    if show_progress
        @showprogress for i in 1:n_experiments
            success = run_experiment!(agent, T)
            push!(successes, success)
        end
    else
        for i in 1:n_experiments
            success = run_experiment!(agent, T)
            push!(successes, success)
        end
    end
    
    success_rate = mean(successes)
    rxlog("info", "Completed $n_experiments experiments: success rate = $(round(success_rate * 100, digits=1))%")
    
    return success_rate
end

"""
    launch(; n_experiments=100, T=4) -> (Float64, Plot)

Run experiments and display final environment state.
Returns (success_rate, plot).
"""
function launch(;
    n_experiments::Int=experiment_params.n_experiments,
    T::Int=experiment_params.planning_horizon
)
    rxlog("info", "Launching PODMPControlApp...")
    
    # Run experiments
    success_rate = run_experiments(n_experiments, T)
    
    # Create final visualization
    env, rx_agent = create_environment()
    agent = PODMPAgent(env, rx_agent)
    run_experiment!(agent, T)  # Run one more for visualization
    
    p = plot_environment(agent.environment)
    display(p)
    
    println("\n✓ Completed $n_experiments experiments")
    println("✓ Success rate: $(round(success_rate * 100, digits=1))%")
    
    return (success_rate, p)
end

export run_experiments, launch

end # module
