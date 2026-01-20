# POMDP Control - Combinatorial Parameter Sweep
# Test all combinations of config parameters across replicates
# Called from run.jl: julia --project=. run.jl sweep
#
# Generates comparison plots and CSV/JSON exports for all combinations

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
using Plots
using DelimitedFiles

"""Format bytes into human-readable string."""
function format_bytes(bytes::Integer)
    bytes < 1024 && return "$(bytes) B"
    bytes < 1024^2 && return "$(round(bytes/1024, digits=1)) KB"
    return "$(round(bytes/1024^2, digits=2)) MB"
end

println("═"^70)
println("   POMDP CONTROL - COMBINATORIAL PARAMETER SWEEP")
println("   $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
println("═"^70)
println()

# ============================================================================
# SWEEP CONFIGURATION - Edit these to define parameter combinations
# ============================================================================

SWEEP_PARAMS = Dict(
    # Planning horizon - longer horizons help with stochastic exploration
    :planning_horizon => [4, 6, 8, 10],
    
    # Exploration temperature - lower = more greedy, higher = more random
    # Key: find balance between exploration and exploitation
    :exploration_temperature => [0.1, 0.2, 0.3, 0.5],
    
    # Number of VMP iterations - more = better inference, slower
    # :n_iterations => [10, 20, 30],
    
    # Prior parameters (uncomment to sweep):
    # :A_diagonal_bias => [0.05, 0.1, 0.2],
    # :B_concentration => [0.5, 1.0, 2.0],
)

# Fixed settings for sweep - increased for better statistics
EXPERIMENTS_PER_CONFIG = 100   # Experiments per parameter combination
REPLICATES_PER_CONFIG = 3     # Replicates per parameter combination

# ============================================================================
# GENERATE ALL COMBINATIONS
# ============================================================================

"""Generate all combinations of parameter values."""
function generate_combinations(params::Dict)
    keys_list = collect(keys(params))
    values_list = [params[k] for k in keys_list]
    
    # Cartesian product of all values
    combinations = []
    
    function recurse(idx, current)
        if idx > length(keys_list)
            push!(combinations, Dict(zip(keys_list, current)))
        else
            for val in values_list[idx]
                recurse(idx + 1, [current..., val])
            end
        end
    end
    
    recurse(1, [])
    return combinations
end

combinations = generate_combinations(SWEEP_PARAMS)
n_configs = length(combinations)
total_experiments = n_configs * EXPERIMENTS_PER_CONFIG * REPLICATES_PER_CONFIG

println("┌" * "─"^68 * "┐")
println("│  SWEEP CONFIGURATION" * " "^47 * "│")
println("├" * "─"^68 * "┤")
for (k, v) in SWEEP_PARAMS
    println("│  $(k): $(v)" * " "^(66 - length(string(k)) - length(string(v))) * "│")
end
println("├" * "─"^68 * "┤")
println("│  Parameter combinations: $n_configs" * " "^(41 - length(string(n_configs))) * "│")
println("│  Experiments per config: $EXPERIMENTS_PER_CONFIG" * " "^(42 - length(string(EXPERIMENTS_PER_CONFIG))) * "│")
println("│  Replicates per config: $REPLICATES_PER_CONFIG" * " "^(43 - length(string(REPLICATES_PER_CONFIG))) * "│")
println("│  Total experiments: $total_experiments" * " "^(47 - length(string(total_experiments))) * "│")
println("└" * "─"^68 * "┘")
println()

# ============================================================================
# OUTPUT SETUP
# ============================================================================

output_dir = abspath(joinpath(PROJECT_DIR, "output", "sweeps"))
mkpath(output_dir)
timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
sweep_log = joinpath(output_dir, "sweep_log_$timestamp.txt")

# Initialize log
open(sweep_log, "w") do io
    println(io, "POMDP Control - Combinatorial Sweep Log")
    println(io, "Timestamp: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "="^70)
    println(io, "Parameters swept: $(keys(SWEEP_PARAMS))")
    println(io, "Combinations: $n_configs")
    println(io, "Experiments per config: $EXPERIMENTS_PER_CONFIG")
    println(io, "Replicates per config: $REPLICATES_PER_CONFIG")
    println(io, "="^70)
    println(io)
end

println("Output: $output_dir")
println("Log: $sweep_log")
println()

# ============================================================================
# RUN ALL COMBINATIONS
# ============================================================================

println("═"^70)
println("   RUNNING $n_configs PARAMETER COMBINATIONS")
println("═"^70)
println()

# Store all results
results = Dict{Int, Dict{Symbol, Any}}()

for (config_idx, config) in enumerate(combinations)
    config_start = time()
    
    # Display config
    config_str = join(["$k=$v" for (k,v) in config], ", ")
    println("┌" * "─"^68 * "┐")
    println("│  Config $config_idx/$n_configs: $config_str" * " "^max(0, 51 - length(config_str) - length(string(config_idx)) - length(string(n_configs))) * "│")
    println("└" * "─"^68 * "┘")
    
    # Get config values with defaults
    T = Int(get(config, :planning_horizon, experiment_params.planning_horizon))
    τ = get(config, :exploration_temperature, experiment_params.exploration_temperature)
    
    # Storage for this config
    config_rates = Float64[]
    config_conv = Float64[]
    config_steps = Float64[]
    
    for rep in 1:REPLICATES_PER_CONFIG
        # Create fresh environment and agent
        env, rx_agent = create_environment()
        agent = PODMPAgent(env, rx_agent)
        
        successes = Bool[]
        steps_list = Int[]
        
        for exp in 1:EXPERIMENTS_PER_CONFIG
            reset!(agent)
            
            success = false
            steps = 0
            
            for t in 1:T
                goal_reached = step!(agent, T - t)
                steps = t
                if goal_reached
                    success = true
                    break
                end
            end
            
            push!(successes, success)
            push!(steps_list, steps)
        end
        
        # Calculate metrics for this replicate
        success_rate = mean(successes)
        cum_rates = cumsum(successes) ./ (1:EXPERIMENTS_PER_CONFIG)
        conv_exp = findfirst(r -> r > 0.5, cum_rates)
        mean_steps = mean(steps_list)
        
        push!(config_rates, success_rate)
        push!(config_conv, conv_exp !== nothing ? conv_exp : EXPERIMENTS_PER_CONFIG)
        push!(config_steps, mean_steps)
    end
    
    # Aggregate across replicates
    mean_rate = mean(config_rates)
    std_rate = REPLICATES_PER_CONFIG > 1 ? std(config_rates) : 0.0
    mean_conv_exp = mean(config_conv)
    mean_steps_agg = mean(config_steps)
    
    config_time = time() - config_start
    
    # Store results
    results[config_idx] = Dict(
        :config => config,
        :mean_success_rate => mean_rate,
        :std_success_rate => std_rate,
        :replicate_rates => config_rates,
        :mean_convergence => mean_conv_exp,
        :mean_steps => mean_steps_agg,
        :runtime => config_time
    )
    
    # Log
    open(sweep_log, "a") do io
        println(io, "Config $config_idx: $config_str")
        println(io, "  Success rate: $(round(mean_rate*100, digits=1))% ± $(round(std_rate*100, digits=1))%")
        println(io, "  Replicates: $config_rates")
        println(io, "  Convergence: $(round(mean_conv_exp, digits=1))")
        println(io, "  Runtime: $(round(config_time, digits=1))s")
        println(io)
    end
    
    println("  Success: $(round(mean_rate*100, digits=1))% ± $(round(std_rate*100, digits=1))%")
    println("  Time: $(round(config_time, digits=1))s")
    println()
end

# ============================================================================
# EXPORT RESULTS
# ============================================================================

println("═"^70)
println("   EXPORTING RESULTS")
println("═"^70)
println()

# JSON export
json_path = joinpath(output_dir, "sweep_results_$timestamp.json")
json_data = Dict(
    "timestamp" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
    "parameters" => Dict(string(k) => v for (k,v) in SWEEP_PARAMS),
    "n_combinations" => n_configs,
    "experiments_per_config" => EXPERIMENTS_PER_CONFIG,
    "replicates_per_config" => REPLICATES_PER_CONFIG,
    "results" => [
        Dict(
            "config_idx" => idx,
            "config" => Dict(string(k) => v for (k,v) in r[:config]),
            "mean_success_rate" => r[:mean_success_rate],
            "std_success_rate" => r[:std_success_rate],
            "replicate_rates" => r[:replicate_rates],
            "mean_convergence" => r[:mean_convergence],
            "mean_steps" => r[:mean_steps],
            "runtime" => r[:runtime]
        )
        for (idx, r) in sort(collect(results), by=x->x[1])
    ]
)
open(json_path, "w") do io
    JSON.print(io, json_data, 2)
end
println("  JSON: $json_path")

# CSV export
csv_path = joinpath(output_dir, "sweep_results_$timestamp.csv")
open(csv_path, "w") do io
    # Header
    param_names = collect(keys(SWEEP_PARAMS))
    println(io, join(["config_idx", string.(param_names)..., "mean_success_rate", "std_success_rate", "mean_convergence", "mean_steps", "runtime"], ","))
    
    # Data rows
    for (idx, r) in sort(collect(results), by=x->x[1])
        row = [
            idx,
            [get(r[:config], p, "") for p in param_names]...,
            round(r[:mean_success_rate], digits=4),
            round(r[:std_success_rate], digits=4),
            round(r[:mean_convergence], digits=1),
            round(r[:mean_steps], digits=2),
            round(r[:runtime], digits=1)
        ]
        println(io, join(row, ","))
    end
end
println("  CSV: $csv_path")

# ============================================================================
# GENERATE COMPARISON PLOTS
# ============================================================================

println()
println("═"^70)
println("   GENERATING COMPARISON PLOTS")
println("═"^70)
println()

# Get parameter names for plotting
param_names = collect(keys(SWEEP_PARAMS))

# If we have 2 parameters, create heatmap
if length(param_names) == 2
    p1, p2 = param_names
    v1, v2 = SWEEP_PARAMS[p1], SWEEP_PARAMS[p2]
    
    # Build success rate matrix
    rate_matrix = zeros(length(v1), length(v2))
    for (idx, r) in results
        i = findfirst(==(r[:config][p1]), v1)
        j = findfirst(==(r[:config][p2]), v2)
        rate_matrix[i, j] = r[:mean_success_rate] * 100
    end
    
    p = heatmap(string.(v2), string.(v1), rate_matrix,
        xlabel=string(p2),
        ylabel=string(p1),
        title="Success Rate (%) by Parameters",
        color=:viridis,
        clims=(0, max(maximum(rate_matrix), 10))
    )
    
    # Add value annotations
    for i in 1:length(v1), j in 1:length(v2)
        annotate!(p, j, i, text("$(round(rate_matrix[i,j], digits=1))", 8, :white))
    end
    
    heatmap_path = joinpath(output_dir, "sweep_heatmap_$timestamp.png")
    savefig(p, heatmap_path)
    println("  Heatmap: $heatmap_path")
end

# Bar chart of all configs
rates = [results[i][:mean_success_rate] * 100 for i in 1:n_configs]
stds = [results[i][:std_success_rate] * 100 for i in 1:n_configs]
labels = [join(["$(k)=$(results[i][:config][k])" for k in param_names], "\n") for i in 1:n_configs]

p_bar = bar(1:n_configs, rates,
    yerror=stds,
    xlabel="Configuration",
    ylabel="Success Rate (%)",
    title="Sweep Results Comparison",
    legend=false,
    xticks=(1:n_configs, 1:n_configs),
    color=:steelblue
)

bar_path = joinpath(output_dir, "sweep_comparison_$timestamp.png")
savefig(p_bar, bar_path)
println("  Comparison: $bar_path")

# Best config highlight
best_idx = argmax(rates)
best_config = results[best_idx][:config]
best_rate = rates[best_idx]

println()
println("┌" * "─"^68 * "┐")
println("│  BEST CONFIGURATION" * " "^48 * "│")
println("├" * "─"^68 * "┤")
for (k, v) in best_config
    println("│  $(k) = $(v)" * " "^(64 - length(string(k)) - length(string(v))) * "│")
end
println("│  Success rate: $(round(best_rate, digits=1))%" * " "^(50 - length(string(round(best_rate, digits=1)))) * "│")
println("└" * "─"^68 * "┘")

# ============================================================================
# SUMMARY
# ============================================================================

println()
println("═"^70)
println("   SWEEP COMPLETE")
println("═"^70)
println()

total_time = sum(r[:runtime] for (_, r) in results)
output_files = filter(f -> startswith(f, "sweep_"), readdir(output_dir))

println("  Configurations tested: $n_configs")
println("  Total experiments: $total_experiments")
println("  Total runtime: $(round(total_time, digits=1))s")
println()
println("  Output files:")
for f in output_files
    if occursin(timestamp, f)
        size = filesize(joinpath(output_dir, f))
        println("    - $f ($(format_bytes(size)))")
    end
end
println()
println("  Log: $sweep_log")
println()
println("✓ Combinatorial sweep complete!")
