module KalmanFilterApp

using TOML

# Include submodules
include("DataGeneration.jl")
include("Models.jl")
include("Visualization.jl")
include("Analysis.jl")
include("LoggingUtils.jl")
include("Reporting.jl")
include("Experiments.jl")

# Re-export necessary components for external use if needed
using .DataGeneration
using .Models
using .Visualization
using .Analysis
using .LoggingUtils
using .Reporting
using .Experiments

export run_all

"""
    load_config(path="config.toml")

Load configuration from TOML file.
"""
function load_config(path="config.toml")
    if !isfile(path)
        println("Warning: Config file $path not found. Using defaults.")
        return Dict()
    end
    return TOML.parsefile(path)
end

"""
    run_all()

Run all examples using configuration.
"""
function run_all(; config_path="config.toml")
    config = load_config(config_path)
    output_dir = get(get(config, "general", Dict()), "output_dir", "output")
    seed = get(get(config, "general", Dict()), "seed", 1234)
    
    mkpath(output_dir)
    println("Using configuration from $config_path")
    println("Output directory: $output_dir")
    
    # Initialize Logger and Report
    logger = setup_logger(output_dir, config_path, seed)
    report_path = setup_report(output_dir)
    
    try
        run_rotating_ssm(config, output_dir, seed, report_path, logger)
        run_identification_problem(config, output_dir, seed, report_path, logger)
        run_rx_identification(config, output_dir, seed, report_path, logger)
        run_smoothing_example(config, output_dir, seed, report_path, logger)
        logger("All examples executed successfully.")
    catch e
        logger("Error during execution: $e")
        rethrow(e)
    end
    
    println("\nAnalysis report saved to $report_path")
    println("Execution log saved to $(joinpath(output_dir, "execution.log"))")
end

end # module
