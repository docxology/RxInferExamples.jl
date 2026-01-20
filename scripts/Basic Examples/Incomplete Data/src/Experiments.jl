module Experiments

using RxInfer
using LinearAlgebra
using Random
using Statistics
using Dates

using ..DataGeneration
using ..Models
using ..Analysis
using ..LoggingUtils
using ..AnalysisUtils
using ..ReportingUtils

export run_incomplete_data

"""
    run_incomplete_data(config, output_dir, seed)

Main entry point for the Incomplete Data experiment.
"""
function run_incomplete_data(config, output_dir, seed)
    # Setup infrastructure
    logger = setup_logger(output_dir, "config.toml", seed)
    report_path = setup_report(output_dir, "Incomplete Data Analysis Report")
    
    log_section(logger, "Initialization")
    logger("Starting Incomplete Data Experiment...")
    
    # Parse config
    n_samples = get(config, "n_samples", 100)
    dimension = get(config, "dimension", 6)
    inference_conf = get(config, "inference", Dict())
    iterations = get(inference_conf, "iterations", 100)
    
    # Data Generation
    log_section(logger, "Data Generation")
    rng = MersenneTwister(seed)
    data = generate_incomplete_data(rng, n_samples, dimension)
    
    logger("Generated $n_samples samples with dimension $dimension")
    logger("Real Mean (first 3): $(data.real_mean[1:min(3, dimension)])...")
    
    # Model Setup
    log_section(logger, "Inference")
    
    constraints = Models.create_constraints()
    init = Models.create_initialization(dimension)
    
    logger("Running inference...")
    result = infer(
        model = incomplete_data(dim=dimension, huge=1e6),
        data = (y=data.observed_data,),
        constraints = constraints,
        initialization = init,
        showprogress = get(inference_conf, "show_progress", true),
        iterations = iterations,
        free_energy = true
    )
    
    logger("Inference completed.")
    logevidence = -result.free_energy[end]
    logger("Final Free Energy: $(result.free_energy[end])")
    
    # Analysis
    log_section(logger, "Analysis")
    
    est_mean = mean(result.posteriors[:m][end])
    est_prec = mean(result.posteriors[:Λ][end])
    est_cov = inv(est_prec)
    
    stats_mean = get_statistics(data.real_mean, est_mean)
    # For precision/covariance, we can just log errors or verify mean primarily
    
    log_info(logger, "Mean Estimation Stats: RMSE=$(stats_mean.rmse), VAF=$(stats_mean.vaf)%")
    
    # Validation
    val_conf = get(config, "validation", Dict())
    val_result = validate_metrics(stats_mean, val_conf)
    log_validation(logger, val_result)
    
    # Reporting
    headers = ["Metric", "Value"]
    rows = [
        ["RMSE (Mean)", string(round(stats_mean.rmse, digits=4))],
        ["MAE (Mean)", string(round(stats_mean.mae, digits=4))],
        ["VAF (Mean)", string(round(stats_mean.vaf, digits=2))],
        ["Max Error", string(round(stats_mean.max_err, digits=4))],
        ["Validation", val_result.passed ? "PASS" : "FAIL"]
    ]
    
    append_to_report(report_path, "### Incomplete Data Inference Results\n")
    append_to_report(report_path, generate_markdown_table(headers, rows))
    append_to_report(report_path, "\nRun completed at $(Dates.now())\n")
    
    # Plotting
    log_section(logger, "Visualization")
    p = plot_posterior_distributions(result, data.real_mean, data.real_precision)
    plot_path = joinpath(output_dir, "posterior_distributions.png")
    savefig(p, plot_path)
    logger("Saved posterior plot to $plot_path")
    
    logger("Experiment finished successfully.")
    
    return result
end

end # module
