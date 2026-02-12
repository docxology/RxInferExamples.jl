using RxInfer, Random, Plots, Distributions, Dates

# Include Support utilities
const SCRIPT_DIR = @__DIR__
const SUPPORT_PATH = abspath(joinpath(SCRIPT_DIR, "../../../support/src/julia/Support.jl"))

include(SUPPORT_PATH)
using .Support

# Alias logging functions for convenience and to avoid export ambiguities
const log_info = Support.LoggingUtils.log_info
const log_warn = Support.LoggingUtils.log_warn
const log_error = Support.LoggingUtils.log_error
const log_section = Support.LoggingUtils.log_section
const log_config = Support.LoggingUtils.log_config
const log_validation = Support.LoggingUtils.log_validation

# Alias reporting functions
const setup_report = Support.ReportingUtils.setup_report
const append_to_report = Support.ReportingUtils.append_to_report
const generate_markdown_table = Support.ReportingUtils.generate_markdown_table

# Alias validation functions
const validate_threshold = Support.ValidationUtils.validate_threshold
const validate_all = Support.ValidationUtils.validate_all

# Alias animation functions
const create_animation = Support.AnimationUtils.create_animation
const save_animation = Support.AnimationUtils.save_animation

# =========================================================================================
# Model Definition
# =========================================================================================

@model function coin_model(y, a, b)
    θ ~ Beta(a, b)
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

# =========================================================================================
# Core Logic
# =========================================================================================

"""
    generate_data(rng, n, theta)
Generate synthetic data for the coin toss model.
"""
function generate_data(rng, n::Int, theta::Float64)
    return float.(rand(rng, Bernoulli(theta), n))
end

"""
    run_batch_inference(dataset, prior_a, prior_b)
Run standard batch inference with free energy calculation.
"""
function run_batch_inference(dataset, prior_a, prior_b)
    # Enable free_energy calculation
    return infer(
        model = coin_model(a = prior_a, b = prior_b), 
        data  = (y = dataset,),
        free_energy = true
    )
end

"""
    run_online_inference(dataset, prior_a, prior_b)
Run online inference to track belief updates over time.
Returns a vector of posteriors, one for each time step.
"""
function run_online_inference(dataset, prior_a, prior_b)
    # For this simple conjugate model, we can analytically compute the sequence of posteriors
    # to avoid overhead of running `infer` n times, or we can use RxInfer's streaming.
    # For clarity and animation purposes, we will simulate the online update analytically here
    # since Beta-Bernoulli is available in closed form: Beta(a + heads, b + tails)
    
    posteriors = Vector{Beta}(undef, length(dataset))
    current_a = prior_a
    current_b = prior_b
    
    for (i, y) in enumerate(dataset)
        if y == 1.0
            current_a += 1
        else
            current_b += 1
        end
        posteriors[i] = Beta(current_a, current_b)
    end
    
    return posteriors
end

# =========================================================================================
# Visualization & Reporting
# =========================================================================================

"""
    create_results_plot(dataset, posterior, real_theta, prior_a, prior_b)
Create the final static result plot.
"""
function create_results_plot(dataset, posterior, real_theta, prior_a, prior_b)
    rθ = range(0, 1, length = 1000)
    p = create_figure(
        title = "Coin Toss Inference (n=$(length(dataset)))",
        xlabel = "θ",
        ylabel = "Probability Density"
    )
    
    plot!(p, rθ, (x) -> pdf(Beta(prior_a, prior_b), x), 
          fillalpha=0.3, fillrange=0, label="Prior P(θ)", c=:blue)
          
    plot!(p, rθ, (x) -> pdf(posterior, x), 
          fillalpha=0.3, fillrange=0, label="Posterior P(θ|y)", c=:green)
          
    vline!(p, [real_theta], label="Real θ = $real_theta", c=:red, linestyle=:dash, linewidth=2)
    return p
end

"""
    create_posterior_animation(dataset, posteriors, real_theta, fps)
Create an animation of the posterior distribution evolving over time.
"""
function create_posterior_animation(dataset, posteriors, real_theta; fps=10)
    rθ = range(0, 1, length = 1000)
    
    # We will sample frames to keep animation size reasonable if n is large
    # Target around 10 seconds of animation max
    n_frames = length(posteriors)
    step_size = max(1, floor(Int, n_frames / (10 * fps)))
    indices = 1:step_size:n_frames
    
    frames = []
    
    for i in indices
        p = create_figure(
            title = "Belief Update (Observation $i)",
            xlabel = "θ", 
            ylabel = "Probability Density"
        )
        ylims!(p, 0, 25) # Fix y-axis to make animation stable
        
        # Plot current posterior
        post = posteriors[i]
        plot!(p, rθ, (x) -> pdf(post, x), 
              fillalpha=0.3, fillrange=0, label="P(θ|y_{1:$i})", c=:green, linewidth=2)
        
        # Plot Real Theta
        vline!(p, [real_theta], label="Real θ", c=:red, linestyle=:dash)
        
        push!(frames, p)
    end
    
    return create_animation(frames, fps=fps)
end
function generate_analysis_report(output_dir, config, results_dict)
    report_path = setup_report(output_dir, "Coin Toss Model Analysis Report")
    
    # 1. Configuration
    append_to_report(report_path, "## Configuration")
    
    seed = get_config_value(config, "inference", "seed", default="N/A")
    n_samples = get_config_value(config, "model", "n_samples", default="N/A")
    theta_real = get_config_value(config, "model", "theta_real", default="N/A")
    prior_a = get_config_value(config, "model", "prior_a", default="N/A")
    prior_b = get_config_value(config, "model", "prior_b", default="N/A")

    config_rows = [
        ["Seed", string(seed)],
        ["Samples", string(n_samples)],
        ["Real θ", string(theta_real)],
        ["Prior a", string(prior_a)],
        ["Prior b", string(prior_b)]
    ]
    append_to_report(report_path, generate_markdown_table(["Parameter", "Value"], config_rows))
    
    # 2. Results
    append_to_report(report_path, "\n## Results")
    
    posterior = results_dict[:posterior]
    mean_est = mean(posterior)
    std_est = std(posterior)
    real_theta = config["model"]["theta_real"]
    error = abs(mean_est - real_theta)
    free_energy = results_dict[:free_energy]
    
    stats_rows = [
        ["Posterior Mean", string(round(mean_est, digits=4))],
        ["Posterior Std", string(round(std_est, digits=4))],
        ["Absolute Error", string(round(error, digits=4))],
        ["Bethe Free Energy", string(round(free_energy[end], digits=4))]
    ]
    append_to_report(report_path, generate_markdown_table(["Metric", "Value"], stats_rows))
    
    # 3. Validation
    append_to_report(report_path, "\n## Validation")
    val_result = results_dict[:validation]
    status = val_result.passed ? "✅ PASS" : "❌ FAIL"
    append_to_report(report_path, "**Overall Status**: $status")
    append_to_report(report_path, "\n### Messages")
    for msg in val_result.messages
        append_to_report(report_path, "- $msg")
    end
    
    return report_path
end

"""
    validate_results(posterior, real_theta, threshold)
Validate the inference results against the ground truth.
"""
function validate_results(posterior, real_theta, threshold)
    mean_est = mean(posterior)
    error = abs(mean_est - real_theta)
    
    v1 = validate_threshold(error, threshold, comparison=:<=, name="Estimation Error")
    v2 = validate_threshold(std(posterior), 0.1, comparison=:<=, name="Posterior Uncertainty")
    
    return validate_all([v1, v2])
end

# =========================================================================================
# Main Execution
# =========================================================================================

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--config"
            help = "Path to the configuration file"
            default = "config.toml"
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    config_arg = parsed_args["config"]
    
    # 1. Configuration
    # Handle relative paths for config file (relative to script or CWD)
    if isabspath(config_arg)
        config_path = config_arg
    else
        # Try finding it relative to CWD first, then relative to script
        if isfile(joinpath(pwd(), config_arg))
            config_path = joinpath(pwd(), config_arg)
        else
            config_path = joinpath(SCRIPT_DIR, config_arg)
        end
    end
    
    if !isfile(config_path)
        error("Configuration file not found at: $config_path")
    end
    
    config = load_toml_config(config_path)
    
    # 2. Setup Environment & Logging
    output_config = get_config_value(config, "output", default=Dict())
    output_dir = get_config_value(output_config, "directory", default="output")
    if !isabspath(output_dir); output_dir = joinpath(SCRIPT_DIR, output_dir); end
    ensure_directory(output_dir)
    
    seed = get_config_value(config, "inference", "seed", default=42)
    logger = setup_logger(output_dir, config_path, seed)
    
    log_section(logger, "Coin Toss Model Analysis")
    log_config(logger, config)
    
    try
        # 3. Initialization
        log_info(logger, "Initializing...")
        rng = MersenneTwister(seed)
        
        n_samples = get_config_value(config, "model", "n_samples", default=100)
        theta_real = get_config_value(config, "model", "theta_real", default=0.75)
        prior_a = get_config_value(config, "model", "prior_a", default=1.0)
        prior_b = get_config_value(config, "model", "prior_b", default=1.0)
        
        # 4. Data Generation
        log_section(logger, "Data Generation")
        dataset = generate_data(rng, n_samples, theta_real)
        log_info(logger, "Generated $n_samples samples.")
        
        # 5. Batch Inference
        log_section(logger, "Batch Inference")
        batch_result = run_batch_inference(dataset, prior_a, prior_b)
        posterior = batch_result.posteriors[:θ]
        free_energy = batch_result.free_energy
        log_info(logger, "Batch inference complete.")
        log_info(logger, "Posterior: Mean=$(round(mean(posterior),digits=4))")
        log_info(logger, "Final Free Energy: $(round(free_energy[end], digits=4)) nats")
        
        # 6. Validation
        log_section(logger, "Validation")
        threshold = get_config_value(config, "validation", "error_threshold", default=0.05)
        val_result = validate_results(posterior, theta_real, threshold)
        log_validation(logger, val_result)
        
        # 7. Visualization
        if get_config_value(output_config, "save_plots", default=true)
            log_section(logger, "Visualization")
            p = create_results_plot(dataset, posterior, theta_real, prior_a, prior_b)
            save_plot(p, "inference_results.png", output_dir)
            log_info(logger, "Static plot saved.")
        end
        
        # 8. Animation (Online Inference)
        if get_config_value(output_config, "save_animation", default=false)
            log_section(logger, "Animation Generation")
            log_info(logger, "Running online inference for animation...")
            # Note: run_online_inference returns a vector of posteriors
            online_posteriors = run_online_inference(dataset, prior_a, prior_b)
            
            fps = get_config_value(output_config, "animation_fps", default=10)
            anim = create_posterior_animation(dataset, online_posteriors, theta_real, fps=fps)
            anim_path = save_animation(anim, "posterior_animation.gif", output_dir, fps=fps)
            log_info(logger, "Animation saved to: $anim_path")
        end
        
        # 9. Reporting
        if get_config_value(output_config, "save_report", default=false)
            log_section(logger, "Reporting")
            results_dict = Dict(
                :posterior => posterior,
                :validation => val_result,
                :free_energy => free_energy
            )
            report_path = generate_analysis_report(output_dir, config, results_dict)
            log_info(logger, "Report saved to: $report_path")
        end
        
        log_section(logger, "Execution Finished Successfully")
        
    catch e
        log_error(logger, "Execution failed: $e")
        rethrow(e)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end