# Bayesian Networks: The Sprinkler Model
# 
# A comprehensive, modular implementation demonstrating inference and CPT learning
# with RxInfer.jl featuring enhanced configuration, validation, and visualization.

using Pkg
Pkg.activate(@__DIR__)

using TOML
using RxInfer
using Plots
using Random
using Statistics
using JSON

# Load configuration
const CONFIG_PATH = joinpath(@__DIR__, "config.toml")
const CONFIG = TOML.parsefile(CONFIG_PATH)
const SUPPORT_ROOT = joinpath(@__DIR__, "..", "..", "..", "support", "src", "julia")

# ============================================================
# RxInfer Models (must be defined before includes)
# ============================================================

@model function sprinkler_model_basic(wet_grass)
    clouded ~ Categorical([0.5, 0.5])
    rain ~ DiscreteTransition(clouded, [0.8 0.2; 0.2 0.8])
    sprinkler ~ DiscreteTransition(clouded, [0.5 0.9; 0.5 0.1])
    wet_grass ~ DiscreteTransition(sprinkler, [1.0 0.1; 0.0 0.9;;; 0.1 0.01; 0.9 0.99], rain)
end

@model function sprinkler_model_extended(wet_grass_data, sprinkler_data, rain_data, clouded_data)
    clouded ~ Categorical([0.5, 0.5])
    clouded_data ~ DiscreteTransition(clouded, diageye(2))
    rain ~ DiscreteTransition(clouded, [0.8 0.2; 0.2 0.8])
    rain_data ~ DiscreteTransition(rain, diageye(2))
    sprinkler ~ DiscreteTransition(clouded, [0.5 0.9; 0.5 0.1])
    sprinkler_data ~ DiscreteTransition(sprinkler, diageye(2))
    wet_grass ~ DiscreteTransition(sprinkler, [1.0 0.1; 0.0 0.9;;; 0.1 0.01; 0.9 0.99], rain)
    wet_grass_data ~ DiscreteTransition(wet_grass, diageye(2))
end

@model function sprinkler_model_cpt(clouded_data, rain_data, sprinkler_data, wet_grass_data, 
                                    cpt_cloud_rain, cpt_cloud_sprinkler, cpt_sprinkler_rain_wet_grass)
    clouded ~ Categorical([0.5, 0.5])
    clouded_data ~ DiscreteTransition(clouded, diageye(2))
    rain ~ DiscreteTransition(clouded, cpt_cloud_rain)
    rain_data ~ DiscreteTransition(rain, diageye(2))
    sprinkler ~ DiscreteTransition(clouded, cpt_cloud_sprinkler)
    sprinkler_data ~ DiscreteTransition(sprinkler, diageye(2))
    wet_grass ~ DiscreteTransition(sprinkler, cpt_sprinkler_rain_wet_grass, rain)
    wet_grass_data ~ DiscreteTransition(wet_grass, diageye(2))
end

@model function learn_sprinkler_model(clouded_data, rain_data, sprinkler_data, wet_grass_data)
    cpt_cloud_rain ~ DirichletCollection(ones(2, 2))
    cpt_cloud_sprinkler ~ DirichletCollection(ones(2, 2))
    cpt_sprinkler_rain_wet_grass ~ DirichletCollection(ones(2, 2, 2))
    for i in 1:length(clouded_data)
        wet_grass_data[i] ~ sprinkler_model_cpt(
            clouded_data = clouded_data[i], rain_data = rain_data[i], sprinkler_data = sprinkler_data[i],
            cpt_cloud_rain = cpt_cloud_rain, cpt_cloud_sprinkler = cpt_cloud_sprinkler,
            cpt_sprinkler_rain_wet_grass = cpt_sprinkler_rain_wet_grass)
    end
end

# ============================================================
# Include Modules
# ============================================================

include(joinpath(SUPPORT_ROOT, "logging", "LoggingUtils.jl"))
using .LoggingUtils: setup_logger, log_info, log_section

include(joinpath(@__DIR__, "src", "DataGeneration.jl"))
include(joinpath(@__DIR__, "src", "Analysis.jl"))
include(joinpath(@__DIR__, "src", "Validation.jl"))
include(joinpath(@__DIR__, "src", "Visualization.jl"))

using .DataGeneration
using .Analysis
using .Validation
using .Visualization

# ============================================================
# Initialization and Constraints
# ============================================================

function create_initialization()
    @initialization begin
        μ(sprinkler) = Categorical([0.5, 0.5])
    end
end

function create_learning_initialization()
    @initialization begin
        q(cpt_cloud_rain) = DirichletCollection(ones(2, 2))
        q(cpt_cloud_sprinkler) = DirichletCollection(ones(2, 2))
        q(cpt_sprinkler_rain_wet_grass) = DirichletCollection(ones(2, 2, 2))
        for init in sprinkler_model_cpt
            μ(sprinkler) = Categorical([0.5, 0.5])
        end
    end
end

function create_learning_constraints()
    @constraints begin
        for q in sprinkler_model_cpt
            q(cpt_cloud_rain, clouded, rain) = q(clouded, rain)q(cpt_cloud_rain)
            q(cpt_cloud_sprinkler, clouded, sprinkler) = q(clouded, sprinkler)q(cpt_cloud_sprinkler)
            q(cpt_sprinkler_rain_wet_grass, sprinkler, rain, wet_grass) = q(sprinkler, rain, wet_grass)q(cpt_sprinkler_rain_wet_grass)
        end
    end
end

# ============================================================
# Experiment Functions
# ============================================================

function run_inference_experiments(logger, output_dir::String, plot_config::PlotConfig)
    logger("="^50)
    logger("INFERENCE EXPERIMENTS")
    logger("="^50)
    
    iterations = CONFIG["inference"]["iterations"]
    init = create_initialization()
    scenarios = CONFIG["scenarios"]
    figures_dir = joinpath(output_dir, get(CONFIG["reporting"], "figures_subdir", "figures"))
    mkpath(figures_dir)
    
    results = Dict()
    
    for scenario in scenarios
        name = scenario["name"]
        desc = scenario["description"]
        logger("Scenario: $desc")
        
        model_type = get(scenario, "model", "basic")
        variables = Symbol.(scenario["variables"])
        
        # Build evidence
        evidence = scenario["evidence"]
        data = Dict{Symbol,Any}()
        for (k, v) in evidence
            if v == "missing"
                data[Symbol(k)] = missing
            else
                data[Symbol(k)] = v
            end
        end
        
        # Run inference
        if model_type == "basic"
            result = infer(model=sprinkler_model_basic(), data=(wet_grass=data[:wet_grass],), 
                          iterations=iterations, initialization=init)
        else
            result = infer(model=sprinkler_model_extended(), data=NamedTuple(data),
                          iterations=iterations, initialization=init)
        end
        
        # Plot
        output_path = joinpath(figures_dir, "posterior_$(name).png")
        plot_posterior_bars_enhanced(result.posteriors, variables, DEFAULT_LABELS, desc, plot_config,
            output_path=output_path)
        
        results[name] = result
        logger("  → Saved: posterior_$(name).png")
    end
    
    logger("Inference experiments completed: $(length(scenarios)) scenarios")
    return results
end

function run_learning_experiment(logger, output_dir::String, plot_config::PlotConfig)
    logger("="^50)
    logger("CPT LEARNING EXPERIMENT")
    logger("="^50)
    
    figures_dir = joinpath(output_dir, get(CONFIG["reporting"], "figures_subdir", "figures"))
    data_dir = joinpath(output_dir, get(CONFIG["reporting"], "data_subdir", "data"))
    mkpath(figures_dir)
    mkpath(data_dir)
    
    n_samples = CONFIG["learning"]["n_samples"]
    iterations = CONFIG["learning"]["learning_iterations"]
    stack_depth = CONFIG["learning"]["limit_stack_depth"]
    seed = CONFIG["general"]["seed"]
    
    # Generate data using config parameters
    logger("Generating $n_samples synthetic samples...")
    params = get_model_params(CONFIG)
    data = prepare_learning_data(n_samples, params, seed=seed)
    
    # Log empirical statistics
    samples = generate_samples(n_samples, params, seed=seed)
    stats = sample_statistics(samples)
    logger("Sample statistics:")
    logger("  P(Cloudy) empirical: $(round(stats["p_cloudy_empirical"], digits=3))")
    logger("  P(Rain) empirical: $(round(stats["p_rain_empirical"], digits=3))")
    
    # Run learning
    logger("Running variational inference ($iterations iterations)...")
    result = infer(model=learn_sprinkler_model(), data=data, constraints=create_learning_constraints(),
        initialization=create_learning_initialization(), iterations=iterations, showprogress=false,
        options=(limit_stack_depth=stack_depth,))
    
    # Analyze results
    logger("Analyzing results...")
    analysis = analyze_learning_result(result, CONFIG)
    
    for (key, val) in analysis.learned_values
        true_val = analysis.true_values[key]
        logger("  $key: $(round(val, digits=4)) (true: $true_val)")
    end
    
    # Validation
    validation_results = ValidationResult[]
    threshold = CONFIG["validation"]["max_cpt_error"]
    
    # Validate rain CPT
    cpt_cr = analysis.learned_cpts["cpt_cloud_rain"]
    true_cr = get_true_cpts_from_config(CONFIG)["P(Rain|Cloudy)"]
    push!(validation_results, validate_cpt_accuracy(cpt_cr, true_cr, threshold, name="P(Rain|Cloudy)"))
    
    # Validate sprinkler CPT
    cpt_cs = analysis.learned_cpts["cpt_cloud_sprinkler"]
    true_cs = get_true_cpts_from_config(CONFIG)["P(Sprinkler|Cloudy)"]
    push!(validation_results, validate_cpt_accuracy(cpt_cs, true_cs, threshold, name="P(Sprinkler|Cloudy)"))
    
    overall = validate_all(validation_results)
    logger(overall.message)
    
    # Visualizations
    logger("Generating visualizations...")
    
    # CPT heatmaps
    learned_cpts = Dict(
        "P(Rain|Cloudy)" => cpt_cr,
        "P(Sprinkler|Cloudy)" => cpt_cs
    )
    plot_cpt_heatmaps(learned_cpts, plot_config, output_path=joinpath(figures_dir, "learned_cpts.png"))
    
    # CPT comparison
    plot_cpt_comparison(learned_cpts, get_true_cpts_from_config(CONFIG), plot_config,
        output_path=joinpath(figures_dir, "cpt_comparison.png"))
    
    # Reports
    if CONFIG["reporting"]["generate_text_report"]
        generate_analysis_report(analysis, validation_results, joinpath(output_dir, "analysis_report.txt"))
    end
    
    if CONFIG["reporting"]["generate_json_summary"]
        generate_json_summary(analysis, validation_results, CONFIG, joinpath(data_dir, "summary.json"))
    end
    
    return (result=result, analysis=analysis, validation=validation_results)
end

# ============================================================
# Main Entry Point
# ============================================================

function main()
    println("="^70)
    println("BAYESIAN NETWORKS: THE SPRINKLER MODEL (Enhanced)")
    println("="^70)
    
    # Setup directories
    output_dir = joinpath(@__DIR__, CONFIG["general"]["output_dir"])
    figures_dir = joinpath(output_dir, get(CONFIG["reporting"], "figures_subdir", "figures"))
    logs_dir = joinpath(output_dir, get(CONFIG["reporting"], "logs_subdir", "logs"))
    data_dir = joinpath(output_dir, get(CONFIG["reporting"], "data_subdir", "data"))
    
    mkpath(figures_dir)
    mkpath(logs_dir)
    mkpath(data_dir)
    
    println("\nConfiguration: $CONFIG_PATH")
    println("Output: $output_dir")
    println("  ├── figures/")
    println("  ├── logs/")
    println("  └── data/")
    
    # Setup
    logger = setup_logger(logs_dir, CONFIG_PATH, CONFIG["general"]["seed"])
    plot_config = PlotConfig(CONFIG)
    setup_plot_theme(plot_config)
    
    logger("BAYESIAN NETWORKS: SPRINKLER MODEL (Enhanced)")
    logger("Seed: $(CONFIG["general"]["seed"])")
    
    # Run experiments
    inference_results = run_inference_experiments(logger, output_dir, plot_config)
    learning_result = run_learning_experiment(logger, output_dir, plot_config)
    
    # Final status
    validation_passed = all(r -> r.passed, learning_result.validation)
    status = validation_passed ? "PASS" : "FAIL"
    
    logger("="^50)
    logger("COMPLETED [$status]")
    logger("="^50)
    
    println("\n" * "="^70)
    println("COMPLETED [$status]")
    println("="^70)
    println("\nOutputs saved to: $output_dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end