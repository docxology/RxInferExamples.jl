# Bayesian Multinomial Regression - Enhanced Version
# 
# This example demonstrates Bayesian inference for multinomial regression using RxInfer.jl.
# Features:
# - Multinomial Classification (learning category probabilities)
# - Multinomial Regression (learning coefficients with covariates)
# - Stick-breaking parameterization for simplex constraints
# - Comprehensive visualization and validation
#
# Configuration is loaded from config.toml

# ============================================================
# Environment Setup
# ============================================================
ENV["GKSwstype"] = "100"  # Non-interactive backend for headless operation

using RxInfer, Plots, StableRNGs, Distributions, ExponentialFamily, StatsPlots
using Statistics, LinearAlgebra, TOML, Printf, JSON

import ExponentialFamily: softmax

# ============================================================
# Configuration Loading
# ============================================================
const SCRIPT_DIR = @__DIR__
const CONFIG_PATH = joinpath(SCRIPT_DIR, "config.toml")
const CONFIG = TOML.parsefile(CONFIG_PATH)

# Setup output directories
const OUTPUT_DIR = joinpath(SCRIPT_DIR, get(CONFIG["general"], "output_dir", "output"))
const FIGURES_DIR = joinpath(OUTPUT_DIR, get(CONFIG["reporting"], "figures_subdir", "figures"))
const LOGS_DIR = joinpath(OUTPUT_DIR, get(CONFIG["reporting"], "logs_subdir", "logs"))
mkpath(FIGURES_DIR)
mkpath(LOGS_DIR)

@info "Configuration loaded from $CONFIG_PATH"
@info "Output directory: $OUTPUT_DIR"

# ============================================================
# Model Definitions (at top level for macro expansion)
# ============================================================

# multinomial_model: Multinomial classification without covariates
# Learns natural parameters ψ directly from observed counts
@model function multinomial_model(obs, N, ξ_ψ, W_ψ)
    ψ ~ MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ)
    obs .~ MultinomialPolya(N, ψ) where {
        dependencies = RequireMessageFunctionalDependencies(
            ψ = MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ)
        )
    }
end

# multinomial_regression: Multinomial regression with covariates
# Learns regression coefficients β from observed counts and features
@model function multinomial_regression(obs, N, X, ϕ, ξβ, Wβ)
    β ~ MvNormalWeightedMeanPrecision(ξβ, Wβ)
    for i in eachindex(obs)
        Ψ[i] := ϕ(X[i]) * β
        obs[i] ~ MultinomialPolya(N, Ψ[i]) where {
            dependencies = RequireMessageFunctionalDependencies(
                ψ = MvNormalWeightedMeanPrecision(zeros(length(obs[i])-1), diageye(length(obs[i])-1))
            )
        }
    end
end

# ============================================================
# Include Modules
# ============================================================
include("src/SimplexUtils.jl")
include("src/DataGeneration.jl")
include("src/Visualization.jl")
include("src/Analysis.jl")
include("src/Validation.jl")
include("src/ExtendedStatistics.jl")
include("src/ExtendedVisualization.jl")
include("src/ExtendedReporting.jl")

using .SimplexUtils
using .DataGeneration
using .Visualization
using .Analysis
using .Validation
using .ExtendedStatistics
using .ExtendedVisualization
using .ExtendedReporting

# ============================================================
# Utility: Logistic Stick Breaking
# ============================================================
σ(x) = 1 / (1 + exp(-x))

function logistic_stick_breaking(ψ::AbstractVector)
    K = length(ψ) + 1
    π = zeros(K)
    remaining = 1.0
    for k in 1:(K-1)
        π[k] = σ(ψ[k]) * remaining
        remaining -= π[k]
    end
    π[K] = remaining
    return π
end

# ============================================================
# Main Experiment Functions
# ============================================================

"""
    run_classification_experiment(config) -> NamedTuple

Run the multinomial classification experiment.
"""
function run_classification_experiment(config::Dict)
    cls = config["classification"]
    seed = config["general"]["seed"]
    
    @info "Running Classification Experiment" N=cls["N"] k=cls["k"] nsamples=cls["nsamples"]
    
    # Generate data
    rng = StableRNG(seed)
    Ψ_true = randn(rng, cls["k"])
    p_true = softmax(Ψ_true)
    X = rand(rng, Multinomial(cls["N"], p_true), cls["nsamples"])
    X = [X[:, i] for i in 1:size(X, 2)]
    
    # Run inference
    k = cls["k"]
    result = infer(
        model = multinomial_model(
            ξ_ψ = zeros(k-1),
            W_ψ = rand(rng, Wishart(cls["wishart_df"], diageye(k-1))),
            N = cls["N"]
        ),
        data = (obs = X,),
        iterations = cls["iterations"],
        free_energy = true,
        showprogress = true,
        options = (limit_stack_depth = 100,)
    )
    
    # Get predictive distribution
    meta_samples = config["prediction"]["polya_meta_samples"]
    predictive = @call_rule MultinomialPolya(:x, Marginalisation) (
        q_N = PointMass(cls["N"]),
        q_ψ = result.posteriors[:ψ][end],
        meta = MultinomialPolyaMeta(meta_samples)
    )
    
    estimated_probs = predictive.p
    free_energy = result.free_energy
    
    @info "Classification Complete" mse=mean((estimated_probs .- p_true).^2)
    
    return (
        estimated_probs = estimated_probs,
        true_probs = p_true,
        free_energy = free_energy,
        Ψ_true = Ψ_true,
        data = X
    )
end

"""
    run_regression_experiment(config) -> NamedTuple

Run the multinomial regression experiment.
"""
function run_regression_experiment(config::Dict)
    reg = config["regression"]
    seed = config["general"]["seed"]
    
    @info "Running Regression Experiment" k=reg["k"] nsamples=reg["nsamples"]
    
    # Parse transform
    transform_name = get(reg, "feature_transform", "identity")
    ϕ = if transform_name == "sin"
        sin
    elseif transform_name == "tanh"
        tanh
    else
        identity
    end
    
    # Generate data
    rng = StableRNG(seed)
    β_true = randn(rng, reg["k"])
    X_raw = randn(rng, reg["nsamples"], reg["k"], reg["k"])
    X = [X_raw[i, :, :] for i in 1:size(X_raw, 1)]
    Ψ = ϕ.(X)
    p = map(x -> logistic_stick_breaking(x * β_true), Ψ)
    obs = map(prob -> rand(rng, Multinomial(reg["N"], prob)), p)
    
    # Run inference
    k = reg["k"]
    result = infer(
        model = multinomial_regression(
            N = reg["N"],
            ϕ = ϕ,
            ξβ = zeros(k),
            Wβ = rand(rng, Wishart(reg["wishart_df"], diageye(k)))
        ),
        data = (obs = obs, X = X),
        iterations = reg["iterations"],
        free_energy = true,
        showprogress = true,
        returnvars = KeepLast(),
        options = (limit_stack_depth = 100,)
    )
    
    # Extract results
    posterior_β = result.posteriors[:β]
    estimated_mean = mean(posterior_β)
    estimated_cov = cov(posterior_β)
    
    @info "Regression Complete" mse=mean((estimated_mean .- β_true).^2)
    
    return (
        estimated_mean = estimated_mean,
        estimated_cov = estimated_cov,
        true_beta = β_true,
        free_energy = result.free_energy,
        obs = obs,
        X = X
    )
end

# ============================================================
# Main Execution
# ============================================================

function main()
    @info "="^60
    @info "BAYESIAN MULTINOMIAL REGRESSION"
    @info "="^60
    
    validation_results = ValidationResult[]
    
    # --- Classification Experiment ---
    cls_result = run_classification_experiment(CONFIG)
    
    # Visualization
    plot_free_energy(cls_result.free_energy;
        title_suffix = "(Classification)",
        output_path = joinpath(FIGURES_DIR, "classification_free_energy.png"))
    
    plot_probability_comparison(cls_result.estimated_probs, cls_result.true_probs;
        output_path = joinpath(FIGURES_DIR, "probability_comparison.png"))
    
    # Analysis
    cls_analysis = analyze_classification_result(
        cls_result.estimated_probs,
        cls_result.true_probs,
        cls_result.free_energy
    )
    
    # Validation
    cls_threshold = CONFIG["validation"]["max_probability_mse"]
    push!(validation_results, validate_classification(cls_analysis.mse, cls_threshold))
    
    # --- Regression Experiment ---
    reg_result = run_regression_experiment(CONFIG)
    
    # Visualization
    plot_free_energy(reg_result.free_energy;
        title_suffix = "(Regression)",
        output_path = joinpath(FIGURES_DIR, "regression_free_energy.png"))
    
    estimated_std = sqrt.(diag(reg_result.estimated_cov))
    plot_beta_comparison(reg_result.estimated_mean, estimated_std, reg_result.true_beta;
        output_path = joinpath(FIGURES_DIR, "beta_comparison.png"))
    
    # Analysis
    reg_analysis = analyze_regression_result(
        reg_result.estimated_mean,
        reg_result.estimated_cov,
        reg_result.true_beta,
        reg_result.free_energy
    )
    
    # Validation
    reg_threshold = CONFIG["validation"]["max_beta_mse"]
    push!(validation_results, validate_regression(reg_analysis.mse, reg_threshold))
    
    # --- Simplex Density Visualization ---
    plot_config = PlotConfig(CONFIG)
    plot_simplex_densities(plot_config;
        output_path = joinpath(FIGURES_DIR, "simplex_densities.png"))
    
    # --- Generate Reports ---
    generate_analysis_report(
        cls_analysis, reg_analysis, validation_results,
        joinpath(LOGS_DIR, "analysis_report.txt")
    )
    
    generate_json_summary(
        cls_analysis, reg_analysis, validation_results, CONFIG,
        joinpath(LOGS_DIR, "summary.json")
    )
    
    generate_validation_report(validation_results,
        joinpath(LOGS_DIR, "validation_report.txt"))
    
    # --- Extended Statistics ---
    @info "Computing extended statistics..."
    
    # Regression metrics
    reg_metrics = compute_regression_statistics(reg_result.estimated_mean, reg_result.true_beta)
    
    # Multivariate metrics
    mv_metrics = compute_multivariate_statistics(reg_result.estimated_cov)
    
    # Convergence metrics
    cls_conv = compute_convergence_statistics(cls_result.free_energy)
    reg_conv = compute_convergence_statistics(reg_result.free_energy)
    
    # Credible intervals
    cred_intervals = compute_credible_intervals(reg_result.estimated_mean, estimated_std)
    
    # Coverage at multiple levels
    cov_50 = compute_coverage(reg_result.estimated_mean, estimated_std, reg_result.true_beta; level=0.5)
    cov_90 = compute_coverage(reg_result.estimated_mean, estimated_std, reg_result.true_beta; level=0.9)
    cov_95 = compute_coverage(reg_result.estimated_mean, estimated_std, reg_result.true_beta; level=0.95)
    
    # Classification extended stats
    cls_metrics = compute_regression_statistics(cls_result.estimated_probs, cls_result.true_probs)
    
    # Residual analysis  
    errors = reg_result.estimated_mean .- reg_result.true_beta
    resid_stats = compute_residual_statistics(errors)
    
    @info "Extended statistics computed"
    @info "Regression R²: $(reg_metrics.r_squared)"
    @info "Spearman correlation: $(reg_metrics.spearman)"
    @info "Condition number: $(mv_metrics.condition_number)"
    @info "Skewness: $(resid_stats.skewness)"
    @info "Kurtosis: $(resid_stats.kurtosis)"
    
    # --- Extended Visualizations ---
    @info "Generating extended visualizations..."
    
    # Coefficient forest plot
    coef_names = ["β$(i)" for i in 1:length(reg_result.true_beta)]
    plot_coefficient_forest(coef_names, reg_result.estimated_mean, estimated_std, reg_result.true_beta;
        output_path = joinpath(FIGURES_DIR, "coefficient_forest.png"))
    
    # Comprehensive dashboard
    plot_comprehensive_dashboard(reg_result.estimated_mean, estimated_std, 
        reg_result.true_beta, reg_result.free_energy;
        output_path = joinpath(FIGURES_DIR, "comprehensive_dashboard.png"))
    
    # Convergence diagnostics
    plot_convergence_diagnostics(cls_result.free_energy;
        output_path = joinpath(FIGURES_DIR, "classification_convergence_diagnostics.png"))
    plot_convergence_diagnostics(reg_result.free_energy;
        output_path = joinpath(FIGURES_DIR, "regression_convergence_diagnostics.png"))
    
    # Error distribution
    plot_error_distribution(errors;
        output_path = joinpath(FIGURES_DIR, "error_distribution.png"))
    
    # Q-Q plot
    plot_qq_normal(errors;
        title = "Regression Error Q-Q Plot",
        output_path = joinpath(FIGURES_DIR, "error_qq_plot.png"))
    
    # Correlation heatmap of posterior
    corr_matrix = compute_correlation_matrix(reg_result.estimated_cov)
    plot_correlation_heatmap(corr_matrix, coef_names;
        output_path = joinpath(FIGURES_DIR, "posterior_correlation.png"))
    
    @info "Extended visualizations generated"
    
    # --- Extended Reports ---
    @info "Generating extended reports..."
    
    generate_comprehensive_report(
        cls_stats = cls_analysis,
        reg_stats = reg_analysis,
        validation_results = validation_results,
        config = CONFIG,
        output_path = joinpath(LOGS_DIR, "comprehensive_report.txt")
    )
    
    generate_detailed_json(
        cls_stats = cls_analysis,
        reg_stats = reg_analysis,
        validation_results = validation_results,
        config = CONFIG,
        output_path = joinpath(LOGS_DIR, "detailed_analysis.json")
    )
    
    generate_markdown_report(
        cls_stats = cls_analysis,
        reg_stats = reg_analysis,
        validation_results = validation_results,
        config = CONFIG,
        figures_dir = FIGURES_DIR,
        output_path = joinpath(LOGS_DIR, "analysis_report.md")
    )
    
    # Extended statistics JSON
    extended_stats = Dict(
        "regression_metrics" => Dict(
            "mse" => reg_metrics.mse,
            "rmse" => reg_metrics.rmse,
            "mae" => reg_metrics.mae,
            "r_squared" => reg_metrics.r_squared,
            "explained_variance" => reg_metrics.explained_variance,
            "pearson" => reg_metrics.pearson,
            "spearman" => reg_metrics.spearman,
            "kendall_tau" => reg_metrics.kendall_tau,
            "max_error" => reg_metrics.max_error,
            "bias" => reg_metrics.bias
        ),
        "classification_metrics" => Dict(
            "mse" => cls_metrics.mse,
            "rmse" => cls_metrics.rmse,
            "pearson" => cls_metrics.pearson,
            "spearman" => cls_metrics.spearman
        ),
        "multivariate" => Dict(
            "condition_number" => mv_metrics.condition_number,
            "effective_dimension" => mv_metrics.effective_dimension,
            "generalized_variance" => mv_metrics.generalized_variance,
            "total_variance" => mv_metrics.total_variance
        ),
        "convergence" => Dict(
            "classification" => Dict(
                "final_value" => cls_conv.final_value,
                "converged" => cls_conv.converged,
                "rate" => cls_conv.convergence_rate
            ),
            "regression" => Dict(
                "final_value" => reg_conv.final_value,
                "converged" => reg_conv.converged,
                "rate" => reg_conv.convergence_rate
            )
        ),
        "coverage" => Dict(
            "50_percent" => cov_50,
            "90_percent" => cov_90,
            "95_percent" => cov_95
        ),
        "residuals" => Dict(
            "mean" => resid_stats.mean,
            "std" => resid_stats.std,
            "skewness" => resid_stats.skewness,
            "kurtosis" => resid_stats.kurtosis,
            "min" => resid_stats.min,
            "max" => resid_stats.max,
            "iqr" => resid_stats.iqr,
            "jarque_bera" => resid_stats.jarque_bera,
            "durbin_watson" => resid_stats.durbin_watson
        )
    )
    
    open(joinpath(LOGS_DIR, "extended_statistics.json"), "w") do io
        JSON.print(io, extended_stats, 2)
    end
    @info "Extended statistics saved: $(joinpath(LOGS_DIR, "extended_statistics.json"))"
    
    # --- Final Summary ---
    overall = validate_all(validation_results)
    @info "="^60
    @info overall.message
    @info "="^60
    
    return (
        classification = cls_result,
        regression = reg_result,
        cls_analysis = cls_analysis,
        reg_analysis = reg_analysis,
        validation = validation_results,
        overall_passed = overall.passed
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end