"""
ExtendedStatistics - Comprehensive statistical analysis for regression and multivariate data.

Provides a complete suite of statistical metrics for Bayesian regression analysis.
"""
module ExtendedStatistics

using Statistics
using LinearAlgebra
using Printf

export RegressionMetrics, MultivariateMetrics, ConvergenceMetrics
export compute_regression_statistics, compute_multivariate_statistics
export compute_convergence_statistics, compute_all_statistics
export compute_information_criteria, compute_effect_sizes
export compute_credible_intervals, compute_hdi, compute_coverage
export compute_residual_statistics, compute_leverage_statistics
export compute_correlation_matrix, compute_interval_width
export compute_mse, compute_rmse, compute_mae, compute_r_squared
export compute_pearson, compute_spearman, compute_kendall_tau

# ============================================================
# Core Error Metrics
# ============================================================

"""
    compute_mse(estimated, true_values) -> Float64

Mean Squared Error.
"""
compute_mse(est, truth) = mean((est .- truth).^2)

"""
    compute_rmse(estimated, true_values) -> Float64

Root Mean Squared Error.
"""
compute_rmse(est, truth) = sqrt(compute_mse(est, truth))

"""
    compute_mae(estimated, true_values) -> Float64

Mean Absolute Error.
"""
compute_mae(est, truth) = mean(abs.(est .- truth))

"""
    compute_mape(estimated, true_values) -> Float64

Mean Absolute Percentage Error.
"""
function compute_mape(est, truth)
    valid = abs.(truth) .> 1e-10
    any(valid) ? mean(abs.((est[valid] .- truth[valid]) ./ truth[valid])) * 100 : NaN
end

"""
    compute_smape(estimated, true_values) -> Float64

Symmetric Mean Absolute Percentage Error.
"""
function compute_smape(est, truth)
    denom = abs.(est) .+ abs.(truth)
    valid = denom .> 1e-10
    any(valid) ? 200 * mean(abs.(est[valid] .- truth[valid]) ./ denom[valid]) : NaN
end

"""
    compute_max_error(estimated, true_values) -> Float64

Maximum Absolute Error.
"""
compute_max_error(est, truth) = maximum(abs.(est .- truth))

"""
    compute_bias(estimated, true_values) -> Float64

Mean Bias (systematic error direction).
"""
compute_bias(est, truth) = mean(est .- truth)

# ============================================================
# R² and Explained Variance
# ============================================================

"""
    compute_r_squared(estimated, true_values) -> Float64

Coefficient of Determination (R²).
"""
function compute_r_squared(est, truth)
    ss_res = sum((truth .- est).^2)
    ss_tot = sum((truth .- mean(truth)).^2)
    ss_tot > 0 ? 1 - ss_res/ss_tot : NaN
end

"""
    compute_adjusted_r_squared(r2, n, p) -> Float64

Adjusted R² accounting for number of predictors.
"""
function compute_adjusted_r_squared(r2::Float64, n::Int, p::Int)
    n > p + 1 ? 1 - (1 - r2) * (n - 1) / (n - p - 1) : NaN
end

"""
    compute_explained_variance(estimated, true_values) -> Float64

Explained Variance Score.
"""
function compute_explained_variance(est, truth)
    residual_var = var(truth .- est)
    total_var = var(truth)
    total_var > 0 ? 1 - residual_var/total_var : NaN
end

# ============================================================
# Correlation Metrics
# ============================================================

"""
    compute_pearson(estimated, true_values) -> Float64

Pearson Correlation Coefficient.
"""
compute_pearson(est, truth) = cor(est, truth)

"""
    compute_spearman(estimated, true_values) -> Float64

Spearman Rank Correlation.
"""
function compute_spearman(est, truth)
    rank_est = sortperm(sortperm(est))
    rank_truth = sortperm(sortperm(truth))
    cor(Float64.(rank_est), Float64.(rank_truth))
end

"""
    compute_kendall_tau(estimated, true_values) -> Float64

Kendall's Tau-b Correlation.
"""
function compute_kendall_tau(est, truth)
    n = length(est)
    concordant = 0
    discordant = 0
    for i in 1:n
        for j in (i+1):n
            sign_est = sign(est[j] - est[i])
            sign_truth = sign(truth[j] - truth[i])
            if sign_est * sign_truth > 0
                concordant += 1
            elseif sign_est * sign_truth < 0
                discordant += 1
            end
        end
    end
    total = concordant + discordant
    total > 0 ? (concordant - discordant) / total : 0.0
end

# ============================================================
# Bayesian Uncertainty Metrics
# ============================================================

"""
    compute_credible_intervals(mean, std; levels=[0.5, 0.9, 0.95, 0.99]) -> Dict

Compute credible intervals at multiple levels assuming Gaussian posterior.
"""
function compute_credible_intervals(μ::AbstractVector, σ::AbstractVector; 
                                   levels=[0.5, 0.9, 0.95, 0.99])
    # Z-scores for symmetric intervals
    z_scores = Dict(0.5 => 0.6745, 0.9 => 1.645, 0.95 => 1.96, 0.99 => 2.576)
    
    result = Dict{Float64, Vector{Tuple{Float64,Float64}}}()
    for level in levels
        z = get(z_scores, level, 1.96)
        result[level] = [(μ[i] - z*σ[i], μ[i] + z*σ[i]) for i in eachindex(μ)]
    end
    return result
end

"""
    compute_hdi(samples; prob=0.95) -> Tuple{Float64, Float64}

Compute Highest Density Interval from samples.
"""
function compute_hdi(samples::AbstractVector; prob=0.95)
    sorted = sort(samples)
    n = length(sorted)
    interval_size = floor(Int, prob * n)
    
    min_width = Inf
    best_start = 1
    
    for start in 1:(n - interval_size)
        width = sorted[start + interval_size] - sorted[start]
        if width < min_width
            min_width = width
            best_start = start
        end
    end
    
    return (sorted[best_start], sorted[best_start + interval_size])
end

"""
    compute_coverage(estimated_mean, estimated_std, true_values; level=0.95) -> Float64

Compute empirical coverage of credible intervals.
"""
function compute_coverage(μ::AbstractVector, σ::AbstractVector, truth::AbstractVector; level=0.95)
    z = level == 0.95 ? 1.96 : (level == 0.9 ? 1.645 : 2.576)
    within = abs.(μ .- truth) .< z .* σ
    return mean(within)
end

"""
    compute_interval_width(std; level=0.95) -> Vector{Float64}

Average credible interval width.
"""
function compute_interval_width(σ::AbstractVector; level=0.95)
    z = level == 0.95 ? 1.96 : (level == 0.9 ? 1.645 : 2.576)
    return 2 * z .* σ
end

# ============================================================
# Multivariate Statistics
# ============================================================

"""
    compute_mahalanobis(x, mean, cov) -> Float64

Mahalanobis distance.
"""
function compute_mahalanobis(x::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix)
    diff = x .- μ
    return sqrt(diff' * inv(Σ) * diff)
end

"""
    compute_condition_number(cov) -> Float64

Condition number of covariance matrix.
"""
function compute_condition_number(Σ::AbstractMatrix)
    ev = eigvals(Σ)
    ev_real = real.(ev)
    min_ev = minimum(ev_real)
    max_ev = maximum(ev_real)
    min_ev > 0 ? max_ev / min_ev : Inf
end

"""
    compute_effective_dimension(cov) -> Float64

Effective dimensionality via participation ratio.
"""
function compute_effective_dimension(Σ::AbstractMatrix)
    ev = max.(real.(eigvals(Σ)), 0.0)
    total = sum(ev)
    total > 0 ? sum(ev)^2 / sum(ev.^2) : 0.0
end

"""
    compute_generalized_variance(cov) -> Float64

Generalized variance (determinant of covariance).
"""
compute_generalized_variance(Σ::AbstractMatrix) = det(Σ)

"""
    compute_total_variance(cov) -> Float64

Total variance (trace of covariance).
"""
compute_total_variance(Σ::AbstractMatrix) = tr(Σ)

"""
    compute_correlation_matrix(cov) -> Matrix{Float64}

Convert covariance to correlation matrix.
"""
function compute_correlation_matrix(Σ::AbstractMatrix)
    d = sqrt.(diag(Σ))
    D_inv = Diagonal(1 ./ d)
    return D_inv * Σ * D_inv
end

# ============================================================
# Information Criteria (for model comparison)
# ============================================================

"""
    compute_dic(deviance_mean, deviance_at_mean) -> Float64

Deviance Information Criterion.
DIC = D̄ + pD where pD = D̄ - D(θ̄)
"""
function compute_dic(deviance_mean::Float64, deviance_at_mean::Float64)
    p_d = deviance_mean - deviance_at_mean  # Effective number of parameters
    return deviance_mean + p_d
end

"""
    compute_waic(log_likelihoods) -> NamedTuple

Widely Applicable Information Criterion from pointwise log-likelihoods.
"""
function compute_waic(log_liks::AbstractMatrix)
    # log_liks: samples × observations
    lppd = sum(log.(mean(exp.(log_liks), dims=1)))
    p_waic = sum(var(log_liks, dims=1))
    waic = -2 * (lppd - p_waic)
    return (waic=waic, lppd=lppd, p_waic=p_waic)
end

# ============================================================
# Effect Size Metrics
# ============================================================

"""
    compute_cohens_d(group1, group2) -> Float64

Cohen's d effect size.
"""
function compute_cohens_d(g1::AbstractVector, g2::AbstractVector)
    pooled_std = sqrt((var(g1) + var(g2)) / 2)
    pooled_std > 0 ? (mean(g1) - mean(g2)) / pooled_std : 0.0
end

"""
    compute_standardized_beta(beta, std_x, std_y) -> Vector{Float64}

Compute standardized regression coefficients.
"""
function compute_standardized_beta(β::AbstractVector, σ_x::AbstractVector, σ_y::Float64)
    return β .* σ_x ./ σ_y
end

# ============================================================
# Convergence and Diagnostics
# ============================================================

"""
    compute_convergence_rate(free_energy) -> Float64

Estimate convergence rate from free energy sequence.
"""
function compute_convergence_rate(fe::AbstractVector)
    n = length(fe)
    n < 3 && return NaN
    
    # Fit exponential decay to |ΔFE|
    diffs = abs.(diff(fe))
    diffs = max.(diffs, 1e-10)  # Avoid log(0)
    log_diffs = log.(diffs)
    
    # Simple linear fit to log(diff) vs iteration
    x = 1:length(log_diffs)
    slope = cor(Float64.(collect(x)), log_diffs) * std(log_diffs) / std(x)
    
    return -slope  # Positive = converging
end

"""
    compute_effective_sample_size(samples) -> Float64

Effective sample size from autocorrelation.
"""
function compute_effective_sample_size(samples::AbstractVector)
    n = length(samples)
    n < 10 && return Float64(n)
    
    # Compute autocorrelation
    centered = samples .- mean(samples)
    var_s = var(samples)
    var_s ≈ 0 && return Float64(n)
    
    # Sum of autocorrelations (truncated)
    max_lag = min(n ÷ 2, 50)
    autocorr_sum = 0.0
    for lag in 1:max_lag
        r = sum(centered[1:end-lag] .* centered[lag+1:end]) / ((n - lag) * var_s)
        abs(r) < 0.05 && break
        autocorr_sum += r
    end
    
    return n / (1 + 2 * autocorr_sum)
end

"""
    compute_rhat(chains) -> Float64

Gelman-Rubin R̂ convergence diagnostic.
"""
function compute_rhat(chains::AbstractMatrix)
    # chains: iterations × chains
    n, m = size(chains)
    
    chain_means = mean(chains, dims=1)[:]
    chain_vars = var(chains, dims=1)[:]
    
    B = n * var(chain_means)  # Between-chain variance
    W = mean(chain_vars)      # Within-chain variance
    
    var_plus = ((n-1)/n) * W + B/n
    return sqrt(var_plus / W)
end

# ============================================================
# Residual Analysis
# ============================================================

"""
    compute_residual_statistics(residuals) -> NamedTuple

Comprehensive residual statistics.
"""
function compute_residual_statistics(residuals::AbstractVector)
    n = length(residuals)
    
    return (
        mean = mean(residuals),
        std = std(residuals),
        skewness = compute_skewness(residuals),
        kurtosis = compute_kurtosis(residuals),
        min = minimum(residuals),
        max = maximum(residuals),
        q25 = quantile(residuals, 0.25),
        median = median(residuals),
        q75 = quantile(residuals, 0.75),
        iqr = quantile(residuals, 0.75) - quantile(residuals, 0.25),
        jarque_bera = compute_jarque_bera(residuals),
        durbin_watson = compute_durbin_watson(residuals)
    )
end

"""
    compute_skewness(x) -> Float64

Sample skewness.
"""
function compute_skewness(x::AbstractVector)
    n = length(x)
    μ = mean(x)
    σ = std(x)
    σ > 0 ? sum(((x .- μ) ./ σ).^3) / n : 0.0
end

"""
    compute_kurtosis(x) -> Float64

Sample excess kurtosis.
"""
function compute_kurtosis(x::AbstractVector)
    n = length(x)
    μ = mean(x)
    σ = std(x)
    σ > 0 ? sum(((x .- μ) ./ σ).^4) / n - 3 : 0.0
end

"""
    compute_jarque_bera(x) -> Float64

Jarque-Bera test statistic for normality.
"""
function compute_jarque_bera(x::AbstractVector)
    n = length(x)
    S = compute_skewness(x)
    K = compute_kurtosis(x)
    return (n/6) * (S^2 + K^2/4)
end

"""
    compute_durbin_watson(residuals) -> Float64

Durbin-Watson statistic for autocorrelation.
"""
function compute_durbin_watson(residuals::AbstractVector)
    n = length(residuals)
    n < 2 && return NaN
    
    diffs_sq = sum(diff(residuals).^2)
    resid_sq = sum(residuals.^2)
    resid_sq > 0 ? diffs_sq / resid_sq : NaN
end

# ============================================================
# Aggregate Statistics Functions
# ============================================================

"""
    struct RegressionMetrics

Container for all regression-related metrics.
"""
struct RegressionMetrics
    mse::Float64
    rmse::Float64
    mae::Float64
    mape::Float64
    smape::Float64
    max_error::Float64
    bias::Float64
    r_squared::Float64
    explained_variance::Float64
    pearson::Float64
    spearman::Float64
    kendall_tau::Float64
end

"""
    compute_regression_statistics(estimated, true_values) -> RegressionMetrics

Compute all regression metrics.
"""
function compute_regression_statistics(est::AbstractVector, truth::AbstractVector)
    RegressionMetrics(
        compute_mse(est, truth),
        compute_rmse(est, truth),
        compute_mae(est, truth),
        compute_mape(est, truth),
        compute_smape(est, truth),
        compute_max_error(est, truth),
        compute_bias(est, truth),
        compute_r_squared(est, truth),
        compute_explained_variance(est, truth),
        compute_pearson(est, truth),
        compute_spearman(est, truth),
        compute_kendall_tau(est, truth)
    )
end

"""
    struct MultivariateMetrics

Container for multivariate analysis metrics.
"""
struct MultivariateMetrics
    condition_number::Float64
    effective_dimension::Float64
    generalized_variance::Float64
    total_variance::Float64
    correlation_matrix::Matrix{Float64}
end

"""
    compute_multivariate_statistics(cov) -> MultivariateMetrics

Compute all multivariate metrics.
"""
function compute_multivariate_statistics(Σ::AbstractMatrix)
    MultivariateMetrics(
        compute_condition_number(Σ),
        compute_effective_dimension(Σ),
        compute_generalized_variance(Σ),
        compute_total_variance(Σ),
        compute_correlation_matrix(Σ)
    )
end

"""
    struct ConvergenceMetrics

Container for convergence diagnostics.
"""
struct ConvergenceMetrics
    final_value::Float64
    total_change::Float64
    final_change::Float64
    convergence_rate::Float64
    n_iterations::Int
    converged::Bool
end

"""
    compute_convergence_statistics(free_energy; threshold=1e-6) -> ConvergenceMetrics

Compute convergence diagnostics.
"""
function compute_convergence_statistics(fe::AbstractVector; threshold=1e-6)
    n = length(fe)
    final_change = n > 1 ? abs(fe[end] - fe[end-1]) : 0.0
    
    ConvergenceMetrics(
        fe[end],
        n > 1 ? abs(fe[end] - fe[1]) : 0.0,
        final_change,
        compute_convergence_rate(fe),
        n,
        final_change < threshold
    )
end

"""
    compute_all_statistics(est_mean, est_cov, truth, free_energy) -> NamedTuple

Compute all available statistics.
"""
function compute_all_statistics(est_mean::AbstractVector, 
                               est_cov::AbstractMatrix,
                               truth::AbstractVector,
                               free_energy::AbstractVector)
    std_est = sqrt.(diag(est_cov))
    
    return (
        regression = compute_regression_statistics(est_mean, truth),
        multivariate = compute_multivariate_statistics(est_cov),
        convergence = compute_convergence_statistics(free_energy),
        credible_intervals = compute_credible_intervals(est_mean, std_est),
        coverage_50 = compute_coverage(est_mean, std_est, truth; level=0.5),
        coverage_90 = compute_coverage(est_mean, std_est, truth; level=0.9),
        coverage_95 = compute_coverage(est_mean, std_est, truth; level=0.95),
        interval_widths_95 = compute_interval_width(std_est; level=0.95)
    )
end

end # module
