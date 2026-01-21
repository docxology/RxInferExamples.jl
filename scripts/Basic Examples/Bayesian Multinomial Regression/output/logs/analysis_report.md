# Bayesian Multinomial Regression Analysis Report

_Generated: 2026-01-20 16:08:45_

## Table of Contents
1. [Configuration](#configuration)
2. [Classification Results](#classification-results)
3. [Regression Results](#regression-results)
4. [Validation](#validation)
5. [Visualizations](#visualizations)

## Configuration

| Parameter | Value |
|-----------|-------|
| Seed | 123 |
| Classification N | 30 |
| Classification k | 40 |
| Classification samples | 5000 |
| Regression N | 3 |
| Regression k | 5 |
| Regression samples | 5000 |

## Classification Results

### Error Metrics
| Metric | Value |
|--------|-------|
| MSE | 0.000000 |
| RMSE | 0.000572 |
| Max Error | 0.001419 |
| Correlation | 0.999863 |

## Regression Results

### Error Metrics
| Metric | Value |
|--------|-------|
| MSE | 0.000078 |
| RMSE | 0.008811 |
| Max Error | 0.014263 |
| Mean Std | 0.012309 |
| 95% Coverage | 100.000000% |

## Validation

**Overall: 2/2 PASSED**

| Test | Status | Details |
|------|--------|---------|
| Probability MSE | ✅ PASS | Probability MSE: 0.000000 ≤ 0.010000 → PASS |
| Coefficient MSE | ✅ PASS | Coefficient MSE: 0.000078 ≤ 0.500000 → PASS |

## Visualizations

### beta_comparison
![beta_comparison.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/beta_comparison.png)

### classification_convergence_diagnostics
![classification_convergence_diagnostics.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/classification_convergence_diagnostics.png)

### classification_free_energy
![classification_free_energy.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/classification_free_energy.png)

### coefficient_forest
![coefficient_forest.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/coefficient_forest.png)

### comprehensive_dashboard
![comprehensive_dashboard.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/comprehensive_dashboard.png)

### error_distribution
![error_distribution.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/error_distribution.png)

### error_qq_plot
![error_qq_plot.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/error_qq_plot.png)

### posterior_correlation
![posterior_correlation.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/posterior_correlation.png)

### probability_comparison
![probability_comparison.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/probability_comparison.png)

### regression_convergence_diagnostics
![regression_convergence_diagnostics.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/regression_convergence_diagnostics.png)

### regression_free_energy
![regression_free_energy.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/regression_free_energy.png)

### simplex_densities
![simplex_densities.png](/Users/4d/Documents/GitHub/RxInferExamples.jl/scripts/Basic Examples/Bayesian Multinomial Regression/output/figures/simplex_densities.png)

