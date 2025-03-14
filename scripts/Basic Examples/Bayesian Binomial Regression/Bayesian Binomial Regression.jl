# This file was automatically generated from examples/Basic Examples/Bayesian Binomial Regression/Bayesian Binomial Regression.ipynb
# by notebooks_to_scripts.jl at 2025-03-14T05:52:01.915
#
# Source notebook: Bayesian Binomial Regression.ipynb

using RxInfer, ReactiveMP, Random, Plots, StableRNGs, LinearAlgebra, StatsPlots, LaTeXStrings

function generate_synthetic_binomial_data(
    n_samples::Int,
    true_beta::Vector{Float64};
    seed::Int=42
)
    n_features = length(true_beta)
    rng = StableRNG(seed)
    
    X = randn(rng, n_samples, n_features)
    
    n_trials = rand(rng, 5:20, n_samples)
    
    logits = X * true_beta
    probs = 1 ./ (1 .+ exp.(-logits))
    
    y = [rand(rng, Binomial(n_trials[i], probs[i])) for i in 1:n_samples]
    
    return X, y, n_trials, probs
end


n_samples = 10000
true_beta =  [-3.0 , 2.6]

X, y, n_trials,probs = generate_synthetic_binomial_data(n_samples, true_beta);
X = [collect(row) for row in eachrow(X)];


@model function binomial_model(prior_xi, prior_precision, n_trials, X, y) 
    β ~ MvNormalWeightedMeanPrecision(prior_xi, prior_precision)
    for i in eachindex(y)
        y[i] ~ BinomialPolya(X[i], n_trials[i], β) where {
            dependencies = RequireMessageFunctionalDependencies(β = MvNormalWeightedMeanPrecision(prior_xi, prior_precision))
        }
    end
end

n_features = length(true_beta)
results = infer(
    model = binomial_model(prior_xi = zeros(n_features), prior_precision = diageye(n_features),),
    data = (X=X, y=y,n_trials=n_trials),
    iterations = 30,
    free_energy = true,
    showprogress = true,
    options = (
        limit_stack_depth = 100, # to prevent stack-overflow errors
    )
)


plot(results.free_energy,fontfamily = "Computer Modern", label="Free Energy", xlabel="Iteration", ylabel="Free Energy", title="Free Energy Convergence")

m = mean(results.posteriors[:β][end])
Σ = cov(results.posteriors[:β][end])

p = plot(xlims=(-3.1,-2.9), ylims=(2.5,2.7),fontfamily = "Computer Modern")
covellipse!(m, Σ, n_std=1, aspect_ratio=1, label="1σ Contours", color=:green, fillalpha=0.2)
covellipse!(m, Σ, n_std=3, aspect_ratio=1, label="3σ Contours", color=:blue, fillalpha=0.2)
scatter!([m[1]], [m[2]], label="Mean estimate", color=:blue)
scatter!([true_beta[1]], [true_beta[2]], label="True Parameters")

y_with_missing = Vector{Union{Missing, Int}}(missing, n_samples)
for i in 1:n_samples
    if i > 8000
        y_with_missing[i] = missing
    else
        y_with_missing[i] = y[i]
    end
end

results_with_missing = infer(
    model = binomial_model(prior_xi = zeros(n_features), prior_precision = diageye(n_features),),
    data = (X=X, y=y_with_missing,n_trials=n_trials),
    iterations = 30,
    showprogress = true,
    options = (
        limit_stack_depth = 100, # to prevent stack-overflow errors
    )
)

probs_prediction = map(d -> d.p,results_with_missing.predictions[:y][end][8000:end])
err = probs_prediction .- probs[8000:end]
mse = mean(err.^2)
println("Mean squared error: ", mse)

function bin_predictions(true_probs, pred_probs; n_bins=20)
    bins = range(0, 1, length=n_bins+1)
    bin_means = Float64[]
    bin_stds = Float64[]
    bin_centers = Float64[]
    
    for i in 1:n_bins
        mask = (true_probs .>= bins[i]) .& (true_probs .< bins[i+1])
        if any(mask)
            push!(bin_means, mean(pred_probs[mask]))
            push!(bin_stds, std(pred_probs[mask]))
            push!(bin_centers, (bins[i] + bins[i+1])/2)
        end
    end
    return bin_centers, bin_means, bin_stds
end

# Create the plot
bin_centers, bin_means, bin_stds = bin_predictions(probs[8000:end], probs_prediction)

p = plot(
    xlabel = "True Probability",
    ylabel = "Predicted Probability",
    title = "Prediction Performance",
    aspect_ratio = 1,
    legend = :bottomright,
    grid = true,
    fontfamily = "Computer Modern",
    dpi = 300
)

# Add perfect prediction line
plot!([0, 1], [0, 1], 
    label = "Perfect Prediction", 
    color = :black, 
    linestyle = :dash,
    linewidth = 2
)

# Add scatter plot with reduced opacity and size
scatter!(
    probs[8000:end], 
    probs_prediction,
    label = "Individual Predictions",
    alpha = 0.1,  # Reduced opacity
    color = :blue,
    markersize = 1,
    markerstrokewidth = 0
)

# Add binned means with error bars
scatter!(
    bin_centers,
    bin_means,
    yerror = bin_stds,
    label = "Binned Mean ± SD",
    color = :red,
    markersize = 4
)

annotate!(
    0.05, 
    0.95, 
    text("MSE = $(round(mse, digits=8))", 8, :left, :top)
)

# Customize axes
plot!(
    xlims = (0,1),
    ylims = (0,1),
    xticks = 0:0.2:1,
    yticks = 0:0.2:1
)