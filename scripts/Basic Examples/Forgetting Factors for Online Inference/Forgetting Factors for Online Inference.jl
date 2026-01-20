# This file was automatically generated from /Users/4d/Documents/GitHub/RxInferExamples.jl/examples/Basic Examples/Forgetting Factors for Online Inference/Forgetting Factors for Online Inference.ipynb
# by NotebookConversion module at 2026-01-19T04:41:26.195
# Do not edit by hand. Edit the notebook instead.
#
# Source notebook: Forgetting Factors for Online Inference.ipynb

using RxInfer, StableRNGs, Plots

rng = StableRNG(1234)
time_points = 0.0:0.1:300.0
unstationary_noise = 100 .+ 80 .* sin.(0.075 * time_points)
data_points = map(d -> rand(rng, NormalMeanPrecision(0.0, d[2])), zip(time_points, unstationary_noise))

p1 = plot(time_points, data_points, seriestype = :line, title = "Signal", xlabel = "Time", ylabel = "Observation")
p2 = plot(time_points, unstationary_noise, seriestype = :line, title = "True Precision of Observations", xlabel = "Time", ylabel = "Std")
plot(p1, p2, layout = (2, 1))

@model function kalman_filter(observation, state_prior_mean, state_prior_var, obs_noise_shape, obs_noise_rate)
    state_prior ~ Normal(mean = state_prior_mean, var = state_prior_var)
    next_state ~ Normal(mean = state_prior, precision = 1000.0)
    noise_precision ~ Gamma(shape = obs_noise_shape, rate = obs_noise_rate)
    observation ~ Normal(mean = next_state, precision = noise_precision)
end

autoupdates_without_forgetting = @autoupdates begin
    # Update the state prior mean and variance
    state_prior_mean = mean(q(next_state))
    state_prior_var = var(q(next_state))
    # Update the noise precision
    obs_noise_shape = shape(q(noise_precision))
    obs_noise_rate = rate(q(noise_precision))
end

initialization = @initialization begin
    q(next_state) = NormalMeanPrecision(0.0, 1000.0)
    q(noise_precision) = GammaShapeRate(1.0, 0.005)
end

result_without_forgetting = infer(
    model = kalman_filter(),
    data = (observation = data_points,),
    autoupdates = autoupdates_without_forgetting,
    initialization = initialization,
    autostart = true,
    constraints = MeanField(),
    historyvars = (noise_precision = KeepLast(), ),
    iterations = 50,
    keephistory = length(data_points)
)

inferred_without_forgetting = result_without_forgetting.history[:noise_precision]

plot(time_points, mean.(inferred_without_forgetting), ribbon = 3std.(inferred_without_forgetting), seriestype = :line, title = "Inferred Noise Precision", xlabel = "Time", ylabel = "Std", label = "Inferred (+/- 3 std)")
plot!(time_points, unstationary_noise, seriestype = :line, title = "True Precision of Observations", xlabel = "Time", ylabel = "Std", label = "True")

mutable struct UpdateParamsWithForgetting
    forgetting_factor::Float64
    previous_params::Union{Nothing, Tuple{Float64, Float64}}

    function UpdateParamsWithForgetting(; Γ = 0.99)
        return new(Γ, nothing)
    end
end

# Create a callable structure
# https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects
# it allows us to have a local state, in this case `previous_params`
function (update::UpdateParamsWithForgetting)(gamma_posterior)

    if isnothing(update.previous_params)
        update.previous_params = (shape(gamma_posterior), rate(gamma_posterior))
        return (shape(gamma_posterior), rate(gamma_posterior))
    else 
        shape_prev, rate_prev = update.previous_params
        shape_current, rate_current = (shape(gamma_posterior), rate(gamma_posterior))
        shape_delta = shape_current - shape_prev
        rate_delta = rate_current - rate_prev

        @assert (shape_delta > 0) && (rate_delta > 0) "Shape and rate deltas must be strictly positive"
        
        Γ = update.forgetting_factor

        f_shape = shape_prev * Γ + shape_delta
        f_rate = rate_prev * Γ + rate_delta

        update.previous_params = (f_shape, f_rate)
        return (f_shape, f_rate)
    end

end

# we make it a function because it must create `obs_update` every time it is called
@autoupdates function autoupdates_with_forgetting(; forgetting_factor)
    # Update the state prior mean and variance
    state_prior_mean = mean(q(next_state))
    state_prior_var = var(q(next_state))

    # Update the forgetting factor, callable structure
    obs_update = UpdateParamsWithForgetting(Γ = forgetting_factor)

    # Update the noise precision, two at once
    obs_noise_shape, obs_noise_rate = obs_update(q(noise_precision))
end

result_with_forgetting = infer(
    model = kalman_filter(),
    data = (observation = data_points,),
    autoupdates = autoupdates_with_forgetting(forgetting_factor = 0.97),
    initialization = initialization,
    autostart = true,
    constraints = MeanField(),
    historyvars = (noise_precision = KeepLast(), ),
    iterations = 100,
    keephistory = length(data_points)
)

inferred_with_forgetting = result_with_forgetting.history[:noise_precision]

plot(time_points, mean.(inferred_with_forgetting), ribbon = 3std.(inferred_with_forgetting), seriestype = :line, title = "Inferred Noise Precision", xlabel = "Time", ylabel = "Std", label = "Inferred (+/- 3 std)")
plot!(time_points, unstationary_noise, seriestype = :line, title = "True Precision of Observations", xlabel = "Time", ylabel = "Precision", label = "True")

result_with_forgetting = infer(
    model = kalman_filter(),
    data = (observation = data_points,),
    autoupdates = autoupdates_with_forgetting(forgetting_factor = 0.99),
    initialization = initialization,
    autostart = true,
    constraints = MeanField(),
    historyvars = (noise_precision = KeepLast(), ),
    iterations = 100,
    keephistory = length(data_points)
)

inferred_with_forgetting = result_with_forgetting.history[:noise_precision]

plot(time_points, mean.(inferred_with_forgetting), ribbon = 3std.(inferred_with_forgetting), seriestype = :line, title = "Inferred Noise Precision", xlabel = "Time", ylabel = "Std", label = "Inferred (+/- 3 std)")
plot!(time_points, unstationary_noise, seriestype = :line, title = "True Precision of Observations", xlabel = "Time", ylabel = "Precision", label = "True")