# This file was automatically generated from /Users/4d/Documents/GitHub/RxInferExamples.jl/examples/Problem Specific/Ising Model/Ising Model.ipynb
# by NotebookConversion module at 2026-01-19T04:41:26.198
# Do not edit by hand. Edit the notebook instead.
#
# Source notebook: Ising Model.ipynb

using Distributions, ExponentialFamilyProjection, Images, Plots, ReactiveMP, RxInfer, StableRNGs, StatsFuns

struct Sigmoid end

@node Sigmoid Stochastic [out, x]

@rule Sigmoid(:x, Marginalisation) (q_out::PointMass,) = begin
    y = mean(q_out)
    y = float(mean(q_out))
    sign = 1-2y
    # Provide logpdf, gradient, and Hessian for 1D logistic-Bernoulli
    _logpdf = (out, x) -> (out[] = -softplus(sign * x))
    _grad = (out, x) -> (out[1] = y - logistic(x))
    _hess = (out, x) -> (out[1, 1] = -logistic(x) * (1 - logistic(x)))
    return ExponentialFamilyProjection.InplaceLogpdfGradHess(_logpdf, _grad, _hess)
end

function BayesBase.prod(::GenericProd, left::UnivariateGaussianDistributionsFamily, right::ExponentialFamilyProjection.InplaceLogpdfGradHess)
    m = mean(left)
    σ = var(left)
    combined_logpdf! = (out, x) -> begin
        right.logpdf!(out, x)
        out[] = logpdf(left, x) + out[]
    end
    combined_gradhes! = (out_grad, out_hess, x) -> begin
        out_grad, out_hess = right.grad_hess!(out_grad, out_hess, x)
        out_grad .= out_grad .- ((x .- m) ./ σ)
        out_hess .= out_hess .- 1 / σ
        return out_grad, out_hess
    end
    return ExponentialFamilyProjection.InplaceLogpdfGradHess(combined_logpdf!, combined_gradhes!)
end

function BayesBase.prod(::GenericProd, left::ExponentialFamilyProjection.InplaceLogpdfGradHess, right::UnivariateGaussianDistributionsFamily)
    return prod(GenericProd(), right, left)
end

rng = StableRNG(112)
mnist_picture = load("mnist_picture.png")
mnist_picture

sample_matrix = convert(Matrix{Float64}, mnist_picture);
normalized_matrix = (sample_matrix .- mean(sample_matrix))/std(sample_matrix)

observation_matrix = begin 
    o = zeros(28, 28)
    for i in 1:28, j in 1:28
        o[i, j] = rand(rng, Bernoulli(logistic(normalized_matrix[i, j])))
    end
    o
end

Gray.(observation_matrix)

@model function sigmoid_ising(h, w, image, connection_force)
    # x_extra_prior ~ NormalMeanVariance(0, 1)
    local x
    # as smaller variance as closer the estimates
    var_used = 1.0/connection_force
    prior ~ NormalMeanVariance(0, var_used)
    for i in 1:h, j in 1:w
        x[i, j] ~ NormalMeanVariance(prior, var_used)
    end 
    for i in 1:h, j in 1:w
        image[i, j] ~ Sigmoid(x[i, j]) 
        if i < h && j < w
           x[i, j] ~ NormalMeanVariance(x[i+1, j], var_used)
           x[i, j] ~ NormalMeanVariance(x[i, j+1], var_used)
        end
        if i < h
            x[i, j] ~ NormalMeanVariance(x[i+1, j], var_used)
        end
        if j < w
            x[i, j] ~ NormalMeanVariance(x[i, j+1], var_used)
        end
    end
end

# Streaming init & autoupdates
sigmoid_init = @initialization begin
    q(x) = NormalMeanVariance(0.0, 1.0)
    q(prior) = NormalMeanVariance(0.5, 1)
end

binary_constraints = @constraints begin
    q(x) :: ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(
        tolerance = 1e-8,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 1), # deterministic
    ))
    q(x, prior) = q(x)q(prior)
    q(x) = MeanField()
    # q(x_prior, x, x_extra, x_extra_prior) = q(x_prior)q(x)q(x_extra, x_extra_prior)
end

result = infer(
    model          = sigmoid_ising(h=28, w=28, connection_force = 1), 
    data           = (image = observation_matrix,),
    returnvars     = KeepEach(),
    # options        = (limit_stack_depth = 100, ),
    iterations     = 5,
    initialization = sigmoid_init,
    constraints    = binary_constraints,
    showprogress   = true,
);

sigmoid_outputs = map(mean, result.posteriors[:x][5]);
normalize_sigmoid_outputs = (sigmoid_outputs .- mean(sigmoid_outputs))/std(sigmoid_outputs)

l = @layout [
    grid(1,3)
]
plot_obj = plot(layout=l)
plot!(plot_obj, Gray.(normalized_matrix), subplot=1, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
plot!(plot_obj, Gray.(observation_matrix), subplot=2, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
plot!(plot_obj, Gray.(normalize_sigmoid_outputs), subplot=3, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)


mask_probability = 0.25
masked_pattern = rand(rng, Bernoulli(mask_probability), size(observation_matrix)...) 

# Apply mask to observations as Union{Missing, Float64}
masked_observation_matrix = Matrix{Union{Missing, Float64}}(undef, size(observation_matrix)...)
@inbounds for j in axes(observation_matrix, 2), i in axes(observation_matrix, 1)
    masked_observation_matrix[i, j] = masked_pattern[i, j] ? missing : Float64(observation_matrix[i, j])
end

@rule Sigmoid(:out, Marginalisation) (q_x::NormalMeanVariance, ) = begin
    return Bernoulli(logistic(mean(q_x)))
end

result_masked = infer(
    model          = sigmoid_ising(h=28, w=28, connection_force=1), 
    data           = (image = masked_observation_matrix,),
    returnvars     = KeepEach(),
    # options        = (limit_stack_depth = 100, ),
    iterations     = 5,
    initialization = sigmoid_init,
    constraints    = binary_constraints,
    showprogress   = true,
);

sigmoid_outputs_masked = map(mean, result_masked.posteriors[:x][5]);
normalize_masked_sigmoid_outputs = (sigmoid_outputs_masked .- mean(sigmoid_outputs_masked))/std(sigmoid_outputs_masked)

yellow = colorant"yellow"# Replace missings with 0 just to build the base gray image in RGB
masked_img = RGB.(Gray.(replace(masked_observation_matrix, missing => 0.0)))
masked_img[masked_pattern] .= yellow


plot_obj_masked = plot(layout=@layout [
    grid(1,5)
])
plot!(plot_obj_masked, Gray.(normalized_matrix), subplot=1, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
plot!(plot_obj_masked, Gray.(observation_matrix), subplot=2, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
plot!(plot_obj_masked, Gray.(normalize_sigmoid_outputs), subplot=3, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
plot!(plot_obj_masked, masked_img, subplot=4, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
plot!(plot_obj_masked, Gray.(normalize_masked_sigmoid_outputs), subplot=5, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)


exponents = range(-10.0, 10.0, length=100)
connection_forces = 10.0 .^ exponents
iterations_anim = 5

function run_reconstruction(image, connection_force; h=28, w=28, iterations=iterations_anim)
    result = infer(
        model          = sigmoid_ising(h=h, w=w, connection_force=connection_force),
        data           = (image = image,),
        returnvars     = KeepEach(),
        iterations     = iterations,
        initialization = sigmoid_init,
        constraints    = binary_constraints,
        showprogress   = false,
    )
    xs = map(mean, result.posteriors[:x][iterations])
    return (xs .- mean(xs)) / std(xs)
end
# helper for animation


anim_noised = @animate for (idx, cf) in enumerate(connection_forces)
    k = exponents[idx]
    recon = run_reconstruction(observation_matrix, cf)
    l = @layout [grid(1,3)]
    plt = plot(layout=l, size=(900, 300), title="c. force = $(round(10^k, sigdigits=2))")
    plot!(plt, Gray.(normalized_matrix), subplot=1, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
    plot!(plt, Gray.(observation_matrix), subplot=2, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
    plot!(plt, Gray.(recon), subplot=3, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
end

gif(anim_noised, "ising_connection_force_noised.gif", fps=20);


anim_masked = @animate for (idx, cf) in enumerate(connection_forces)
    k = exponents[idx]
    recon = run_reconstruction(masked_observation_matrix, cf)
    l = @layout [grid(1,3)]
    plt = plot(layout=l, size=(900, 300), title="c. force = $(round(10^k, sigdigits=2))")
    plot!(plt, Gray.(normalized_matrix), subplot=1, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
    plot!(plt, masked_img, subplot=2, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
    plot!(plt, Gray.(recon), subplot=3, legend=false, framestyle=:none, ticks=nothing, aspect_ratio=:equal)
end

gif(anim_masked, "ising_connection_force_masked.gif", fps=20);
