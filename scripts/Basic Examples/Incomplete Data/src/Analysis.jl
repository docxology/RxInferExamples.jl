module Analysis

using Plots
using Distributions
using LinearAlgebra
using Statistics

export plot_posterior_distributions, generate_summary_report

"""
    plot_posterior_distributions(result, real_mean, real_precision, max_dim=3)

Plot posterior distributions against true values.
"""
function plot_posterior_distributions(result, real_mean, real_precision, max_dim=3)
    # Get final posteriors
    final_m_posterior = result.posteriors[:m][end]
    final_Λ_posterior = result.posteriors[:Λ][end]
    
    max_dim = min(max_dim, length(real_mean))
    
    # Plot mean posterior for first few dimensions
    p1 = plot(title="Posterior Distribution of Mean (first $max_dim dimensions)", 
              xlabel="Value", ylabel="Density")
    
    for i in 1:max_dim
        # Extract marginal distribution for dimension i
        marginal_mean = mean(final_m_posterior)[i]
        marginal_var = inv(mean(final_Λ_posterior))[i,i]
        
        # Plot the Gaussian
        sigma = sqrt(marginal_var)
        x_range = range(marginal_mean - 4*sigma, marginal_mean + 4*sigma, length=200)
        
        gaussian = Normal(marginal_mean, sigma)
        plot!(p1, x_range, pdf.(gaussian, x_range), 
              label="Dim $i Est", linewidth=2, color=i)
        
        # Add vertical line for true value with same color
        vline!(p1, [real_mean[i]], color=i, linestyle=:dash, alpha=0.7, 
               linewidth=2, label="Dim $i True")
    end
    
    return p1
end

end # module
