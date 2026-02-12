module RxVisualization

using Plots
using RxInfer
using Statistics

export plot_hidden_states, plot_free_energy

"""
    plot_hidden_states(posteriors::Vector{<:Categorical}; title="Hidden States", labels=nothing)

Plots a heatmap of latent state probabilities over time.
Input `posteriors` should be a vector of Categorical distributions (one per time step).
"""
function plot_hidden_states(posteriors; title="Hidden States", labels=nothing)
    # Extract probability vectors: [p1, p2, ..., pT]
    probs = ReactiveMP.probvec.(posteriors)
    
    # Convert to matrix: States x Time
    # hcat splats the vector of vectors into a matrix
    prob_matrix = hcat(probs...)
    
    n_states, n_time = size(prob_matrix)
    
    yticks_val = (1:n_states, labels === nothing ? ["State $i" for i in 1:n_states] : labels)
    
    heatmap(
        prob_matrix, 
        xlabel="Time Step", 
        ylabel="State", 
        title=title,
        color=:blues,
        yticks=yticks_val,
        clims=(0, 1)
    )
end

"""
    plot_free_energy(results; margin=5)

Plots the Bethe Free Energy from an inference result.
`margin` determines how many initial iterations to skip (default 5) to remove initialization artifacts.
"""
function plot_free_energy(results; margin=5)
    fe = results.free_energy
    
    if length(fe) <= margin
        margin = 0
    end
    
    plot(
        margin+1:length(fe), 
        fe[margin+1:end],
        xlabel="Iteration",
        ylabel="Bethe Free Energy [nats]",
        title="Convergence",
        legend=false,
        linewidth=2,
        marker=:circle
    )
end

end # module
