module RxUtils

using LinearAlgebra

export index_to_one_hot, one_hot_to_index
export create_transition_matrix, construct_diag_collection

"""
    index_to_one_hot(index::Int, size::Int)

Creates a one-hot encoded vector of length `size` with 1.0 at `index` and 0.0 elsewhere.
"""
function index_to_one_hot(index::Int, size::Int)
    v = zeros(Float64, size)
    v[index] = 1.0
    return v
end

"""
    one_hot_to_index(v::Vector)

Returns the index of the largest element in a vector (argmax), useful for decoding one-hot or probability vectors.
"""
function one_hot_to_index(v::Vector)
    return argmax(v)
end

"""
    create_transition_matrix(n_states::Int, transition_probs::Dict{Int, Vector{Float64}})

Creates a transition matrix where rows sum to 1. `transition_probs` maps state indices to their transition probability vectors.
Missing states default to uniform probabilities or self-loops (configurable).
"""
function create_transition_matrix(n_states::Int, transition_probs::Dict{Int, Vector{Float64}}; default=:self_loop)
    A = zeros(Float64, n_states, n_states)
    
    for i in 1:n_states
        if haskey(transition_probs, i)
            # Use provided probabilities
            row = transition_probs[i]
            @assert length(row) == n_states "Transition probability for state $i must have length $n_states"
            @assert sum(row) ≈ 1.0 "Transition probabilities for state $i must sum to 1"
            A[:, i] = row
        else
            # Default behavior
            if default == :self_loop
                A[i, i] = 1.0
            elseif default == :uniform
                A[:, i] .= 1.0 / n_states
            else
                error("Unknown default behavior: $default")
            end
        end
    end
    
    return A
end

"""
    construct_diag_collection(size::Int, value::Float64=1.0)

Constructs a vector of vectors for initializing `DirichletCollection` with a strong diagonal preference (sticky).
Each column of the resulting matrix ( conceptually) will have `value` + 10.0 on the diagonal and `value` elsewhere.
"""
function construct_diag_collection(size::Int, value::Float64=1.0)
    # This matches the pattern B ~ DirichletCollection([ 10.0 1.0; 1.0 10.0 ])
    # Input to DirichletCollection is often a Matrix where each column is a Dirichlet parameter
    
    mat = fill(value, size, size)
    for i in 1:size
        mat[i, i] += 9.0 * value # Make the diagonal 10x stronger
    end
    return mat
end

end # module
