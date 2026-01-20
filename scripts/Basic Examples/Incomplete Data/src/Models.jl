module Models

using RxInfer
using LinearAlgebra
using Distributions

export incomplete_data, create_constraints, create_initialization

@model function incomplete_data(y, dim, huge)
    Λ ~ Wishart(dim, diagm(ones(dim)))
    m ~ MvNormal(mean=zeros(dim), precision=diagm(ones(dim)))
    for i in 1:size(y, 1)
        x[i] ~ MvNormal(mean=m, precision=Λ)
        for j in 1:dim
            # Create basis vector manually to avoid dependency issues
            # We need to ensure basis is treated as constant or use a different formulation
            # But creating a constant vector inside model loop is fine
            # y[i, j] ~ Normal(mean=dot(x[i], basis), precision=huge)
            # However, creating basis vector inside loop might be inefficient or unsupported if not tracked?
            # Better: use indexing if supported? x[i][j]?
            # RxInfer allows x[i][j] usually.
            # Let's try direct indexing:
            # y[i, j] ~ Normal(mean=x[i][j], precision=huge)
            # But x[i] is MvNormal. Accessing components of MvNormal variable?
            # Standard way is dot product.
            # Let's use dot.
            
            y[i, j] ~ Normal(mean=dot(x[i], basis_vector(j, dim)), precision=huge)
        end
    end
end

function basis_vector(j, dim)
    v = zeros(dim)
    v[j] = 1.0
    return v
end

function create_constraints()
    return @constraints begin
        q(x, m, Λ) = q(x, m)q(Λ) 
    end
end

function create_initialization(dimension)
    return @initialization begin
        q(Λ) = Wishart(dimension, diagm(ones(dimension)))
    end
end

end # module
