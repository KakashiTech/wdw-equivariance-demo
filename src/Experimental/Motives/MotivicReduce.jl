# TIER 3 — EXPERIMENTAL: Motivic dimension reduction via SVD
module MotivicReduce

using LinearAlgebra
using ..Motives: motivic_features

# Build a feature matrix from a list of (A,b) systems
function motivic_feature_matrix(eqs::AbstractVector, primes::AbstractVector{<:Integer})
    F = zeros(Float64, length(eqs), length(primes))
    for (i, pair) in enumerate(eqs)
        A, b = pair
        F[i, :] = motivic_features(A, b, collect(primes))
    end
    F
end

# PCA-like linear projection using SVD; returns embeddings Z and projection P (right singular vecs)
function motivic_dimreduce(F::AbstractMatrix{<:Real}, k::Int)
    @assert 1 <= k <= min(size(F)...)
    sv = svd(Matrix{Float64}(F))
    P = sv.V[:, 1:k]
    Z = Matrix{Float64}(F) * P
    Z, P
end

export motivic_feature_matrix, motivic_dimreduce

end
