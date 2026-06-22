# TIER 3 — EXPERIMENTAL: Lanczos and Krylov spread complexity
module Krylov

using LinearAlgebra

function lanczos_tridiagonal(H::AbstractMatrix, v0::AbstractVector, m::Int)
    n = size(H,1)
    @assert size(H,1) == size(H,2) "H debe ser cuadrada"
    @assert length(v0) == n
    m < 1 && return (zeros(eltype(H), 0, 0), zeros(eltype(H), 0), zeros(eltype(H), 0))
    nrm = norm(v0)
    nrm == 0 && return (zeros(eltype(H), m, m), zeros(eltype(H), m), zeros(eltype(H), m-1))
    v = copy(v0) / nrm
    w = similar(v)
    alpha = zeros(eltype(H), m)
    beta = zeros(eltype(H), m-1)
    V = zeros(eltype(H), n, m)
    V[:,1] = v
    for j in 1:m
        w = H * v
        if j > 1
            w .-= beta[j-1] * V[:,j-1]
        end
        alpha[j] = dot(v, w)
        w .-= alpha[j] * v
        bj = norm(w)
        if j < m
            beta[j] = bj
            if bj == 0
                V[:,j+1:end] .= 0
                break
            end
            v = w / bj
            V[:,j+1] = v
        end
    end
    T = zeros(eltype(H), m, m)
    for j in 1:m
        T[j,j] = alpha[j]
        if j < m
            T[j,j+1] = beta[j]
            T[j+1,j] = beta[j]
        end
    end
    T, alpha, beta
end

"""
    krylov_spread_complexity(T)

Spread complexity of the tridiagonal matrix T.

Computes C = (Σ_{i,j} |T_{i,j}|·|i-j|) / (Σ |T_{i,j}| · (n-1))  normalized to [0,1].

This measures how far T is from diagonal — higher values indicate more
off-diagonal spreading (operator growth in Krylov space).  Renamed from
krylov_offdiag_complexity for consistency with standard terminology.
Returns 0 if T is empty or zero.
"""
function krylov_spread_complexity(T::AbstractMatrix{Tv}) where Tv
    n = size(T, 1)
    n < 2 && return zero(Tv)
    num = zero(Tv)
    den = zero(Tv)
    for i in 1:n
        for j in 1:n
            if i != j
                w = abs(T[i,j])
                num += w * abs(i - j)
                den += w * (n - 1)
            end
        end
    end
    return den > 0 ? num / den : zero(Tv)
end

# Old name kept as alias for backward compatibility
const krylov_offdiag_complexity = krylov_spread_complexity

export lanczos_tridiagonal, krylov_spread_complexity, krylov_offdiag_complexity

# --- Simple predictor for next Lanczos coefficients (linear extrapolation) ---
function predict_next_coeffs(alpha::AbstractVector{T}, beta::AbstractVector{T}) where {T<:Real}
    isempty(alpha) && return (zero(T), zero(T))
    isempty(beta) && return (alpha[end], zero(T))
    return alpha[end], beta[end]
end

export predict_next_coeffs

end
