# TIER 3 — EXPERIMENTAL: Quiver representation theory and layers
module Algebra

using LinearAlgebra

struct Quiver
    nodes::Vector{Int}
    edges::Vector{Tuple{Int,Int}}
end

function adjacency_matrix(q::Quiver)
    n = length(q.nodes)
    idx = Dict(q.nodes[i] => i for i in 1:n)
    A = zeros(Float64, n, n)
    for (u,v) in q.edges
        A[idx[v], idx[u]] += 1.0
    end
    A
end

function normalized_adjacency(q::Quiver)
    A = adjacency_matrix(q)
    d = vec(sum(A, dims=1))
    Dinv = zeros(size(A))
    for i in 1:length(d)
        Dinv[i,i] = d[i] > 0 ? 1.0/d[i] : 0.0
    end
    A*Dinv
end

function power_spectral_radius(A::AbstractMatrix; iters::Int=200)
    n = size(A,1)
    v = ones(n)
    v ./= norm(v)
    λ = 0.0
    for _ in 1:iters
        w = A*v
        λ = norm(w)
        if λ == 0.0
            return 0.0
        end
        v = w/λ
    end
    λ
end

function is_spectrally_stable(A::AbstractMatrix; tol=1e-6)
    ρ = power_spectral_radius(A)
    ρ < 1.0 + tol
end

# --- Near-ring over linear maps (non-commutative multiplication via composition) ---
near_ring_add(A::AbstractMatrix, B::AbstractMatrix) = A + B
near_ring_mul(A::AbstractMatrix, B::AbstractMatrix) = A * B

relu(x) = max(0.0, x)

struct QuiverLayer
    q::Quiver
    in_dim::Int
    out_dim::Int
    edge_maps::Dict{Tuple{Int,Int}, Matrix{Float64}}
end

function QuiverLayer(q::Quiver, in_dim::Int, out_dim::Int; edge_maps::Dict{Tuple{Int,Int}, Matrix{Float64}}=Dict{Tuple{Int,Int}, Matrix{Float64}}())
    # default identity maps if dims align and not provided
    em = Dict{Tuple{Int,Int}, Matrix{Float64}}()
    for e in q.edges
        if haskey(edge_maps, e)
            em[e] = edge_maps[e]
        else
            if in_dim == out_dim
                em[e] = Matrix{Float64}(I, out_dim, in_dim)
            else
                em[e] = zeros(out_dim, in_dim)
            end
        end
    end
    QuiverLayer(q, in_dim, out_dim, em)
end

function apply_quiver(layer::QuiverLayer, X::AbstractMatrix)
    # X: n_nodes x in_dim, returns n_nodes x out_dim
    q = layer.q
    n = length(q.nodes)
    @assert size(X,1) == n && size(X,2) == layer.in_dim
    idx = Dict(q.nodes[i] => i for i in 1:n)
    Y = zeros(Float64, n, layer.out_dim)
    for (u,v) in q.edges
        U = idx[u]; V = idx[v]
        W = layer.edge_maps[(u,v)]
        contrib = W * (X[U, :])
        Y[V, :] .+= contrib
    end
    Y
end

function apply_quiver_walks(layer::QuiverLayer, X::AbstractMatrix; depth::Int=1)
    # Aggregate contributions along paths up to given depth via near_ring_mul of edge maps
    depth <= 1 && return apply_quiver(layer, X)
    q = layer.q
    n = length(q.nodes)
    idx = Dict(q.nodes[i] => i for i in 1:n)
    # precompute adjacency of maps per pair
    maps = Dict{Tuple{Int,Int}, Matrix{Float64}}()
    for (u,v) in q.edges
        maps[(u,v)] = layer.edge_maps[(u,v)]
    end
    # dynamic programming for path maps
    accum = apply_quiver(layer, X)
    current = accum
    for _ in 2:depth
        Y = zeros(Float64, n, layer.out_dim)
        for (a,b) in q.edges
            for (u,v) in q.edges
                if b == u
                    W = near_ring_mul(maps[(u,v)], maps[(a,b)])
                    A = idx[a]; V = idx[v]
                    contrib = W * (current[A, :])
                    Y[V, :] .+= contrib
                end
            end
        end
        accum .+= Y
        current = Y
    end
    accum
end

function mv_activation(Y::AbstractMatrix, acts::Vector{Function})
    # returns n x (k*out_dim)
    n, d = size(Y)
    outs = Vector{Matrix{Float64}}()
    for f in acts
        push!(outs, map(f, Y))
    end
    hcat(outs...)
end

export Quiver, adjacency_matrix, normalized_adjacency, power_spectral_radius, is_spectrally_stable,
       near_ring_add, near_ring_mul, QuiverLayer, apply_quiver, apply_quiver_walks, mv_activation, relu

end
