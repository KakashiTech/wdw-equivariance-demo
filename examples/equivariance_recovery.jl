#!/usr/bin/env julia
# Minimal reproducible demo: Equivariance detection and recovery via projection

using LinearAlgebra
using Random
using Printf
using DelimitedFiles: writedlm

# Build dihedral group D_n acting on n points as permutations
struct Dihedral
    perms::Vector{Vector{Int}}
end

function dihedral_group(n::Int)
    # rotations r^k
    rot(k) = [((i + k - 1) % n) + 1 for i in 1:n]
    # reflection s (reverse)
    refl = reverse(collect(1:n))
    ps = Vector{Vector{Int}}()
    for k in 0:n-1
        push!(ps, rot(k))
        # s r^k
        rk = rot(k)
        push!(ps, [refl[i] for i in rk])
    end
    # unique by value
    Dihedral(unique(ps))
end

# Apply permutation p to vector x: y[i] = x[p[i]]
@inline function act(p::Vector{Int}, x::AbstractVector)
    y = similar(x)
    @inbounds for i in eachindex(x)
        y[i] = x[p[i]]
    end
    y
end

# Permutation matrix from p (P * x == act(p, x))
function perm_matrix(p::Vector{Int}, ::Type{T}=Float64) where {T}
    n = length(p)
    P = zeros(T, n, n)
    @inbounds for i in 1:n
        P[i, p[i]] = one(T)
    end
    P
end

# Project W onto the commutant of the group action: average conjugation
function project_equivariant(W::AbstractMatrix, G::Dihedral)
    n = size(W,1)
    @assert size(W,2) == n
    acc = zeros(eltype(W), n, n)
    for p in G.perms
        P = perm_matrix(p, eltype(W))
        acc .+= transpose(P) * W * P
    end
    acc ./= length(G.perms)
    acc
end

# Equivariance error: average || P*(W*x) - W*(P*x) || over basis vectors and a subset of perms
function eqerr(W::AbstractMatrix, G::Dihedral; max_perms::Union{Int,Nothing}=nothing)
    n = size(W,1)
    m = max_perms === nothing ? min(length(G.perms), n) : min(length(G.perms), max_perms)
    acc = 0.0
    for i in 1:m
        p = G.perms[i]
        for t in 1:n
            x = zeros(n); x[t] = 1.0
            PxW = act(p, W * x)
            WPx = W * act(p, x)
            acc += norm(PxW - WPx)
        end
    end
    acc / (m * n)
end

function main()
    n     = length(ARGS) >= 1 ? parse(Int, ARGS[1])     : 12
    noise = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.10
    seed  = length(ARGS) >= 3 ? parse(Int, ARGS[3])     : 0
    outcsv = length(ARGS) >= 4 ? ARGS[4] : nothing
    outpng = length(ARGS) >= 5 ? ARGS[5] : nothing
    Random.seed!(seed)

    G = dihedral_group(n)

    # 1) Build an exactly equivariant operator by projecting a random matrix
    M = randn(n, n)
    W_eq = project_equivariant(M, G)
    e_eq = eqerr(W_eq, G)

    # 2) Induce a rupture (break equivariance)
    R = randn(n, n)
    W_rupt = W_eq + noise * R
    e_rupt = eqerr(W_rupt, G)

    # 3) Recover via projection
    W_rec = project_equivariant(W_rupt, G)
    e_rec = eqerr(W_rec, G)

    @printf "n = %d, noise = %.3f, seed = %d\n" n noise seed
    @printf "Equivariance error (equivariant base): %.3e\n" e_eq
    @printf "After induced rupture:                %.3e\n" e_rupt
    @printf "After projection (recovery):          %.3e\n" e_rec
    @printf "Recovery factor e_rec/e_rupt:         %.2e\n" e_rec / max(e_rupt, eps())

    ok = (e_eq <= 1e-8) && (e_rec <= 1e-8) && (e_rec <= 1e-2 * e_rupt)
    println(ok ? "STATUS: OK (equivariance recovered)" : "STATUS: CHECK (expected strong reduction)")

    if outcsv !== nothing
        header = ["n","noise","seed","err_base","err_rupt","err_recover","ratio"]
        row = [n, noise, seed, e_eq, e_rupt, e_rec, e_rec / max(e_rupt, eps())]
        tbl = [header; row]
        try
            writedlm(outcsv, tbl, ',')
            @printf "Wrote CSV: %s\n" outcsv
        catch err
            @printf "WARN: could not write CSV to %s (%s)\n" string(outcsv) string(err)
        end
    end

    if outpng !== nothing
        try
            # Load Plots if available
            Base.require(Base.PkgId(Base.UUID("91a5bcdd-55d7-5caf-9e0b-520d859cae80"), "Plots"))
            @eval using Plots
            vals = [e_eq, e_rupt, e_rec]
            p = Plots.bar(["base","rupt","recover"], vals; title = "Equivariance Error", ylabel = "‖⋅‖", legend = false)
            Plots.savefig(p, outpng)
            @printf "Wrote PNG: %s\n" outpng
        catch err
            @printf "WARN: could not write PNG to %s (Plots not available or error: %s)\n" string(outpng) string(err)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
