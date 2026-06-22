# TIER 2 — RESEARCH: Group-equivariant neural network framework
module Quantum

using LinearAlgebra

struct CyclicGroup
    n::Int
end

function act(g::CyclicGroup, k::Int, x::AbstractVector)
    n = g.n
    y = similar(x)
    for i in 1:n
        j = 1 + mod(i - 1 - k, n)
        y[i] = x[j]
    end
    y
end

function project_equivariant(M::AbstractMatrix, g::CyclicGroup)
    n = g.n
    acc = zeros(eltype(M), n, n)
    for k in 0:n-1
        for i in 1:n, j in 1:n
            pi = 1 + mod(i - 1 - k, n)
            pj = 1 + mod(j - 1 - k, n)
            acc[i,j] += M[pi, pj]
        end
    end
    acc ./= n
    acc
end

function is_equivariant(W::AbstractMatrix, g::CyclicGroup; trials::Int=3)
    n = g.n
    for k in 0:n-1
        for t in 1:min(trials, n)
            x = zeros(eltype(W), n)
            x[t] = one(eltype(W))
            lhs = act(g,k, W * x)
            rhs = W * act(g,k,x)
            if norm(lhs - rhs) > 1e-6
                return false
            end
        end
    end
    true
end

export CyclicGroup, act, project_equivariant, is_equivariant

struct FinitePermGroup
    n::Int
    perms::Vector{Vector{Int}}
end

function act(G::FinitePermGroup, p::Vector{Int}, x::AbstractVector)
    n = G.n
    y = similar(x)
    for i in 1:n
        y[i] = x[p[i]]
    end
    y
end

function project_equivariant(M::AbstractMatrix, G::FinitePermGroup)
    n = G.n
    acc = zeros(eltype(M), n, n)
    for p in G.perms
        for i in 1:n, j in 1:n
            acc[i,j] += M[p[i], p[j]]
        end
    end
    acc ./= length(G.perms)
    acc
end

export FinitePermGroup

function dihedral_group(n::Int)
    perms = Vector{Vector{Int}}()
    for k in 0:n-1
        p = [1 + mod(i - 1 + k, n) for i in 1:n]
        push!(perms, p)
    end
    r = [n - i + 1 for i in 1:n]
    for k in 0:n-1
        rot = [1 + mod(i - 1 + k, n) for i in 1:n]
        p = [r[rot[i]] for i in 1:n]
        push!(perms, p)
    end
    FinitePermGroup(n, perms)
end

export dihedral_group

# Build a permutation group from generators by closure (BFS) with a safety cap
function _invperm(p::Vector{Int})
    n = length(p)
    q = similar(p)
    for i in 1:n
        q[p[i]] = i
    end
    q
end

function _compose_perm(p::Vector{Int}, q::Vector{Int})
    # p ∘ q
    n = length(p)
    r = Vector{Int}(undef, n)
    for j in 1:n
        r[j] = p[q[j]]
    end
    r
end

function _word_to_perm(gens::Vector{Vector{Int}}, invgens::Vector{Vector{Int}}, word::Vector{Int})
    n = length(gens[1])
    cur = collect(1:n)
    for w in word
        if w > 0
            cur = _compose_perm(gens[w], cur)
        else
            cur = _compose_perm(invgens[-w], cur)
        end
    end
    cur
end

function verify_relations(n::Int, generators::Vector{Vector{Int}}, relations::Vector{Vector{Int}})
    invgens = [ _invperm(g) for g in generators ]
    idp = collect(1:n)
    for rel in relations
        p = _word_to_perm(generators, invgens, rel)
        if p != idp
            return false
        end
    end
    true
end

export verify_relations

function presented_perm_group(n::Int, generators::Vector{Vector{Int}}; relations::Vector{Vector{Int}}=Vector{Vector{Int}}(), max_size::Int=100000)
    if !isempty(relations)
        @assert verify_relations(n, generators, relations) "Generators do not satisfy provided relations"
    end
    # normalize generators length
    gens = [copy(g) for g in generators]
    invgens = [ _invperm(g) for g in gens ]
    seen = Dict{String,Int}()
    perms = Vector{Vector{Int}}()
    add!(p) = (key = join(p, ","); haskey(seen, key) ? false : (push!(perms, p); seen[key] = length(perms); true))
    # identity
    idp = collect(1:n)
    add!(idp)
    i = 1
    while i <= length(perms) && length(perms) < max_size
        base = perms[i]
        for (idx, g) in enumerate(gens)
            # compose base ∘ g
            p = [ base[g[j]] for j in 1:n ]
            add!(p)
            # compose g ∘ base
            q = [ g[base[j]] for j in 1:n ]
            add!(q)
            # also compose with inverse generators
            ginv = invgens[idx]
            p2 = [ base[ginv[j]] for j in 1:n ]
            add!(p2)
            q2 = [ ginv[base[j]] for j in 1:n ]
            add!(q2)
        end
        i += 1
    end
    FinitePermGroup(n, perms)
end

export presented_perm_group

struct CachedPermGroup{T}
    n::Int
    perms::Vector{Vector{Int}}
    P::Vector{Matrix{T}}
    Pinv::Vector{Matrix{T}}
end

function cache_permutation_matrices(G::FinitePermGroup; T::Type{<:Number}=Float64)
    n = G.n
    P = Matrix{T}[]
    Pinv = Matrix{T}[]
    for p in G.perms
        mP = zeros(T, n, n)
        mPinv = zeros(T, n, n)
        invp = similar(p)
        for i in 1:n
            invp[p[i]] = i
        end
        for i in 1:n
            mP[i,p[i]] = one(T)
            mPinv[i,invp[i]] = one(T)
        end
        push!(P, mP)
        push!(Pinv, mPinv)
    end
    CachedPermGroup{T}(n, G.perms, P, Pinv)
end

export CachedPermGroup, cache_permutation_matrices

function act(G::CachedPermGroup, k::Int, x::AbstractVector)
    G.P[k] * x
end

function project_equivariant(M::AbstractMatrix, G::CachedPermGroup)
    n = G.n
    acc = zeros(eltype(M), n, n)
    # Use perms directly for O(n^2 * |G|) index-based accumulation
    for p in G.perms
        for i in 1:n, j in 1:n
            acc[i,j] += M[p[i], p[j]]
        end
    end
    acc ./= length(G.perms)
    acc
end

struct LinearGroup{T}
    n::Int
    ops::Vector{AbstractMatrix{T}}
    invops::Vector{AbstractMatrix{T}}
end

function act(G::LinearGroup, k::Int, x::AbstractVector)
    G.ops[k] * x
end

function project_equivariant(M::AbstractMatrix, G::LinearGroup)
    n = G.n
    acc = zeros(eltype(M), n, n)
    for i in 1:length(G.ops)
        acc .+= G.ops[i] * M * G.invops[i]
    end
    acc ./= length(G.ops)
    acc
end

function is_equivariant(W::AbstractMatrix, G::FinitePermGroup; trials::Int=0, tol::Real=1e-6)
    n = G.n
    for p in G.perms
        for t in 1:n
            x = zeros(eltype(W), n)
            x[t] = one(eltype(W))
            lhs = act(G, p, W * x)
            rhs = W * act(G, p, x)
            if norm(lhs - rhs) > tol
                return false
            end
        end
    end
    true
end

function is_equivariant(W::AbstractMatrix, G::CachedPermGroup; trials::Int=0, tol::Real=1e-6)
    n = G.n
    for i in 1:length(G.P)
        for t in 1:n
            x = zeros(eltype(W), n)
            x[t] = one(eltype(W))
            lhs = G.P[i] * (W * x)
            rhs = W * (G.P[i] * x)
            if norm(lhs - rhs) > tol
                return false
            end
        end
    end
    true
end

function is_equivariant(W::AbstractMatrix, G::LinearGroup; trials::Int=0, tol::Real=1e-4)
    n = G.n
    for i in 1:length(G.ops)
        for t in 1:n
            x = zeros(eltype(W), n)
            x[t] = one(eltype(W))
            lhs = G.ops[i] * (W * x)
            rhs = W * (G.ops[i] * x)
            if norm(lhs - rhs) > tol
                return false
            end
        end
    end
    true
end

function _frac_shift_matrix(n::Int, shift::Float64)
    A = zeros(Float64, n, n)
    for i in 1:n
        s = i - shift
        j0 = floor(Int, s)
        w = s - float(j0)
        # wrap indices to 1..n using mod (nonnegative)
        j0mod = 1 + mod(j0 - 1, n)
        j1mod = 1 + mod(j0, n)
        A[i, j0mod] += 1 - w
        A[i, j1mod] += w
    end
    A
end

function SO2_linear_group(n::Int, K::Int)
    ops = Matrix{Float64}[]
    invops = Matrix{Float64}[]
    for k in 0:K-1
        θ = 2*pi * (k / K)
        shift = n * θ / (2*pi)
        A = _frac_shift_matrix(n, shift)
        Ainv = _frac_shift_matrix(n, -shift)
        push!(ops, A)
        push!(invops, Ainv)
    end
    LinearGroup{Float64}(n, ops, invops)
end

export LinearGroup, SO2_linear_group

function _Rz(θ)
    c = cos(θ); s = sin(θ)
    [c -s 0.0;
     s  c 0.0;
     0.0 0.0 1.0]
end

function _Ry(θ)
    c = cos(θ); s = sin(θ)
    [ c 0.0  s;
      0.0 1.0 0.0;
     -s 0.0  c]
end

function SO3_linear_group(Kα::Int, Kβ::Int, Kγ::Int)
    ops = Matrix{Float64}[]
    invops = Matrix{Float64}[]
    for ia in 0:Kα-1
        α = 2*pi * (ia / Kα)
        for ib in 0:Kβ-1
            β = pi * (ib / (Kβ==1 ? 1 : (Kβ-1)))
            for ig in 0:Kγ-1
                γ = 2*pi * (ig / Kγ)
                R = _Rz(α) * _Ry(β) * _Rz(γ)
                push!(ops, R)
                push!(invops, transpose(R))
            end
        end
    end
    LinearGroup{Float64}(3, ops, invops)
end

export SO3_linear_group

# --- Word reduction utilities for presented groups ---
inv_word(w::Vector{Int}) = [-x for x in reverse(w)]

function free_reduce(w::Vector{Int})
    out = Int[]
    for x in w
        if !isempty(out) && out[end] == -x
            pop!(out)
        else
            push!(out, x)
        end
    end
    out
end

function _remove_once!(w::Vector{Int}, pat::Vector{Int})
    m = length(pat)
    if m == 0 || length(w) < m
        return false
    end
    i = 1
    while i <= length(w) - m + 1
        ok = true
        for k in 1:m
            if w[i+k-1] != pat[k]
                ok = false
                break
            end
        end
        if ok
            deleteat!(w, i:i+m-1)
            return true
        end
        i += 1
    end
    false
end

function reduce_word(w::Vector{Int}, relations::Vector{Vector{Int}})
    # include inverse relations too
    rels = vcat(relations, [inv_word(r) for r in relations])
    changed = true
    cur = free_reduce(w)
    while changed
        changed = false
        # try free cancellations repeatedly via pattern length 2
        cur = free_reduce(cur)
        # remove any relation occurrences
        for r in rels
            if _remove_once!(cur, r)
                changed = true
                break
            end
        end
    end
    cur
end

export reduce_word

function presented_perm_group_reduced(n::Int, generators::Vector{Vector{Int}}; relations::Vector{Vector{Int}}=Vector{Vector{Int}}(), max_size::Int=100000, max_word_len::Int=1000)
    if !isempty(relations)
        @assert verify_relations(n, generators, relations) "Generators do not satisfy provided relations"
    end
    gens = [copy(g) for g in generators]
    invgens = [ _invperm(g) for g in gens ]
    # BFS on reduced words ±1..±m
    seen = Dict{String,Int}()
    perms = Vector{Vector{Int}}()
    words = Vector{Vector{Int}}()
    add_perm!(p) = (key = join(p, ","); haskey(seen, key) ? false : (push!(perms, p); seen[key] = length(perms); true))
    # identity
    idp = collect(1:n)
    add_perm!(idp)
    push!(words, Int[])
    i = 1
    while i <= length(words) && length(perms) < max_size
        w = words[i]
        for j in 1:length(gens)
            for sgn in (+1, -1)
                wnew = copy(w)
                push!(wnew, sgn>0 ? j : -j)
                if !isempty(relations)
                    wnew = reduce_word(wnew, relations)
                else
                    wnew = free_reduce(wnew)
                end
                if length(wnew) > max_word_len
                    continue
                end
                p = _word_to_perm(gens, invgens, wnew)
                if add_perm!(p)
                    push!(words, wnew)
                end
            end
        end
        i += 1
    end
    FinitePermGroup(n, perms)
end

export presented_perm_group_reduced

end
