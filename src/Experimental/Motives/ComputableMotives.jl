# TIER 3 — EXPERIMENTAL: Computable motivic features and Betti numbers
module Motives

struct Graph
    n::Int
    edges::Vector{Tuple{Int,Int}}
end

function betti0(g::Graph)
    g.n < 1 && return 0
    for (u,v) in g.edges
        @assert 1 <= u <= g.n && 1 <= v <= g.n "vertex out of range"
    end
    parent = collect(1:g.n)
    function find(a)
        1 <= a <= g.n || error("find: vertex $a out of range [1,$(g.n)]")
        while parent[a] != a
            parent[a] = parent[parent[a]]
            a = parent[a]
        end
        a
    end
    function unite(a,b)
        ra = find(a); rb = find(b)
        if ra != rb
            parent[rb] = ra
        end
    end
    for (u,v) in g.edges
        unite(u,v)
    end
    length(unique(find(i) for i in 1:g.n))
end

function relabel(g::Graph, π::Vector{Int})
    edges2 = [(π[u], π[v]) for (u,v) in g.edges]
    Graph(g.n, edges2)
end

function _modp(a::Integer, p::Integer)
    p > 0 || error("_modp: modulus must be positive, got $p")
    mod(a, p)
end

function _invmod(a::Integer, p::Integer)
    p > 0 || error("_invmod: modulus must be positive, got $p")
    a = _modp(a, p)
    a == 0 && return 0
    t, newt = 0, 1
    r, newr = p, a
    while newr != 0
        q = r ÷ newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    end
    r != 1 && error("no inverse")
    _modp(t, p)
end

function rank_modp(A::AbstractMatrix{<:Integer}, p::Integer)
    m, n = size(A)
    M = [ _modp(A[i,j], p) for i in 1:m, j in 1:n ]
    r = 0
    col = 1
    for i in 1:m
        piv = 0
        for j in col:n
            if M[i,j] % p != 0
                piv = j
                break
            end
        end
        if piv == 0
            for k in i+1:m
                for j in col:n
                    if M[k,j] % p != 0
                        M[i,:], M[k,:] = M[k,:], M[i,:]
                        piv = j
                        break
                    end
                end
                piv != 0 && break
            end
            if piv == 0
                continue
            end
        end
        inv = _invmod(M[i,piv], p)
        for j in piv:n
            M[i,j] = _modp(M[i,j] * inv, p)
        end
        for k in 1:m
            if k != i && M[k,piv] % p != 0
                factor = M[k,piv]
                for j in piv:n
                    M[k,j] = _modp(M[k,j] - factor * M[i,j], p)
                end
            end
        end
        r += 1
        col = piv + 1
        if col > n
            break
        end
    end
    r
end

function count_solutions_modp(A::AbstractMatrix{<:Integer}, b::AbstractVector{<:Integer}, p::Integer)
    m, n = size(A)
    @assert length(b) == m
    rA = rank_modp(A, p)
    Ab = hcat(A, reshape(b, m, 1))
    rAb = rank_modp(Ab, p)
    if rA != rAb
        return 0
    end
    p == 0 ? 0 : Int(big(p)^(n - rA))
end

# --- Correspondences (cycles on X×Y) as multi-valued linear maps ---
function correspondence_matrix(cycles::Vector{Tuple{Int,Int,Float64}}, m::Int, n::Int)
    M = zeros(Float64, n, m)
    for (i,j,w) in cycles
        @assert 1 <= i <= m
        @assert 1 <= j <= n
        M[j,i] += w
    end
    M
end

apply_correspondence(M::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}) = M * x

compose_correspondences(M2::AbstractMatrix{<:Real}, M1::AbstractMatrix{<:Real}) = M2 * M1

# --- Motivic feature vector over primes (counts of F_p-solutions) ---
function motivic_features(A::AbstractMatrix{<:Integer}, b::AbstractVector{<:Integer}, primes::Vector{Int})
    feats = Float64[]
    for p in primes
        cnt = count_solutions_modp(A, b, p)
        push!(feats, log10(float(cnt) + 1.0))
    end
    feats
end

export Graph, betti0, relabel, rank_modp, count_solutions_modp,
       correspondence_matrix, apply_correspondence, compose_correspondences, motivic_features

end
