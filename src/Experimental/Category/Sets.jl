# TIER 3 — EXPERIMENTAL: Finite set theory and function maps
module Category

struct FinSet{T}
    elements::Vector{T}
end

struct FunctionMap{A,B}
    dom::FinSet{A}
    codom::FinSet{B}
    map::Dict{A,B}
end

function pullback(f::FunctionMap{A,C}, g::FunctionMap{B,C}) where {A,B,C}
    pairs = Tuple{A,B}[]
    for a in f.dom.elements
        fa = f.map[a]
        for b in g.dom.elements
            if g.map[b] == fa
                push!(pairs, (a,b))
            end
        end
    end
    pb_dom = FinSet(pairs)
    π1 = FunctionMap(pb_dom, f.dom, Dict(p => p[1] for p in pairs))
    π2 = FunctionMap(pb_dom, g.dom, Dict(p => p[2] for p in pairs))
    return pb_dom, π1, π2
end

function omega()
    FinSet([false, true])
end

function terminal()
    FinSet([nothing])
end

function true_map()
    FunctionMap(terminal(), omega(), Dict(nothing => true))
end

function characteristic(X::FinSet{T}, subset::Vector{T}) where {T}
    m = Dict{T,Bool}()
    s = Set(subset)
    for x in X.elements
        m[x] = in(x, s)
    end
    FunctionMap(X, omega(), m)
end

export FinSet, FunctionMap, pullback, omega, terminal, true_map, characteristic

end
