# TIER 3 — EXPERIMENTAL: Kripke semantics for modal logic
module Semantics

using ..Logic

struct KripkeModel
    worlds::Vector{Int}
    leq::BitMatrix
    valuation::Dict{Symbol, BitVector}
end

function is_monotone(m::KripkeModel)
    for (_, vec) in m.valuation
        for i in eachindex(m.worlds), j in eachindex(m.worlds)
            if m.leq[i,j] && vec[i] && !vec[j]
                return false
            end
        end
    end
    true
end

forces(m::KripkeModel, w::Int, ::Logic.Top) = true
forces(m::KripkeModel, w::Int, ::Logic.Bot) = false
forces(m::KripkeModel, w::Int, φ::Logic.Var) = m.valuation[φ.name][w]
forces(m::KripkeModel, w::Int, φ::Logic.And) = forces(m,w,φ.left) && forces(m,w,φ.right)
forces(m::KripkeModel, w::Int, φ::Logic.Or) = forces(m,w,φ.left) || forces(m,w,φ.right)
function forces(m::KripkeModel, w::Int, φ::Logic.Imply)
    for j in eachindex(m.worlds)
        if m.leq[w,j] && forces(m,j,φ.left) && !forces(m,j,φ.right)
            return false
        end
    end
    true
end
function forces(m::KripkeModel, w::Int, φ::Logic.Not)
    for j in eachindex(m.worlds)
        if m.leq[w,j] && forces(m,j,φ.inner)
            return false
        end
    end
    true
end

export KripkeModel, is_monotone, forces

end
