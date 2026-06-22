# TIER 3 — EXPERIMENTAL: Sheaf theory with constant sheaves and gluing
module Sheaves

using ..Knowledge: TopSpace, Partial, glue_partial

struct ConstantSheaf{T,U}
    space::TopSpace{T}
    values::Vector{U}
end

function glue(s::ConstantSheaf{T,U}, cover::Vector{Vector{T}}, sections::Vector{U}) where {T,U}
    if length(cover) != length(sections)
        return false, nothing
    end
    for Uset in cover
        if !(any(Set(Uset) == Set(O) for O in s.space.opens))
            return false, nothing
        end
    end
    if isempty(sections)
        return false, nothing
    end
    firstval = sections[1]
    for v in sections
        if v != firstval
            return false, nothing
        end
    end
    true, firstval
end

function sections_to_partials(s::ConstantSheaf{T,U}, cover::Vector{Vector{T}}, sections::Vector{U}) where {T,U}
    if length(cover) != length(sections)
        error("cover/sections length mismatch")
    end
    ps = Partial{T,U}[]
    for (Uset, val) in zip(cover, sections)
        push!(ps, Partial{T,U}(Uset, val))
    end
    ps
end

function glue_via_partials(s::ConstantSheaf{T,U}, cover::Vector{Vector{T}}, sections::Vector{U}) where {T,U}
    ps = sections_to_partials(s, cover, sections)
    glue_partial(ps)
end

export ConstantSheaf, glue, sections_to_partials, glue_via_partials

end
