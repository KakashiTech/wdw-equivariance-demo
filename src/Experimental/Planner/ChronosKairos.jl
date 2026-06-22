# TIER 3 — EXPERIMENTAL: Chronos-Kairos scheduling algorithms
module Planner

function interleave_roundrobin(seq::Vector{T}, pool::Vector{T}, k::Int) where {T}
    out = T[]
    p = 1
    c = 0
    for x in seq
        push!(out, x)
        c += 1
        if c == k && !isempty(pool)
            push!(out, pool[p])
            p = p == length(pool) ? 1 : p + 1
            c = 0
        end
    end
    out
end

"""
    schedule_ck(seq::Vector{T}, k::Int, branch_count::Int) where T

Construye un horario Chronos-Kairos simple intercalando cada k pasos de `seq` con eventos de una
lista de futuros de tamaño `branch_count` (identificados por enteros 1..branch_count).
"""
function schedule_ck(seq::Vector{T}, k::Int, branch_count::Int) where {T}
    futures = collect(1:branch_count)
    interleave_roundrobin(seq, futures, k)
end

export interleave_roundrobin, schedule_ck

end
