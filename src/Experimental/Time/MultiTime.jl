# TIER 3 — EXPERIMENTAL: Multi-agent time evolution
module TimeMulti

using LinearAlgebra

function step(A_list::Vector{<:AbstractMatrix}, x::AbstractVector, dt_list::Vector{<:Real})
    @assert length(A_list) == length(dt_list)
    y = copy(x)
    for (A, dt) in zip(A_list, dt_list)
        L = A' * A
        y = y - dt * (L * y)
    end
    y
end

function simulate(A_list::Vector{<:AbstractMatrix}, x0::AbstractVector, dt_list::Vector{<:Real}, steps::Int)
    x = copy(x0)
    traj = Vector{typeof(x0)}()
    push!(traj, copy(x))
    for _ in 1:steps
        x = step(A_list, x, dt_list)
        push!(traj, copy(x))
    end
    traj
end

function is_stable(traj; factor::Real=10.0)
    n0 = norm(traj[1]) + 1e-12
    all(norm(x) <= factor * n0 for x in traj)
end

export step, simulate, is_stable

end
