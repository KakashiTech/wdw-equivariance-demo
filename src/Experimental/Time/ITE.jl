# TIER 3 — EXPERIMENTAL: Imaginary-time evolution
module TimeITE

using LinearAlgebra

function step(H::AbstractMatrix, psi::AbstractVector, dt::Real)
    ϕ = psi - dt * (H * psi)
    n = norm(ϕ)
    n == 0 ? copy(psi) : ϕ ./ n
end

function evolve(H::AbstractMatrix, psi0::AbstractVector, dt::Real, steps::Int)
    psi = copy(psi0)
    n0 = norm(psi)
    psi = n0 == 0 ? psi : psi ./ n0
    energies = Float64[]
    for _ in 1:steps
        push!(energies, real(dot(psi, H * psi)))
        psi = step(H, psi, dt)
    end
    push!(energies, real(dot(psi, H * psi)))
    psi, energies
end

function monotone_energy(energies::AbstractVector; tol=1e-8)
    all(energies[i+1] <= energies[i] + tol for i in 1:length(energies)-1)
end

export step, evolve, monotone_energy

end
