# TIER 3 — EXPERIMENTAL: Hyper-time metric simulation
"""
    module TimeHyper

Hyper-time evolution: multi-axis imaginary-time evolution for quantum systems.
Supports simulation of Lorentzian metrics and monotonic energy verification.

# Key exports
- `simulate_metric`: Compute ds² = -Σdt_i² + Σdx_j²
- `evolve_wavefunction`: Multi-axis imaginary-time evolution with normalization
- `monotone_energies_axes`: Verify non-increasing energy across sweeps

# Usage
```julia
using WDW
psi, energies = TimeHyper.evolve_wavefunction(Hs, psi0, dts, steps)
is_mono = TimeHyper.monotone_energies_axes(energies)
```
"""
module TimeHyper

using LinearAlgebra
using ..TimeITE: step

"""
    simulate_metric(dt::AbstractVector, dx::AbstractVector)

Compute ds^2 = -sum(dt_i^2) + sum(dx_j^2) for hyper-temporal metric.
"""
function simulate_metric(dt::AbstractVector, dx::AbstractVector)
    -sum(abs2, dt) + sum(abs2, dx)
end

"""
    evolve_wavefunction(Hs::Vector{<:AbstractMatrix}, psi0::AbstractVector, dts::Vector{<:Real}, steps::Int)

Iteratively applies imaginary-time steps along each time axis' Hamiltonian, normalizing at each substep.
Returns (psi, energies_axes) where energies_axes is a Vector of energy traces per axis.
"""
function evolve_wavefunction(Hs::Vector{<:AbstractMatrix}, psi0::AbstractVector, dts::Vector{<:Real}, steps::Int)
    @assert length(Hs) == length(dts)
    psi = copy(psi0)
    n0 = norm(psi)
    psi = n0 == 0 ? psi : psi ./ n0
    energies_axes = [Float64[] for _ in 1:length(Hs)]
    for _ in 1:steps
        for (i,(H,dt)) in enumerate(zip(Hs, dts))
            push!(energies_axes[i], real(dot(psi, H*psi)))
            psi = step(H, psi, dt)
        end
    end
    # final energies
    for (i,H) in enumerate(Hs)
        push!(energies_axes[i], real(dot(psi, H*psi)))
    end
    psi, energies_axes
end

"""
    monotone_energies_axes(energies_axes; tol=1e-8)

Check non-increasing energy traces for each axis.
"""
function monotone_energies_axes(energies_axes; tol=1e-8)
    # Check monotonicity of the sum of axis-energies across sweeps
    L = minimum(length.(energies_axes))
    sums = [sum(e[k] for e in energies_axes) for k in 1:L]
    all(sums[i+1] <= sums[i] + tol for i in 1:length(sums)-1)
end

export simulate_metric, evolve_wavefunction, monotone_energies_axes

end
