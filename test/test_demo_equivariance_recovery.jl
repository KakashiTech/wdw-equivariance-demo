using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "equivariance_recovery.jl"))

@testset "demo_equivariance_recovery" begin
    n = 12
    G = dihedral_group(n)
    # Deterministic base matrix
    M = reshape(collect(1:n*n), n, n) .* 0.01
    W_eq = project_equivariant(M, G)
    e_eq = eqerr(W_eq, G)

    # Deterministic rupture (non-equivariant w.r.t. permutations)
    R = Diagonal(collect(1:n))
    W_rupt = W_eq + 0.10 * R
    e_rupt = eqerr(W_rupt, G)

    W_rec = project_equivariant(W_rupt, G)
    e_rec = eqerr(W_rec, G)

    @test e_eq <= 1e-8
    @test e_rec <= 1e-8
    @test e_rec <= 1e-2 * e_rupt
end
