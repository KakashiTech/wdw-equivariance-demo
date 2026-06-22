#!/usr/bin/env julia
# Verify bispectrum claims:
# 1. Shift-invariance: B_z(ω) same under any cyclic shift
# 2. Dₙ-sensitivity: B_z(ω) changes under reflection
# 3. Time-reversal pair discrimination

include("../../src/Core/FFTGroup.jl")
using .FFTGroup
using LinearAlgebra, Statistics, Printf, Random

function shift(x::Vector{T}, k::Int) where T
    n = length(x)
    return [x[mod1(i-k, n)] for i in 1:n]
end

function reflect(x::Vector{T}) where T
    n = length(x)
    return [x[mod1(-i+2, n)] for i in 1:n]
end

function make_signal(n::Int; seed=1)
    rng = MersenneTwister(seed)
    n2 = n ÷ 2
    x̂ = zeros(Complex{Float64}, n)
    x̂[1] = randn(rng) * sqrt(n)
    for ω in 2:n2
        mag = abs(randn(rng)) * sqrt(n / 2)
        θ = rand(rng) * 2π
        x̂[ω] = mag * exp(im * θ)
        x̂[n-ω+2] = conj(x̂[ω])
    end
    if n % 2 == 0
        x̂[n2+1] = randn(rng) * sqrt(n / 2)
    end
    x = real(FFTGroup.ifft_dispatch(x̂))
    return x / sqrt(sum(abs2, x))
end

function main()
    n = 32
    layer = CyclicFourierLayer(n; seed=42)

    println("="^60)
    println("BISPECTRUM THEORETICAL VERIFICATION")
    println("="^60)

    # Test 1: Shift invariance
    println("\n[1/4] Shift invariance test...")
    x = make_signal(n; seed=1)
    feat0 = combined_bispec_features(x, layer)
    max_diff = 0.0
    for k in 1:n-1
        x_shifted = shift(x, k)
        feat_k = combined_bispec_features(x_shifted, layer)
        d = norm(feat0 - feat_k)
        max_diff = max(max_diff, d)
    end
    @printf "  Max ‖feat(shifted) - feat(orig)‖: %.2e\n" max_diff
    @printf "  Shift invariance: %s\n" (max_diff < 1e-10 ? "✓ PERFECT" : "⚠ BROKEN")

    # Test 2: Dₙ sensitivity
    println("\n[2/4] Dₙ sensitivity test...")
    x_rev = reflect(x)
    feat_rev = combined_bispec_features(x_rev, layer)
    dn_diff = norm(feat0 - feat_rev)
    @printf "  ‖feat(reflected) - feat(orig)‖: %.4f\n" dn_diff
    @printf "  Dₙ sensitivity: %s\n" (dn_diff > 0.01 ? "✓ CHANGES under reflection" : "⚠ NO change")

    # Test 3: Time-reversal pair discrimination
    println("\n[3/4] Time-reversal pair discrimination...")
    x1 = make_signal(n; seed=1)
    x2 = reflect(x1)  # time reversal
    f1 = combined_bispec_features(x1, layer)
    f2 = combined_bispec_features(x2, layer)
    pair_dist = norm(f1 - f2)
    cross_dist = norm(f1 - combined_bispec_features(make_signal(n; seed=2), layer))
    @printf "  ‖feat(x) - feat(rev(x))‖: %.4f\n" pair_dist
    @printf "  ‖feat(x) - feat(other)‖: %.4f\n" cross_dist
    @printf "  Within-pair distinguishability: %s\n" (pair_dist > 0.01 ? "✓ DISTINGUISHABLE" : "⚠ NOT distinguishable")

    # Under Dₙ: reflected x1 should look like x2
    f1_dn = combined_bispec_features(reflect(x1), layer)
    dn_mistake = norm(f1_dn - f2)
    @printf "  ‖feat(reflect(x)) - feat(rev(x))‖: %.4f\n" dn_mistake
    @printf "  Dₙ confusion: %s\n" (dn_mistake < 0.01 ? "✓ COMPLETE confusion (reflect(x) looks like rev(x))" : 
                                    dn_mistake < pair_dist ? "⚠ PARTIAL confusion" : "✗ NO confusion")

    # Test 4: Feature values distribution
    println("\n[4/4] Feature statistics (across random signals)...")
    rng = MersenneTwister(99)
    n_signals = 100
    feat_re_mags = Float64[]
    feat_im_mags = Float64[]
    power_mags = Float64[]
    for _ in 1:n_signals
        x = make_signal(n; seed=rand(rng, 1:10000))
        f = combined_bispec_features(x, layer)
        n_power = n
        push!(power_mags, norm(f[1:n_power]) / sqrt(n_power))
        push!(feat_re_mags, norm(f[n_power+1:n_power+(n-2)]) / sqrt(n-2))
        push!(feat_im_mags, norm(f[n_power+(n-2)+1:end]) / sqrt(n-2))
    end
    @printf "  Power spectrum σ: %.4f\n" std(power_mags)
    @printf "  Bispec Re σ:      %.4f\n" std(feat_re_mags)
    @printf "  Bispec Im σ:      %.4f\n" std(feat_im_mags)
    m1 = mean(power_mags); m2 = mean(feat_re_mags); m3 = mean(feat_im_mags)
    @printf "  Mean |feat| ratio: power=%.2f bispec_re=%.2f bispec_im=%.2f\n" m1 m2 m3

    println("\n" * "="^60)
    if max_diff < 1e-10 && dn_diff > 0.01 && pair_dist > 0.01
        println("THEORETICAL VERDICT: ✓ Bispectrum works as claimed")
        println("  ✓ Shift-invariant (cancels phase)")
        println("  ✓ Dₙ-sensitive (changes under reflection)")
        println("  ✓ Distinguishes time-reversal pairs")
    else
        println("THEORETICAL VERDICT: ✗ Issues found")
    end
    println("="^60)
end

main()
