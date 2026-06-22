#!/usr/bin/env julia
# Cₙ≠Dₙ v4 — Bispectrum with careful training
# Phase 1: Train Wc ONLY (A=I frozen) — can we learn from bispectrum?
# Phase 2: Fine-tune A with low LR

include("../../src/Core/FFTGroup.jl")
using .FFTGroup
using LinearAlgebra, Random, Statistics, Printf, Zygote

const N = 32
const N_CLASSES = 4
const N_PAIRS = 2
const N_TRAIN = 800
const N_TEST = 200
const BATCH_SIZE = 20
const SEED = 42

function shift(x::Vector{T}, k::Int) where T
    n = length(x)
    return [x[mod1(i-k, n)] for i in 1:n]
end

function reflect(x::Vector{T}) where T
    n = length(x)
    return [x[mod1(-i+2, n)] for i in 1:n]
end

function make_random_signal(n::Int; noise=0.05)
    x̂ = Complex{Float64}[]
    n2 = n ÷ 2
    push!(x̂, randn() * sqrt(n))
    for ω in 2:n2
        mag = abs(randn()) * sqrt(n / 2)
        θ = rand() * 2π
        push!(x̂, mag * exp(im * θ))
    end
    n % 2 == 0 && push!(x̂, randn() * sqrt(n / 2))
    for ω in n2+2:n
        push!(x̂, conj(x̂[n-ω+2]))
    end
    x = real(FFTGroup.ifft_dispatch(x̂))
    x .+= noise * randn(n)
    return x / sqrt(sum(abs2, x))
end

function make_dataset(n::Int, n_pairs::Int, n_classes::Int, n_train::Int, n_test::Int; seed=42)
    rng = MersenneTwister(seed)
    xs_train = Vector{Float64}[]
    ys_train = Int[]
    xs_test = Vector{Float64}[]
    ys_test = Int[]
    for pair in 1:n_pairs
        base = make_random_signal(n; noise=0.05)
        rev_base = reflect(base)
        for (cls_idx, sig) in enumerate([base, rev_base])
            cls = 2*(pair-1) + cls_idx
            for _ in 1:(n_train÷n_classes)
                push!(xs_train, shift(sig, rand(rng, 0:n-1)))
                push!(ys_train, cls)
            end
            for _ in 1:(n_test÷n_classes)
                push!(xs_test, shift(sig, rand(rng, 0:n-1)))
                push!(ys_test, cls)
            end
        end
    end
    return xs_train, ys_train, xs_test, ys_test
end

# Custom step that only trains Wc, bc (freezes A, b)
function train_wc_only!(layer, Wc, bc, xs, ys, lr)
    gs = Zygote.gradient(
        (Wc_, bc_) -> begin
            tot = zero(Float64)
            for i in eachindex(ys)
                logits = combined_bispec_features(xs[i], layer) |> f -> Wc_ * f + bc_
                lm = maximum(logits)
                ps = exp.(logits .- lm) / sum(exp.(logits .- lm))
                tot += -log(max(ps[ys[i]], eps(Float64)))
            end
            return tot / length(ys)
        end,
        Wc, bc)
    Wc .-= lr * gs[1]
    bc .-= lr * gs[2]
end

function main()
    rng = MersenneTwister(SEED)
    println("="^65)
    println("Cₙ≠Dₙ v4 — Bispectrum, phased training")
    println("n=$N, n_classes=$N_CLASSES, shift_aug=YES")
    println("="^65)

    xs_train, ys_train, xs_test, ys_test = make_dataset(N, N_PAIRS, N_CLASSES, N_TRAIN, N_TEST, seed=SEED)
    n_actual = length(Set(ys_train))
    n_feat = 3 * N
    @printf "Train: %d, Test: %d, Classes: %d, Feat: %d\n" length(ys_train) length(ys_test) n_actual n_feat

    layer = CyclicFourierLayer(N; seed=SEED)
    Wc = zeros(n_actual, n_feat)
    bc = zeros(n_actual)

    # Phase 1: Train ONLY Wc (A=I frozen) on bispectrum features
    println("\n── Phase 1: Wc only (A=I frozen) ──")
    best_cn = 0.0
    for epoch in 1:200
        perm = randperm(length(ys_train))
        lr = 0.05 * (1 - epoch/200) + 0.001  # linear decay
        for b_start in 1:BATCH_SIZE:length(ys_train)
            b_end = min(b_start + BATCH_SIZE - 1, length(ys_train))
            idx = perm[b_start:b_end]
            train_wc_only!(layer, Wc, bc, xs_train[idx], ys_train[idx], lr)
        end
        if epoch % 40 == 0
            acc = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=false)
            best_cn = max(best_cn, acc)
            @printf "  epoch %3d | Cₙ test: %.1f%% (best: %.1f%%)\n" epoch acc best_cn
        end
    end
    acc_wc = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=false)
    @printf "\n  After Phase 1: Cₙ = %.1f%% (A=I, only Wc trained)\n" acc_wc

    # Evaluation before A fine-tune
    xs_test_dn = [reflect(x) for x in xs_test]
    acc_dn_before = accuracy_bispec(layer, Wc, bc, xs_test_dn, ys_test; dn=false)
    @printf "                Dₙ = %.1f%%\n" acc_dn_before
    @printf "             gap  = %.1fpp\n" (acc_wc - acc_dn_before)

    # Phase 2: Joint fine-tune A with low LR — shorter run
    # As A moves from I → asymmetric, gap DECREASES from 100pp.
    # Goal: find A with measurable asymmetry AND >10pp gap.
    println("\n── Phase 2: Joint fine-tune (λ_asym=-0.05) ──")
    best_gap = -Inf
    for epoch in 1:60
        perm = randperm(length(ys_train))
        n_batches = max(1, length(ys_train) ÷ BATCH_SIZE)
        for b in 1:n_batches
            idx = perm[(b-1)*BATCH_SIZE+1:min(b*BATCH_SIZE, length(ys_train))]
            train_bispec_step!(layer, Wc, bc, xs_train[idx], ys_train[idx], 0.002,
                               λ_asym=-0.05, A_norm_max=5.0)
        end
        acc_cn = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=false)
        acc_dn = accuracy_bispec(layer, Wc, bc, xs_test_dn, ys_test; dn=false)
        gap = acc_cn - acc_dn
        best_gap = max(best_gap, gap)
        asym = cn_ne_dn_loss(layer)
        A_mags = [abs(layer.A[ω]) for ω in 1:N]
        @printf "  epoch %2d | Cₙ: %.1f%% | Dₙ: %.1f%% | gap: %.1fpp | asym: %.4f | |A|: [%.2f, %.2f]\n" epoch acc_cn acc_dn gap asym minimum(A_mags) maximum(A_mags)
    end

    # Final
    println("\n── Final ──")
    acc_cn = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=false)
    acc_dn = accuracy_bispec(layer, Wc, bc, xs_test_dn, ys_test; dn=false)
    acc_oracle = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=true)
    asym = cn_ne_dn_loss(layer)
    @printf "  Cₙ:       %.1f%%\n" acc_cn
    @printf "  Dₙ:       %.1f%%\n" acc_dn
    @printf "  Oracle:   %.1f%%\n" acc_oracle
    @printf "  GAP:      %.1fpp\n" (acc_cn - acc_dn)
    @printf "  Best:     %.1fpp\n" best_gap
    @printf "  Asym:     %.4f\n" asym
    @printf "  MDL:      %.0f×\n" fft_mdl_ratio(N)
    xr = exact_recovery(xs_test[1], layer)
    @printf "  Recovery: %.2e\n" mean(abs2, xs_test[1] - xr)

    gap = acc_cn - acc_dn
    println("\n" * "="^65)
    if gap >= 10
        println("✓✓✓ GAP >= 10pp — VALIDATED with shift augmentation!")
    elseif best_gap >= 10
        println("⚠ Best gap was $(round(best_gap,digits=1))pp but final is $(round(gap,digits=1))pp")
    else
        println("✗ Gap = $(round(gap,digits=1))pp")
    end
    println("="^65)
end

main()
