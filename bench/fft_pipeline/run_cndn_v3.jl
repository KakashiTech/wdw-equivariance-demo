#!/usr/bin/env julia
# Cₙ≠Dₙ v3 — BISPECTRUM features (shift-invariant, Dₙ-sensitive)
# Bispectrum: B_z(ω) = ẑ_ω · ẑ₁ · conj(ẑ_{ω+1})
# Under shift: EACH factor rotates, PRODUCT cancels → shift-INVARIANT
# Under Dₙ reflect: x̂ → conj(x̂_rev) → B_z changes → Dₙ-SENSITIVE
#
# This fixes the core issue: [Re(ẑ); Im(ẑ)] has zero mean under shifts.
# Bispectrum has NON-ZERO mean and captures phase structure.

include("../../src/Core/FFTGroup.jl")
using .FFTGroup
using LinearAlgebra, Random, Statistics, Printf, Zygote

const N = 32
const N_CLASSES = 4      # 2 time-reversal pairs
const N_PAIRS = 2
const N_TRAIN = 800      # 200 per class
const N_TEST = 200
const EPOCHS_WARM = 100
const EPOCHS_JOINT = 500
const BATCH_SIZE = 10
const LR = 0.01
const SEED = 42

# =============================================================================
# DATASET — time-reversal pairs WITH shift augmentation
# Bispectrum is shift-invariant, so augmentation helps (doesn't hurt).
# =============================================================================

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
    if n % 2 == 0
        mag = abs(randn()) * sqrt(n / 2)
        push!(x̂, (rand() > 0.5 ? 1.0 : -1.0) * mag)
    end
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
            cls = 2 * (pair - 1) + cls_idx
            for _ in 1:(n_train ÷ n_classes)
                k = rand(rng, 0:n-1)
                push!(xs_train, shift(sig, k))
                push!(ys_train, cls)
            end
            for _ in 1:(n_test ÷ n_classes)
                k = rand(rng, 0:n-1)
                push!(xs_test, shift(sig, k))
                push!(ys_test, cls)
            end
        end
    end
    return xs_train, ys_train, xs_test, ys_test
end

# =============================================================================
# TRAINING
# =============================================================================

function main()
    rng = MersenneTwister(SEED)
    println("="^65)
    println("Cₙ≠Dₙ v3 — BISPECTRUM (shift-invariant + Dₙ-sensitive)")
    println("n=$N, n_classes=$N_CLASSES, n_train=$N_TRAIN, shift_aug=YES")
    println("="^65)

    # Dataset with shift augmentation
    println("\nGenerating dataset ($N_PAIRS time-reversal pairs)...")
    xs_train, ys_train, xs_test, ys_test = make_dataset(N, N_PAIRS, N_CLASSES, N_TRAIN, N_TEST, seed=SEED)
    n_actual = length(Set(ys_train))
    n_feat_bispec = 3 * N  # power(N) + bispec_re(N) + bispec_im(N)
    @printf "  Train: %d samples, %d classes | Test: %d\n" length(ys_train) n_actual length(ys_test)
    @printf "  Feature dim: %d (power=%d + bispec=%d)\n" n_feat_bispec N (2*N)

    # Model
    layer = CyclicFourierLayer(N; seed=SEED)
    Wc = 0.01 * randn(n_actual, n_feat_bispec)
    bc = zeros(n_actual)

    # Phase 1: Warm-start — power spectrum only
    # Compute features once outside AD to avoid Zygote adjoint issues with ifft_dispatch
    println("\n── Phase 1: Warm-start (power spectrum) ──")
    Wc_p = 0.01 * randn(n_actual, N)
    bc_p = zeros(n_actual)
    ps_feats_train = [combined_bispec_features(x, layer)[1:N] for x in xs_train]
    ps_feats_test = [combined_bispec_features(x, layer)[1:N] for x in xs_test]
    for epoch in 1:EPOCHS_WARM
        perm = randperm(length(ys_train))
        for batch_start in 1:BATCH_SIZE:length(ys_train)
            batch_end = min(batch_start + BATCH_SIZE - 1, length(ys_train))
            idx = perm[batch_start:batch_end]
            gs = Zygote.gradient((Wc_, bc_) -> begin
                tot = 0.0
                for i in idx
                    logits = Wc_ * ps_feats_train[i] + bc_
                    lm = maximum(logits)
                    ps = exp.(logits .- lm) / sum(exp.(logits .- lm))
                    tot += -log(max(ps[ys_train[i]], eps()))
                end
                tot / length(idx)
            end, Wc_p, bc_p)
            Wc_p .-= LR * gs[1]
            bc_p .-= LR * gs[2]
        end
        if epoch % 50 == 0
            acc = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=false)
            @printf "  epoch %3d | power-spectrum test acc: %.1f%%\n" epoch acc
        end
    end
    # Copy power-spectrum weights to bispec Wc (first N columns)
    for c in 1:n_actual
        for j in 1:N
            Wc[c, j] = Wc_p[c, j]
        end
    end
    acc_warm = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=false)
    @printf "\n  After warm-start — bispec Cₙ: %.1f%%\n" acc_warm

    # Phase 2: Joint fine-tune with bispectrum + negative λ_asym
    println("\n── Phase 2: Joint fine-tune (bispectrum, λ_asym=-0.1) ──")
    best_gap = -Inf
    for epoch in 1:EPOCHS_JOINT
        perm = randperm(length(ys_train))
        for batch_start in 1:BATCH_SIZE:length(ys_train)
            batch_end = min(batch_start + BATCH_SIZE - 1, length(ys_train))
            idx = perm[batch_start:batch_end]
            train_bispec_step!(layer, Wc, bc, xs_train[idx], ys_train[idx], LR * 0.5,
                               λ_asym=-0.1, A_norm_max=5.0)
        end
        if epoch % 50 == 0
            acc_cn = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=false)
            xs_test_dn = [reflect(x) for x in xs_test]
            acc_dn = accuracy_bispec(layer, Wc, bc, xs_test_dn, ys_test; dn=false)
            gap = acc_cn - acc_dn
            asym = cn_ne_dn_asymmetry(layer)
            best_gap = max(best_gap, gap)
            @printf "  epoch %3d | Cₙ: %.1f%% | Dₙ: %.1f%% | gap: %.1fpp | asym: %.4f\n" epoch acc_cn acc_dn gap asym
        end
    end

    # Final evaluation
    println("\n── Final evaluation ──")
    xs_test_dn = [reflect(x) for x in xs_test]
    acc_cn = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=false)
    acc_dn = accuracy_bispec(layer, Wc, bc, xs_test_dn, ys_test; dn=false)
    acc_dn_oracle = accuracy_bispec(layer, Wc, bc, xs_test, ys_test; dn=true)
    asym = cn_ne_dn_asymmetry(layer)
    A_mags = [abs(layer.A[ω]) for ω in 1:N]

    @printf "\n"
    @printf "  Cₙ accuracy:          %.1f%%\n" acc_cn
    @printf "  Dₙ accuracy:          %.1f%% (reflected input)\n" acc_dn
    @printf "  Dₙ oracle:            %.1f%% (symmetrized A)\n" acc_dn_oracle
    @printf "  Cₙ≠Dₙ GAP:            %.1f pp\n" (acc_cn - acc_dn)
    @printf "  Best gap (epoch):     %.1f pp\n" best_gap
    @printf "  A asymmetry:          %.4f\n" asym
    @printf "  A mag range:          [%.2f, %.2f]\n" minimum(A_mags) maximum(A_mags)

    # Recovery & MDL
    x_test = xs_test[1]
    x_rec = exact_recovery(x_test, layer)
    mse = mean(abs2, x_test - x_rec)
    mdl_ratio = fft_mdl(layer) / (N * 64)
    @printf "\n  Recovery MSE:         %.2e\n" mse
    @printf "  MDL ratio (n=%d):     %.3f\n" N mdl_ratio

    gap = acc_cn - acc_dn
    println("\n" * "="^65)
    if gap >= 10
        println("✓✓✓ Cₙ≠Dₙ GAP >= 10pp — CLAIM VALIDATED with shift augmentation!")
    elseif gap >= 5
        println("✓ Cₙ≠Dₙ gap = $(round(gap,digits=1))pp — approaching target")
    else
        println("✗ Cₙ≠Dₙ gap = $(round(gap,digits=1))pp — needs work")
    end
    println("  Bispectrum features are shift-INVARIANT and Dₙ-SENSITIVE")
    println("="^65)
end

main()
