#!/usr/bin/env julia
# Cₙ≠Dₙ experiment v2 — combined features + negative λ_asym + A norm
# Strategy:
#   1. Warm-start: freeze A=identity, train Wc on power spectrum (400 epochs)
#   2. Fine-tune: enable A + negative λ_asym + A norm, joint training
#   3. Evaluate Cₙ vs Dₙ accuracy gap

include("../../src/Core/FFTGroup.jl")
using .FFTGroup
using LinearAlgebra, Random, Statistics, Printf

# =============================================================================
# CONFIG
# =============================================================================

const N = 32
const N_CLASSES = 4      # 2 time-reversal pairs
const N_PAIRS = 2
const N_TRAIN = 800      # 200 per class
const N_TEST = 200       # 50 per class
const EPOCHS_WARM = 100
const EPOCHS_JOINT = 500
const BATCH_SIZE = 10
const LR = 0.01
const SEED = 42

# =============================================================================
# NO-SHIFT TIME-REVERSAL DATASET
# Training with FIXED phase (no random shifts) so the linear classifier
# can learn phase-specific patterns. Power spectrum alone can only
# distinguish PAIRS (50% for 4 classes). Combined features can reach 100%.
# Under Dₙ (reflection), complex features reverse → accuracy drops.
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

function make_dataset(n::Int, n_pairs::Int, n_classes::Int, n_train::Int, n_test::Int; seed=42, shift_aug=true)
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
                if shift_aug
                    k = rand(rng, 0:n-1)
                    push!(xs_train, shift(sig, k))
                else
                    push!(xs_train, copy(sig))
                end
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

function warm_start!(layer, Wc, bc, xs, ys; epochs=200, lr=0.01, batch_size=5)
    println("Warm-start: training Wc with A=identity (power spectrum only)...")
    n_feat_parity = layer.n + (layer.n ÷ 2 - 1)
    n_actual = length(Set(ys))
    Wc_p = 0.01 * randn(n_actual, n_feat_parity)
    bc_p = zeros(n_actual)
    n = length(ys)
    for epoch in 1:epochs
        perm = randperm(n)
        for batch_start in 1:batch_size:n
            batch_end = min(batch_start + batch_size - 1, n)
            idx = perm[batch_start:batch_end]
            train_step!(layer, Wc_p, bc_p, xs[idx], ys[idx], lr, λ_cndn=0.0)
        end
        if epoch % 50 == 0
            acc_cn = accuracy(layer, Wc_p, bc_p, xs, ys; dn=false)
            @printf "  epoch %4d | Cₙ parity acc: %.1f%%\n" epoch acc_cn
        end
    end
    n_actual = size(Wc, 1)
    for c in 1:n_actual
        for j in 1:layer.n
            Wc[c, j] = Wc_p[c, j]
        end
    end
    return Wc_p, bc_p
end

function joint_finetune!(layer, Wc, bc, xs, ys; epochs=300, lr=0.005, λ_asym=-0.1, A_norm_max=5.0, batch_size=5)
    println("Joint fine-tune: A active, λ_asym=$λ_asym, A_norm_max=$A_norm_max...")
    n = length(ys)
    for epoch in 1:epochs
        perm = randperm(n)
        for batch_start in 1:batch_size:n
            batch_end = min(batch_start + batch_size - 1, n)
            idx = perm[batch_start:batch_end]
            train_cndn_step!(layer, Wc, bc, xs[idx], ys[idx], lr,
                             λ_asym=λ_asym, A_norm_max=A_norm_max)
        end
        if epoch % 50 == 0
            acc_cn = accuracy_combined(layer, Wc, bc, xs, ys; dn=false)
            asym = cn_ne_dn_loss(layer)
            @printf "  epoch %4d | Cₙ comb acc: %.1f%% | asym: %.4f\n" epoch acc_cn asym
        end
    end
end

# =============================================================================
# MAIN
# =============================================================================

function main()
    rng = MersenneTwister(SEED)
    println("="^65)
    println("Cₙ≠Dₙ v2 — Combined Features + Negative λ_asym + A Norm")
    println("n=$N, n_classes=$N_CLASSES, n_train=$N_TRAIN, n_test=$N_TEST")
    println("="^65)

    # Dataset — NO SHIFT AUGMENTATION during training
    println("\nGenerating dataset ($N_PAIRS time-reversal pairs, $N_CLASSES classes, NO shift aug)...")
    xs_train, ys_train, xs_test, ys_test = make_dataset(N, N_PAIRS, N_CLASSES, N_TRAIN, N_TEST, seed=SEED, shift_aug=false)
    n_actual = length(Set(ys_train))
    @printf "  Train: %d samples, %d classes | Test: %d samples\n" length(ys_train) n_actual length(ys_test)

    # Model — start with A ≈ identity (slightly perturbed)
    layer = CyclicFourierLayer(N; seed=SEED)
    n_feat = 3 * N  # power (N) + complex_re (N) + complex_im (N)
    Wc = 0.01 * randn(n_actual, n_feat)
    bc = zeros(n_actual)

    # Phase 1: warm-start (power spectrum only)
    println("\n── Phase 1: Warm-start ──")
    Wc_p, bc_p = warm_start!(layer, Wc, bc, xs_train, ys_train,
                             epochs=EPOCHS_WARM, lr=LR, batch_size=BATCH_SIZE)

    # Evaluate after warm-start (parity features — inherently Dₙ-symmetric)
    acc_cn_warm = accuracy(layer, Wc_p, bc_p, xs_test, ys_test; dn=false)
    acc_dn_warm = accuracy(layer, Wc_p, bc_p, xs_test, ys_test; dn=true)
    @printf "\n  After warm-start (parity) — Cₙ: %.1f%% | Dₙ: %.1f%% | gap: %.1fpp\n" acc_cn_warm acc_dn_warm (acc_cn_warm - acc_dn_warm)

    @printf "  After warm-start (combined) — Cₙ: %.1f%%\n" accuracy_combined(layer, Wc, bc, xs_test, ys_test; dn=false)

    # Phase 2: joint fine-tune (combined features, negative λ_asym)
    println("\n── Phase 2: Joint fine-tune ──")
    joint_finetune!(layer, Wc, bc, xs_train, ys_train,
                    epochs=EPOCHS_JOINT, lr=LR * 0.5,
                    λ_asym=-0.1, A_norm_max=5.0)

    # Final evaluation
    # Cₙ: parity logits on original signals (uses f+g features, g non-zero with asymmetric A)
    # Dₙ: parity logits on REFLECTED signals (power spectrum unchanged, g depends on |x̂|² which is same)
    # Combined: full combined features (power + complex) on original vs reflected signals
    println("\n── Final evaluation ──")

    # Parity features — Cₙ vs Dₙ gap from g being zeroed under Dₙ
    acc_par_cn = accuracy(layer, Wc_p, bc_p, xs_test, ys_test; dn=false)
    acc_par_dn = accuracy(layer, Wc_p, bc_p, xs_test, ys_test; dn=true)

    # Combined features — Cₙ vs Dₙ gap from complex features changing under reflection
    xs_test_rev = [reflect(x) for x in xs_test]
    acc_cn = accuracy_combined(layer, Wc, bc, xs_test, ys_test; dn=false)
    acc_dn = accuracy_combined(layer, Wc, bc, xs_test_rev, ys_test; dn=false)

    asym = cn_ne_dn_loss(layer)
    A_mags = [abs(layer.A[ω]) for ω in 1:N]
    @printf "\n  Parity Cₙ:         %.1f%% (f+g features)\n" acc_par_cn
    @printf "  Parity Dₙ:         %.1f%% (f+g features, Dₙ-symmetrized)\n" acc_par_dn
    @printf "  Parity Cₙ≠Dₙ gap:  %.1f pp\n" (acc_par_cn - acc_par_dn)
    @printf "\n  Combined Cₙ:       %.1f%% (power + complex)\n" acc_cn
    @printf "  Combined Dₙ:       %.1f%% (power + complex, reflected input)\n" acc_dn
    @printf "  Combined Cₙ≠Dₙ gap: %.1f pp\n" (acc_cn - acc_dn)
    @printf "\n  A asymmetry:       %.4f\n" asym
    @printf "  A mag range:       [%.2f, %.2f]\n" minimum(A_mags) maximum(A_mags)

    # Recovery verification
    x_test = xs_test[1]
    x_rec = exact_recovery(x_test, layer)
    mse = mean(abs2, x_test - x_rec)
    @printf "\n  Recovery MSE: %.2e\n" mse

    # MDL ratio
    mdl_ratio = fft_mdl_ratio(N)
    @printf "  MDL ratio (n=%d): %.1f×\n" N mdl_ratio

    # Summary
    gap = acc_cn - acc_dn
    println("\n"^2 * "="^65)
    if gap > 10
        println("✓✓✓ Cₙ≠Dₙ GAP > 10pp — CLAIM VALIDATED!")
    elseif gap > 5
        println("✓ Cₙ≠Dₙ gap = $(round(gap, digits=1))pp — improving but <10pp target")
    else
        println("✗ Cₙ≠Dₙ gap = $(round(gap, digits=1))pp — needs improvement")
    end
    println("="^65)

    # Save results
    open("bench/fft_pipeline/results_cndn_v2.csv", "w") do io
        println(io, "acc_cn,acc_dn,gap,asym,recovery_mse,mdl_ratio")
        println(io, "$acc_cn,$acc_dn,$gap,$asym,$mse,$mdl_ratio")
    end
    println("Results saved to bench/fft_pipeline/results_cndn_v2.csv")
end

main()
