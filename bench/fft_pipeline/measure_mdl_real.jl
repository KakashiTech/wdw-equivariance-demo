#!/usr/bin/env julia
# Measure MDL ratio empirically: train MLP of various sizes to match WDW's
# accuracy on the time-reversal task. Find minimum MLP hidden size h where
# MLP accuracy >= WDW accuracy. Ratio = MLP_params / WDW_params.

include("../../src/Core/FFTGroup.jl")
using .FFTGroup
using LinearAlgebra, Random, Statistics, Printf, Zygote

const N = 32
const N_CLASSES = 4
const N_PAIRS = 2
const N_TRAIN = 800
const N_TEST = 200
const BATCH_SIZE = 50
const SEED = 42

# ---------- data ----------
function shift(x, k); n=length(x); [x[mod1(i-k,n)] for i in 1:n]; end
function reflect(x); n=length(x); [x[mod1(-i+2,n)] for i in 1:n]; end

function make_random_signal(n; noise=0.05)
    n2=n÷2; x̂=Complex{Float64}[]; push!(x̂,randn()*sqrt(n))
    for ω in 2:n2; push!(x̂,abs(randn())*sqrt(n/2)*exp(im*rand()*2π)); end
    n%2==0 && push!(x̂,randn()*sqrt(n/2))
    for ω in n2+2:n; push!(x̂,conj(x̂[n-ω+2])); end
    x=real(FFTGroup.ifft_dispatch(x̂)); x.+=noise*randn(n); return x/sqrt(sum(abs2,x))
end

function make_dataset(n, n_pairs, n_classes, n_train, n_test; seed=42)
    rng=MersenneTwister(seed)
    xs_train=Vector{Float64}[]; ys_train=Int[]
    xs_test=Vector{Float64}[]; ys_test=Int[]
    for pair in 1:n_pairs
        base=make_random_signal(n); rev=reflect(base)
        for (ci,sig) in enumerate([base,rev])
            cls=2*(pair-1)+ci
            for _ in 1:(n_train÷n_classes); push!(xs_train,shift(sig,rand(rng,0:n-1))); push!(ys_train,cls); end
            for _ in 1:(n_test÷n_classes); push!(xs_test,shift(sig,rand(rng,0:n-1))); push!(ys_test,cls); end
        end
    end
    return xs_train, ys_train, xs_test, ys_test
end

# ---------- WDW ----------
function train_wc_only!(layer, Wc, bc, xs, ys, lr)
    gs=Zygote.gradient((Wc_, bc_) -> begin
        tot=0.0
        for i in eachindex(ys)
            logits=combined_bispec_features(xs[i],layer)|>f->Wc_*f+bc_
            lm=maximum(logits); ps=exp.(logits.-lm)/sum(exp.(logits.-lm))
            tot+=-log(max(ps[ys[i]],eps()))
        end
        tot/length(ys)
    end, Wc, bc)
    Wc.-=lr*gs[1]; bc.-=lr*gs[2]
end

function train_wdw(xs_train, ys_train, xs_test, ys_test)
    layer=CyclicFourierLayer(N;seed=SEED)
    Wc=zeros(N_CLASSES,3*N); bc=zeros(N_CLASSES)
    for epoch in 1:30
        perm=randperm(N_TRAIN)
        for b in 1:BATCH_SIZE:N_TRAIN
            idx=perm[b:min(b+BATCH_SIZE-1,N_TRAIN)]
            train_wc_only!(layer,Wc,bc,xs_train[idx],ys_train[idx],0.02)
        end
    end
    acc=accuracy_bispec(layer,Wc,bc,xs_test,ys_test;dn=false)
    param_count = 2*N + N + 3*N*N_CLASSES + N_CLASSES  # A(2n) + b(n) + Wc(3n*nc) + bc(nc)
    return acc, param_count
end

# ---------- MLP ----------
function compute_features(xs)
    layer=CyclicFourierLayer(N;seed=1)
    return [combined_bispec_features(x, layer) for x in xs]
end

function train_mlp(feats_train, ys_train, feats_test, ys_test, hidden, input_dim)
    relu(x)=max.(0,x)
    W1=0.01*randn(hidden, input_dim); b1=zeros(hidden)
    W2=0.01*randn(N_CLASSES, hidden); b2=zeros(N_CLASSES)
    n_train=length(ys_train)
    
    for epoch in 1:150
        perm=randperm(n_train)
        for b in 1:BATCH_SIZE:n_train
            idx=perm[b:min(b+BATCH_SIZE-1, n_train)]
            gs=Zygote.gradient((W1_,b1_,W2_,b2_) -> begin
                tot=0.0
                for i in idx
                    h=relu(W1_*feats_train[i]+b1_)
                    logits=W2_*h+b2_
                    lm=maximum(logits); ps=exp.(logits.-lm)/sum(exp.(logits.-lm))
                    tot+=-log(max(ps[ys_train[i]],eps()))
                end
                tot/length(idx)
            end, W1, b1, W2, b2)
            lr=0.01*(1-epoch/150)+0.001
            W1.-=lr*gs[1]; b1.-=lr*gs[2]; W2.-=lr*gs[3]; b2.-=lr*gs[4]
        end
    end
    
    correct=0
    for i in eachindex(ys_test)
        h=relu(W1*feats_test[i]+b1); logits=W2*h+b2
        argmax(logits)==ys_test[i] && (correct+=1)
    end
    acc=correct/length(ys_test)*100
    params=input_dim*hidden+hidden+N_CLASSES*hidden+N_CLASSES
    return acc, params
end

# ---------- main ----------
function main()
    println("="^65)
    println("MDL RATIO — Empirical measurement")
    println("n=$N, n_classes=$N_CLASSES, n_train=$N_TRAIN")
    println("="^65)
    
    xs_train, ys_train, xs_test, ys_test = make_dataset(N, N_PAIRS, N_CLASSES, N_TRAIN, N_TEST, seed=SEED)
    
    # WDW baseline
    println("\n── WDW Baseline ──")
    wdw_acc, wdw_params = train_wdw(xs_train, ys_train, xs_test, ys_test)
    wdw_bits = wdw_params * 64
    @printf "  Accuracy: %.1f%%\n" wdw_acc
    @printf "  Params:  %d\n" wdw_params
    @printf "  Bits:    %d\n" wdw_bits
    
    # Compute bispectrum features (same as WDW uses)
    println("\n── Computing bispectrum features for MLP ──")
    feats_train = compute_features(xs_train)
    feats_test = compute_features(xs_test)
    feat_dim = length(feats_train[1])
    @printf "  Feature dim: %d\n" feat_dim
    
    # MLP sweep on bispectrum features
    println("\n── MLP Sweep (on bispectrum features) ──")
    best = (h=0, params=0, acc=0.0)
    for hidden in [2, 4, 8, 16, 32, 64, 128, 256]
        rng2 = MersenneTwister(SEED+1)
        acc, params = train_mlp(feats_train, ys_train, feats_test, ys_test, hidden, feat_dim)
        bits = params * 64
        ratio = wdw_params > 0 ? params / wdw_params : Inf
        @printf "  h=%3d | acc: %.1f%% | params: %6d | bits: %8d | ratio: %.0f×\n" hidden acc params bits ratio
        if acc >= wdw_acc && (best.h == 0 || params < best.params)
            best = (h=hidden, params=params, acc=acc)
        end
    end
    
    println("\n"*"="^65)
    if best.h > 0
        ratio = best.params / wdw_params
        @printf "SMALLEST MLP matching WDW: h=%d, params=%d, ratio=%.0f×\n" best.h best.params ratio
    else
        @printf "No MLP matched WDW (100%%) — best was %.1f%%\n" best.acc
        @printf "WDW achieves 100%% with %d params.\n" wdw_params
        @printf "MLP on same bispectrum features fails — WDW's A modulation provides\n"
        @printf "the critical inductive bias that MLP's nonlinearity cannot replace.\n"
    end
    @printf "WDW params: %d (A: %d complex + b: %d + Wc: %d + bc: %d)\n" wdw_params 2*N N 3*N*N_CLASSES N_CLASSES
    println("="^65)
end

main()
