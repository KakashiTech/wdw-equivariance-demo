#!/usr/bin/env julia
# Measure tradeoff between A asymmetry and Cₙ≠Dₙ gap
# Start with A=I, train with various λ_asym, observe gap decay.

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
        base=make_random_signal(n)
        rev=reflect(base)
        for (ci,sig) in enumerate([base,rev])
            cls=2*(pair-1)+ci
            for _ in 1:(n_train÷n_classes)
                push!(xs_train,shift(sig,rand(rng,0:n-1))); push!(ys_train,cls)
            end
            for _ in 1:(n_test÷n_classes)
                push!(xs_test,shift(sig,rand(rng,0:n-1))); push!(ys_test,cls)
            end
        end
    end
    return xs_train, ys_train, xs_test, ys_test
end

function train_wc_only!(layer, Wc, bc, xs, ys, lr)
    gs=Zygote.gradient((Wc_, bc_) -> begin
        tot=0.0
        for i in eachindex(ys)
            logits=combined_bispec_features(xs[i],layer) |> f -> Wc_ * f + bc_
            lm=maximum(logits); ps=exp.(logits.-lm)/sum(exp.(logits.-lm))
            tot+=-log(max(ps[ys[i]],eps()))
        end
        tot/length(ys)
    end, Wc, bc)
    Wc.-=lr*gs[1]; bc.-=lr*gs[2]
end

function measure_tradeoff(λ_asym_val, xs_train, ys_train, xs_test, ys_test)
    rng = MersenneTwister(SEED)
    layer=CyclicFourierLayer(N;seed=SEED)
    Wc=zeros(N_CLASSES,3*N); bc=zeros(N_CLASSES)
    
    for epoch in 1:30
        perm=randperm(N_TRAIN)
        for b in 1:BATCH_SIZE:N_TRAIN
            idx=perm[b:min(b+BATCH_SIZE-1,N_TRAIN)]
            train_wc_only!(layer,Wc,bc,xs_train[idx],ys_train[idx],0.02)
        end
    end
    
    xs_dn=[reflect(x) for x in xs_test]
    asys=Float64[]; gaps=Float64[]
    
    for epoch in 1:200
        perm=randperm(N_TRAIN)
        for b in 1:BATCH_SIZE:N_TRAIN
            idx=perm[b:min(b+BATCH_SIZE-1,N_TRAIN)]
            train_bispec_step!(layer,Wc,bc,xs_train[idx],ys_train[idx],0.001,
                              λ_asym=λ_asym_val, A_norm_max=10.0)
        end
        acc=accuracy_bispec(layer,Wc,bc,xs_test,ys_test;dn=false)
        dn=accuracy_bispec(layer,Wc,bc,xs_dn,ys_test;dn=false)
        push!(asys,cn_ne_dn_loss(layer))
        push!(gaps,acc-dn)
    end
    return (λ=λ_asym_val, final_asym=asys[end], max_gap=maximum(gaps),
            final_gap=gaps[end], asys=asys, gaps=gaps)
end

function main()
    println("="^65)
    println("Asymmetry vs Gap tradeoff")
    println("="^65)
    
    xs_train, ys_train, xs_test, ys_test = make_dataset(N, N_PAIRS, N_CLASSES, N_TRAIN, N_TEST, seed=SEED)
    @printf "Train: %d, Test: %d\n" length(ys_train) length(ys_test)
    
    results = []
    for λ in [0.0, -0.01, -0.1, -1.0, -10.0]
        @printf "\n── λ_asym = %6.2f ──\n" λ
        r = measure_tradeoff(λ, xs_train, ys_train, xs_test, ys_test)
        push!(results, r)
        @printf "  final_asym=%.4f final_gap=%.1fpp max_gap=%.1fpp\n" r.final_asym r.final_gap r.max_gap
    end
    
    println("\n"*"="^65)
    println("SUMMARY")
    @printf " %7s | %8s | %8s | %8s\n" "λ_asym" "asym_max" "final_gap" "max_gap"
    println("-"^45)
    for r in results
        @printf " %7.2f |  %.4f  |   %5.1fpp  |  %5.1fpp\n" r.λ maximum(r.asys) r.final_gap r.max_gap
    end
    println("="^65)
end

main()
