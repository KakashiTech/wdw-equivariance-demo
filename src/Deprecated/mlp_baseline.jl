# DEPRECATED: Legacy MLP baseline (kept for backward compatibility)
"""
    mlp_baseline.jl

Three-layer MLP baseline for comparison against WDW.
Uses Zygote for gradient computation. Provides forward, loss, predict, train.

# DEPRECATED
This file is NOT included in WDW.jl. Use `RigorousMetrics.evaluate_all_baselines_v3`
or `WDWAutoencoder.train_baseline_fair` for up-to-date baseline comparisons.

# Key exports
- `MLP`: Three-layer struct with weight matrices and bias vectors
- `mlp_forward`: Forward pass with tanh activations
- `mlp_loss`: Cross-entropy loss over a batch
- `mlp_predict`: Argmax classifier
- `mlp_train_step!`: Single SGD step via Zygote.gradient

# Usage
```julia
model = MLP(64, 32, 10)
mlp_train_step!(model, xs, ys, 0.01)
preds = mlp_predict(model, xs)
```
"""
# MLP baseline for comparison (DEPRECATED — see module docstring above)
# Three-layer MLP with similar parameter count to WDW

struct MLP{T}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
    W3::Matrix{T}
    b3::Vector{T}
end

function MLP(n::Int, hidden_dim::Int, n_classes::Int; seed=42)
    rng = MersenneTwister(seed)
    W1 = randn(rng, Float64, hidden_dim, n) * sqrt(2.0 / n)
    b1 = zeros(Float64, hidden_dim)
    W2 = randn(rng, Float64, hidden_dim, hidden_dim) * sqrt(2.0 / hidden_dim)
    b2 = zeros(Float64, hidden_dim)
    W3 = randn(rng, Float64, n_classes, hidden_dim) * sqrt(2.0 / hidden_dim)
    b3 = zeros(Float64, n_classes)
    return MLP(W1, b1, W2, b2, W3, b3)
end

function mlp_forward(model::MLP, x::Vector)
    x = tanh.(model.W1 * x + model.b1)
    x = tanh.(model.W2 * x + model.b2)
    return model.W3 * x + model.b3
end

function mlp_loss(model::MLP, xs::AbstractVector{<:AbstractVector}, ys::Vector{Int})
    total = 0.0
    for i in eachindex(ys)
        logits = mlp_forward(model, xs[i])
        lmax = maximum(logits)
        exps = exp.(logits .- lmax)
        probs = exps / sum(exps)
        total -= log(max(probs[ys[i]], 1e-10))
    end
    return total / length(ys)
end

function mlp_predict(model::MLP, xs::AbstractVector{<:AbstractVector})
    return [argmax(mlp_forward(model, x)) for x in xs]
end

function mlp_predict(model::MLP, xs::Matrix)
    return [argmax(mlp_forward(model, xs[:, i])) for i in 1:size(xs, 2)]
end

function mlp_train_step!(model::MLP, xs::AbstractVector{<:AbstractVector}, ys::Vector{Int}, lr::Float64)
    grads = Zygote.gradient(
        (W1, b1, W2, b2, W3, b3) -> begin
            m_tmp = MLP(W1, b1, W2, b2, W3, b3)
            mlp_loss(m_tmp, xs, ys)
        end,
        model.W1, model.b1, model.W2, model.b2, model.W3, model.b3
    )
    model.W1 .-= lr * grads[1]
    model.b1 .-= lr * grads[2]
    model.W2 .-= lr * grads[3]
    model.b2 .-= lr * grads[4]
    model.W3 .-= lr * grads[5]
    model.b3 .-= lr * grads[6]
    return nothing
end
