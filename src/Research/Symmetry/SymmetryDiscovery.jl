# TIER 2 — RESEARCH: Automated symmetry profiling and discovery
"""
    module SymmetryDiscovery

AI representation auditing via provably invariant bispectrum anchors.
Every dataset and every neural network layer has a symmetry signature;
WHERE THEY DIVERGE = where the model learns spurious correlations.

The bispectrum B(x) is PROVABLY shift-invariant, providing an anchor
to measure symmetry structure of: input data, model internals, and
to regularize models to respect data symmetries.

# Key exports
- `symmetry_profile`: Bispectrum-based symmetry fingerprint of a dataset
- `layer_symmetry_profile`: Per-layer symmetry profile of model activations
- `profile_divergence`: Measure how much model diverges from data symmetry
- `detect_spurious_layers`: Find layers exploiting non-data-symmetric features
- `symmetry_regularization`: Loss term to enforce data symmetry in model
- `ood_score_by_symmetry`: OOD detection via symmetry profile divergence

# Usage
```julia
using WDW
profile = SymmetryDiscovery.symmetry_profile(data)
divs = SymmetryDiscovery.detect_spurious_layers(model_fn, data)
```
"""
module SymmetryDiscovery

using LinearAlgebra, Random, Printf, Statistics, Zygote
using ..FFTGroup

export symmetry_profile,
       layer_symmetry_profile,
       profile_divergence,
       detect_spurious_layers,
       symmetry_regularization,
       ood_score_by_symmetry

# =============================================================================
# SYMMETRY PROFILE OF A DATASET OR ACTIVATION SPACE
# =============================================================================
# The symmetry profile of a set of vectors {x_i} measures how the bispectrum
# changes under candidate transformations. It's a VECTOR where each entry
# corresponds to a different transformation probe.
#
# For each probe transformation T_k, we compute:
#   P_k = mean_i ‖B(x_i) - B(T_k(x_i))‖
#
# The full profile P = [P_1, P_2, ..., P_m] is a FINGERPRINT of the
# symmetry structure of the dataset.

function symmetry_profile(xs::AbstractVector{<:AbstractVector};
                          probes::Vector{Matrix{Float64}} = default_probes(length(xs[1])))
    n = length(xs[1])
    ae = CyclicFourierLayer(n; seed=42)
    fill!(ae.A, 1.0 + 0.0im)
    m = length(probes)
    profile = zeros(Float64, m)
    for (k, M) in enumerate(probes)
        vals = [norm(bispec_features(x, ae) - bispec_features(M * x, ae)) for x in xs]
        profile[k] = mean(vals)
    end
    return profile
end

function symmetry_profile(xs::AbstractVector{<:AbstractVector},
                          layer::CyclicFourierLayer, probes::Vector{Matrix{Float64}})
    m = length(probes)
    profile = zeros(Float64, m)
    for (k, M) in enumerate(probes)
        vals = [norm(bispec_features(x, layer) - bispec_features(M * x, layer)) for x in xs]
        profile[k] = mean(vals)
    end
    return profile
end

# =============================================================================
# LAYER-WISE MODEL AUDITING
# =============================================================================
# Given a model function f(x) that returns activations at each layer,
# compute the symmetry profile at each layer.
#
# f: x → [activations_layer1, activations_layer2, ..., predictions]
# Each activation tensor must be a vector (or reshapable to one).

function layer_symmetry_profile(model_fn, xs::AbstractVector{<:AbstractVector};
                                probes::Vector{Matrix{Float64}} = default_probes(length(xs[1])),
                                activation_dims::Vector{Int})

    # Get activations for one sample to infer number of layers
    sample_out = model_fn(xs[1])
    n_layers = length(sample_out)

    n_probes = length(probes)
    profiles = [zeros(Float64, n_probes) for _ in 1:n_layers]

    for i in eachindex(xs)
        x = xs[i]
        acts = model_fn(x)
        for (k, M) in enumerate(probes)
            x_trans = M * x
            acts_trans = model_fn(x_trans)
            for l in 1:n_layers
                a = vec(acts[l])
                a_t = vec(acts_trans[l])
                # Reshape to match expected dimension for bispectrum
                nf = min(length(a), length(a_t))
                a = a[1:nf]; a_t = a_t[1:nf]
                layer_ae = CyclicFourierLayer(nf; seed=l * 100)
                fill!(layer_ae.A, 1.0 + 0.0im)
                b_orig = bispec_features(a, layer_ae)
                b_trans = bispec_features(a_t, layer_ae)
                profiles[l][k] += norm(b_orig - b_trans) / length(xs)
            end
        end
    end
    return profiles
end

# =============================================================================
# PROFILE DIVERGENCE
# =============================================================================
# How much does the model's symmetry profile differ from the data's?
# Higher divergence = more spurious correlations.
#
# Divergence at layer l:
#   D_l = ‖P_data - P_model_layer‖
#
# The "spurious layer" is where D_l is maximized.

function profile_divergence(data_profile::Vector{Float64},
                            model_profile::Vector{Float64})
    return norm(data_profile - model_profile) / sqrt(length(data_profile))
end

function profile_divergence(profiles::Vector{Vector{Float64}},
                            reference::Vector{Float64})
    return [norm(reference - p) / sqrt(length(reference)) for p in profiles]
end

# =============================================================================
# SPURIOUS LAYER DETECTION
# =============================================================================
# Returns layer indices where profile divergence > threshold.
# These layers are exploiting features that DON'T respect data symmetries.

function detect_spurious_layers(model_fn, xs::AbstractVector{<:AbstractVector};
                                threshold::Float64 = 0.1)
    n = length(xs[1])
    probes = default_probes(n)

    # Data profile
    data_prof = symmetry_profile(xs; probes=probes)

    # Model layer profiles
    acts = model_fn(xs[1])
    n_layers = length(acts)
    act_dims = [length(vec(a)) for a in acts]
    layer_profs = layer_symmetry_profile(model_fn, xs; probes=probes, activation_dims=act_dims)

    # Divergence at each layer
    divs = [profile_divergence(lp, data_prof) for lp in layer_profs]

    spurious = findall(d -> d > threshold, divs)

    return (; divergences=divs, spurious_layers=spurious,
            data_profile=data_prof, layer_profiles=layer_profs)
end

# =============================================================================
# SYMMETRY REGULARIZATION
# =============================================================================
# Regularization loss that forces a model's internal representations to
# have the same symmetry profile as the data.
#
# Use during training:
#   L_total = task_loss + λ * symmetry_regularization(model, x, data_profile)

function symmetry_regularization(model_fn, x::Vector{T},
                                 reference_profile::Vector{Float64},
                                 probes::Vector{Matrix{Float64}}) where T
    acts = model_fn(x)
    loss = zero(T)
    for (l, a) in enumerate(acts)
        a_flat = vec(a)
        nf = length(a_flat)
        layer_ae = CyclicFourierLayer(nf; seed=l * 100)
        fill!(layer_ae.A, 1.0 + 0.0im)
        layer_prof = zeros(T, length(probes))
        for (k, M) in enumerate(probes)
            x_trans = M * x
            a_trans = vec(model_fn(x_trans)[l])
            nf2 = min(length(a_flat), length(a_trans))
            b_orig = bispec_features(a_flat[1:nf2], layer_ae)
            b_trans = bispec_features(a_trans[1:nf2], layer_ae)
            layer_prof[k] = norm(b_orig - b_trans)
        end
        loss += norm(reference_profile - layer_prof) / sqrt(length(reference_profile))
    end
    return loss / length(acts)
end

# =============================================================================
# OOD DETECTION VIA SYMMETRY PROFILE
# =============================================================================
# For a single input, compute how its symmetry profile differs from the
# training distribution. Higher score = more likely OOD.

function ood_score_by_symmetry(x::Vector{T}, layer::CyclicFourierLayer{T},
                                reference_profile::Vector{Float64},
                                probes::Vector{Matrix{Float64}}) where T
    n = length(x)
    m = length(probes)
    prof = zeros(T, m)
    for (k, M) in enumerate(probes)
        prof[k] = norm(bispec_features(x, layer) - bispec_features(M * x, layer))
    end
    return norm(reference_profile - prof) / sqrt(m)
end

# =============================================================================
# DEFAULT PROBES
# =============================================================================
# A standard set of probe transformations for C_n signals.
# These probe different frequency bands and patterns.

function default_probes(n::Int)
    n < 2 && return Matrix{Float64}[]
    probes = Matrix{Float64}[]
    # Shifts of various sizes (true C_n symmetries)
    for k in [1, n ÷ 4, n ÷ 2, 3n ÷ 4, n - 1]
        k = max(1, min(k, n - 1))
        M = zeros(Float64, n, n)
        for i in 1:n
            M[i, mod1(i - k, n)] = 1.0
        end
        push!(probes, M)
    end
    # Reflection (checks D_n sensitivity)
    M_ref = zeros(Float64, n, n)
    for i in 1:n
        M_ref[i, mod1(-i + 2, n)] = 1.0
    end
    push!(probes, M_ref)
    # Frequency modulations (phase scrambling)
    for scramble_amt in [0.25, 0.5]
        M = zeros(Float64, n, n)
        for i in 1:n
            s = 1.0 + scramble_amt * sin(2π * (i - 1) / n)
            M[i, i] = s
        end
        M = M / norm(M)
        push!(probes, M)
    end
    return probes
end

# =============================================================================
# BISPECTRUM UTILITIES (re-exported for convenience)
# =============================================================================

function bispec_features(x::Vector{T}, layer::CyclicFourierLayer{T}) where T
    n = layer.n
    n < 2 && return zeros(T, 0)
    x̂ = FFTGroup.myfft(x)
    z = [layer.A[ω] * x̂[ω] for ω in 1:n]
    re = [real(z[ω] * z[2] * conj(z[mod(ω, n) + 1])) for ω in 1:n]
    im = [imag(z[ω] * z[2] * conj(z[mod(ω, n) + 1])) for ω in 1:n]
    return vcat(re, im)
end

# =============================================================================
# DEMO HELPER FUNCTIONS
# =============================================================================

function make_shift_matrix(n::Int, k::Int)
    M = zeros(Float64, n, n)
    for i in 1:n
        M[i, mod1(i - k, n)] = 1.0
    end
    return M
end

function make_reflection_matrix(n::Int)
    M = zeros(Float64, n, n)
    for i in 1:n
        M[i, mod1(-i + 2, n)] = 1.0
    end
    return M
end

function symmetry_probe(x::Vector{T}, M::Matrix{T}, layer::CyclicFourierLayer{T}) where T
    return norm(bispec_features(x, layer) - bispec_features(M * x, layer))
end

function symmetry_breaking_profile(x::Vector{T}, M::Matrix{T}, layer::CyclicFourierLayer{T};
                                   levels::Int=3) where T
    n = length(x)
    prof = zeros(T, levels)
    for l in 1:levels
        M_pow = M^l
        prof[l] = symmetry_probe(x, M_pow, layer)
    end
    return prof
end

function discover_symmetry_transform(xs::Vector{Vector{T}}, n::Int; epochs::Int=200,
                                     lr::Float64=0.05) where T
    K = randn(T, n, n) ./ sqrt(n)
    ae = CyclicFourierLayer(n, Complex{T}.(ones(T, n)), zeros(T, n))
    for ep in 1:epochs
        g = Zygote.gradient(K) do Kmat
            M = Kmat' * Kmat
            M = M / norm(M)
            loss = zero(T)
            for x in xs
                loss += norm(bispec_features(x, ae) - bispec_features(M * x, ae))
            end
            return loss / length(xs)
        end
        K .-= lr .* g[1]
        if ep % 50 == 0
            println("  Discovery epoch $ep")
        end
    end
    M = K' * K
    M = M / norm(M)
    return M, K
end

end  # module
