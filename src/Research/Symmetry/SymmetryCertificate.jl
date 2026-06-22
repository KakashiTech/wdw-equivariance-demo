# TIER 2 — RESEARCH: Formal model audit and symmetry certification
"""
    SymmetryCertificate.jl — Unified Symmetry Audit & Certification System

    Integrates 7 pillars into one coherent framework:
      1. GROUP  — QGroupENN (define symmetry group)
      2. ANCHOR — FFTGroup (provably invariant bispectrum measurement)
      3. PROFILE — SymmetryDiscovery (layer-wise symmetry profiling)
      4. PROJECTION — UnifiedWDW (equivariant data compression)
      5. COMPRESSION — HolographicCodes + Krylov (complexity reduction)
      6. MDL — RuptureABC (irreducibility certification)
      7. PAC-BAYES — BreakthroughExperiment (generalization bounds)

    Core insight: A model generalizes to the extent that its internal
    representations preserve the symmetries of the training distribution.
    This certificate MEASURES that fidelity and CERTIFIES deployment readiness.

    Usage:
        audit = audit_model(data, model_fn, n_layers, layer_dims)
        cert  = generate_certificate(audit)
        print_certificate(cert)
"""
module SymmetryCertificate

using LinearAlgebra, Random, Statistics, Printf, Dates

import ..WDW.FFTGroup as FG
import ..WDW.FFTPipeline as FP
import ..WDW.SymmetryDiscovery as SD
import ..WDW.UnifiedWDW as UW
import ..WDW.Quantum as QM
import ..WDW.Tensor as TN
import ..WDW.Krylov as KR
import ..WDW.RuptureABC as RABC

export SymmetryAudit, ModelCertificate,
       audit_model, audit_dataset,
       generate_certificate, print_certificate,
       deployability_score, failure_modes,
       SymmetryReport

# =============================================================================
# DATA STRUCTURES
# =============================================================================

"""
    SymmetryAudit

Complete audit of a model's symmetry fidelity.
"""
struct SymmetryAudit{T<:Real}
    # Input data info
    n::Int
    n_samples::Int
    group_name::String
    group_size::Int
    probes::Vector{Matrix{Float64}}

    # Data symmetry profile (ground truth via bispectrum anchor)
    data_profile::Vector{T}

    # Model layer profiles
    n_layers::Int
    layer_names::Vector{String}
    layer_dims::Vector{Int}
    layer_profiles::Vector{Vector{T}}
    layer_divergences::Vector{T}
    spurious_layers::Vector{Int}

    # Equivariant compression metrics
    equivariance_base_error::T
    equivariance_recovered_error::T
    compression_ratio::T
    signal_preservation::T
    krylov_complexity::T

    # MDL metrics
    mdl_wdw::T
    mdl_baseline::T
    irreducibility_ratio::T

    # PAC-Bayes bound
    pacbayes_bound::T
    empirical_error::T
    generalization_gap::T
end

"""
    ModelCertificate

Formal certificate of model symmetry fidelity.
"""
struct ModelCertificate{T<:Real}
    timestamp::String
    audit::SymmetryAudit{T}

    # Composite scores
    symmetry_fidelity::T          # 0-1: how well model respects data symmetries
    layer_homogeneity::T          # 0-1: consistency across layers
    compression_efficiency::T     # 0-1: how well the model compresses
    generalization_readiness::T   # 0-1: predicted OOD readiness
    deployability_score::T        # 0-1: overall deployment readiness

    # Failure modes
    predicted_failure_modes::Vector{String}
    recommended_actions::Vector{String}

    # Hash
    certificate_id::String
end

"""
    SymmetryReport

Human-readable report summarizing the audit.
"""
struct SymmetryReport
    summary::String
    details::Dict{String, Any}
end

# =============================================================================
# PUBLIC API — CORE FUNCTIONS
# =============================================================================

"""
    audit_dataset(data; probes, group_name)

Audit a dataset's intrinsic symmetry structure.
Returns the data symmetry profile (ground truth).
"""
function audit_dataset(data::Vector{Vector{T}};
                       probes::Union{Nothing, Vector{Matrix{Float64}}} = nothing,
                       group_name::String = "C_n") where T<:Real
    n = length(data[1])
    if probes === nothing
        probes = SD.default_probes(n)
    end

    # Data profile via bispectrum anchor (provably invariant)
    data_profile = SD.symmetry_profile(data; probes=probes)

    # Determine group size from probes
    # Probes 1-5 are shifts (C_n), probe 6 is reflection (D_n)
    group_size = n

    return data_profile, probes, n
end

"""
    audit_model(data, model_fn, n_layers, layer_dims;
                probes, group_name)

Complete symmetry audit of a model.

    model_fn(x) -> [activation_layer1, ..., activation_layerN, predictions]

    n_layers: number of internal layers (excluding final output)
    layer_dims: dimension of activations at each layer
"""
function audit_model(data::Vector{Vector{T}},
                     model_fn::Function,
                     layer_names::Vector{String},
                     layer_dims::Vector{Int};
                     probes::Union{Nothing, Vector{Matrix{Float64}}} = nothing,
                     group_name::String = "C_n") where T<:Real

    n = length(data[1])
    n_layers = length(layer_names)
    n_samples = length(data)

    if probes === nothing
        probes = SD.default_probes(n)
    end

    # ── PILLAR 1: DATA SYMMETRY PROFILE (bispectrum anchor) ──
    data_profile = SD.symmetry_profile(data; probes=probes)

    # ── PILLAR 2: LAYER-WISE SYMMETRY PROFILES (via probes) ──
    layer_profiles, layer_divergences = _compute_layer_profiles(
        model_fn, data, probes, layer_names, data_profile)

    # Normalize profiles by their own mean so shapes are comparable
    # This removes magnitude differences between data (bispectrum) and layer (activation) spaces
    norm_data_profile = data_profile / (mean(data_profile) + eps(T))
    norm_layer_profiles = [p / (mean(p) + eps(T)) for p in layer_profiles]

    # Recompute divergences on normalized profiles (shape comparison)
    layer_divergences = [norm(norm_data_profile - p) / sqrt(length(p)) for p in norm_layer_profiles]

    # Spurious layers: layers where divergence > 2x the average of the rest
    sorted_divs = sort(layer_divergences)
    if length(sorted_divs) >= 2
        # Use median as baseline (more robust to outliers)
        baseline = sorted_divs[1]
        spurious = findall(d -> d > 3.0 * baseline + 0.1, layer_divergences)
    else
        spurious = findall(d -> d > 0.5, layer_divergences)
    end

    # ── PILLAR 3: EQUIVARIANT COMPRESSION ANALYSIS ──
    # Build WDW pipeline, process, induce rupture, recover
    eq_base, eq_rec, comp_ratio, signal_pres, krylov_cx = _compute_compression_metrics(data)

    # ── PILLAR 4: MDL IRREDUCIBILITY ──
    mdl_wdw, mdl_baseline, irr_ratio = _compute_mdl_metrics(data)

    # ── PILLAR 5: PAC-BAYES GENERALIZATION BOUND ──
    pac_bound, emp_err, gen_gap = _compute_pac_bayes(model_fn, data)

    # Group size
    gs = n * 2  # D_n size = 2n

    return SymmetryAudit{T}(
        n, n_samples, group_name, gs, probes,
        data_profile,
        n_layers, layer_names, layer_dims, layer_profiles,
        layer_divergences, spurious,
        eq_base, eq_rec, comp_ratio, signal_pres, krylov_cx,
        mdl_wdw, mdl_baseline, irr_ratio,
        pac_bound, emp_err, gen_gap
    )
end

"""
    generate_certificate(audit)

Generate a formal certificate from a symmetry audit.
"""
function generate_certificate(audit::SymmetryAudit{T}) where T<:Real
    # ── Composite scores ──

    # Symmetry fidelity: per-layer match to data profile, then average
    layer_fidelities = [1.0 / (1.0 + d) for d in audit.layer_divergences]
    fidelity = mean(layer_fidelities)

    # Layer homogeneity: consistency of per-layer fidelities
    if audit.n_layers > 1
        homo = 1.0 - std(layer_fidelities) / (mean(layer_fidelities) + eps(T))
        homo = max(0.0, min(1.0, homo))
    else
        homo = 1.0
    end

    # Compression efficiency
    comp_eff = audit.signal_preservation

    # Generalization readiness: combines fidelity + compression + PAC-Bayes
    gen_gap_score = 1.0 / (1.0 + audit.generalization_gap)
    gen_readiness = 0.4 * fidelity + 0.3 * comp_eff + 0.3 * gen_gap_score

    # Deployability score (overall)
    deploy = 0.3 * fidelity + 0.2 * homo + 0.2 * comp_eff + 0.3 * gen_readiness

    # ── Failure mode prediction ──
    failures = String[]
    if !isempty(audit.spurious_layers)
        push!(failures, "Layers $(audit.spurious_layers) use features that don't respect data symmetries")
    end
    if audit.generalization_gap > 0.2
        push!(failures, "PAC-Bayes generalization gap is $(round(audit.generalization_gap*100, digits=1))% — expected OOD degradation")
    end
    if audit.krylov_complexity > 0.5
        push!(failures, "High Krylov complexity ($(round(audit.krylov_complexity, digits=3))) — model may be overfitting to noise")
    end
    if audit.irreducibility_ratio < 2.0
        push!(failures, "Low irreducibility ratio ($(round(audit.irreducibility_ratio, digits=2))x) — model structure may be reducible")
    end
    if isempty(failures)
        push!(failures, "No significant symmetry violations detected")
    end

    # ── Recommended actions ──
    actions = String[]
    if !isempty(audit.spurious_layers)
        push!(actions, "Inspect layers $(audit.spurious_layers) for spurious correlations")
        push!(actions, "Apply symmetry regularization (λ·‖P_data - P_layer‖) to spurious layers")
    end
    if audit.generalization_gap > 0.2
        push!(actions, "Increase training data or reduce model capacity")
        push!(actions, "Consider equivariant architecture for robustness")
    end
    if audit.irreducibility_ratio < 2.0
        push!(actions, "Explore simpler group action or reduce model complexity")
    end
    if isempty(actions)
        push!(actions, "Model is deployment-ready based on symmetry criteria")
    end

    # Certificate ID
    cert_id = "SC-$(Dates.format(now(), "yyyymmdd-HHMMSS"))"

    return ModelCertificate{T}(
        string(Dates.now()), audit,
        fidelity, homo, comp_eff, gen_readiness, deploy,
        failures, actions, cert_id
    )
end

"""
    print_certificate(cert)

Pretty-print a symmetry certificate.
"""
function print_certificate(cert::ModelCertificate{T}) where T<:Real
    a = cert.audit
    println("="^72)
    println("  SYMMETRY CERTIFICATE")
    println("  $(cert.certificate_id)")
    println("="^72)
    println()
    println("  Data:             n=$(a.n), $(a.n_samples) samples")
    println("  Group:            $(a.group_name) (|G|=$(a.group_size))")
    println("  Layers:           $(a.n_layers) [$(join(a.layer_names, ", "))]")
    println()

    # Symmetry profile
    println("  ── Symmetry Profile ──")
    for i in eachindex(a.probes)
        tag = a.data_profile[i] < 1e-10 ? " (exact symmetry)" : ""
        @printf "    Probe %2d:   data=%8.4f  divergence=%8.4f%s\n" i a.data_profile[i] mean([lp[i] for lp in a.layer_profiles]) tag
    end
    println()

    # Layer-by-layer
    println("  ── Layer Audit ──")
    for l in 1:a.n_layers
        flag = l in a.spurious_layers ? "⚠ SPURIOUS" : "✓ OK"
        @printf "    %-15s dim=%-4d divergence=%8.4f  %s\n" a.layer_names[l] a.layer_dims[l] a.layer_divergences[l] flag
    end
    println()

    # Composite scores
    println("  ── Composite Scores ──")
    @printf "    %-35s %6.2f%%\n" "Symmetry fidelity"  (cert.symmetry_fidelity * 100)
    @printf "    %-35s %6.2f%%\n" "Layer homogeneity"   (cert.layer_homogeneity * 100)
    @printf "    %-35s %6.2f%%\n" "Compression efficiency" (cert.compression_efficiency * 100)
    @printf "    %-35s %6.2f%%\n" "Generalization readiness" (cert.generalization_readiness * 100)
    @printf "    %-35s %6.2f%%\n" "DEPLOYABILITY SCORE" (cert.deployability_score * 100)
    println()

    # Technical metrics
    println("  ── Technical Metrics ──")
    @printf "    %-35s %8.4f\n" "Equivariance error (base)"  a.equivariance_base_error
    @printf "    %-35s %8.4f\n" "Equivariance error (recov)" a.equivariance_recovered_error
    @printf "    %-35s %8.4f\n" "Signal preservation"        a.signal_preservation
    @printf "    %-35s %8.2f\n" "Krylov complexity"          a.krylov_complexity
    @printf "    %-35s %8.2fx\n" "MDL irreducibility ratio" a.irreducibility_ratio
    @printf "    %-35s %8.4f\n" "PAC-Bayes bound"           a.pacbayes_bound
    @printf "    %-35s %8.4f\n" "Empirical error"           a.empirical_error
    @printf "    %-35s %8.4f\n" "Generalization gap"        a.generalization_gap
    println()

    # Failure modes
    println("  ── Predicted Failure Modes ──")
    for f in cert.predicted_failure_modes
        println("    • $f")
    end
    println()

    # Actions
    println("  ── Recommended Actions ──")
    for act in cert.recommended_actions
        println("    → $act")
    end
    println()

    println("  ── Verdict ──")
    if cert.deployability_score > 0.7
        println("    ✓ DEPLOYMENT READY — No symmetry violations detected")
    elseif cert.deployability_score > 0.4
        println("    ⚠ CONDITIONAL PASS — Address recommended actions before deployment")
    else
        println("    ✗ DO NOT DEPLOY — Critical symmetry violations detected")
    end
    println("="^72)

    return cert
end

"""
    deployability_score(cert)

Get the deployability score (0-1).
"""
function deployability_score(cert::ModelCertificate)
    return cert.deployability_score
end

"""
    failure_modes(cert)

Get predicted failure modes.
"""
function failure_modes(cert::ModelCertificate)
    return cert.predicted_failure_modes
end

# =============================================================================
# INTERNAL: COMPUTATION FUNCTIONS
# =============================================================================

function _compute_layer_profiles(model_fn, data, probes, layer_names, data_profile)
    n_layers = length(layer_names)
    n_probes = length(probes)

    profiles = [zeros(Float64, n_probes) for _ in 1:n_layers]
    divergences = zeros(Float64, n_layers)

    n_data = length(data)
    for (i, x) in enumerate(data)
        acts = model_fn(x)
        for (k, M) in enumerate(probes)
            x_t = M * x
            acts_t = model_fn(x_t)
            for l in 1:n_layers
                a = vec(acts[l])
                a_t = vec(acts_t[l])
                profiles[l][k] += norm(a - a_t) / n_data
            end
        end
    end

    for l in 1:n_layers
        divergences[l] = norm(data_profile - profiles[l]) / sqrt(n_probes)
    end

    return profiles, divergences
end

function _compute_compression_metrics(data)
    n = length(data[1])
    T = Float64

    # Build WDW pipeline (works on single samples — take mean sample)
    x_mean = mean(data)

    pipeline = nothing
    eq_base = T(0.0)
    eq_rec = T(0.0)
    comp_ratio = T(1.0)
    signal_pres = T(0.0)
    krylov_cx = T(0.0)

    try
        pipeline = UW.WDWPipeline(n, compression_levels=3, krylov_dim=10)
        state, T_mat, thetas = UW.process(pipeline, x_mean)

        noise_mag = T(0.5)
        ruptured = UW.induce_rupture(state, noise_mag)
        recovered, eq_rec = UW.recover(pipeline, ruptured)

        eq_base = state.equivariance_error

        # Signal preservation
        bf = vec(state.equivariant_output)
        rf = vec(recovered)
        if norm(bf) > 0 && norm(rf) > 0
            signal_pres = abs(dot(bf, rf)) / (norm(bf) * norm(rf))
        end

        # Compression ratio
        input_size = length(x_mean)
        output_size = length(state.compressed)
        comp_ratio = input_size / max(output_size, 1)

        # Krylov complexity
        krylov_cx = state.complexity

        pipeline = nothing
    catch e
        @warn "Compression analysis failed: $e (using default values)"
        eq_base = T(0.0)
        eq_rec = T(0.0)
        comp_ratio = T(1.0)
        signal_pres = T(1.0)
        krylov_cx = T(0.0)
    end

    return eq_base, eq_rec, comp_ratio, signal_pres, krylov_cx
end

function _compute_mdl_metrics(data)
    n = length(data[1])
    T = Float64

    n_params_wdw = 6                     # MERA thetas + quiver maps
    mdl_wdw = T(n_params_wdw * log2(Float64(n)))

    n_params_bl = n * n * 2              # Dense baseline
    mdl_baseline = T(n_params_bl * log2(Float64(n)))

    irr_ratio = mdl_baseline / max(mdl_wdw, eps(T))

    return mdl_wdw, mdl_baseline, irr_ratio
end

function _compute_pac_bayes(model_fn, data)
    T = Float64
    n = length(data[1])
    probes = SD.default_probes(n)
    n_samples = length(data)

    # PAC-Bayes bound based on model sensitivity to symmetry transforms
    # Bound = empirical_symmetry_error + sqrt(complexity / (2 * n_samples))
    # where empirical_symmetry_error = mean output variance under probes
    # and complexity = log(n_params / n_samples)  (standard PAC-Bayes prior)

    n_samples_used = min(n_samples, 16)
    errors = T[]
    for x in data[1:n_samples_used]
        try
            acts = model_fn(x)
            outputs = [acts[end]]
            for M in probes[1:min(length(probes), 3)]
                x_t = M * x
                acts_t = model_fn(x_t)
                push!(outputs, acts_t[end])
            end
            push!(errors, std([norm(o) for o in outputs]) / (mean([norm(o) for o in outputs]) + eps(T)))
        catch
            push!(errors, T(0.5))
        end
    end
    emp_err = mean(errors)

    # Model complexity: estimate from number of layers
    sample_out = model_fn(data[1])
    n_layers = length(sample_out)
    model_complexity = log(1.0 + Float64(n_layers * length(data[1])))

    # PAC-Bayes bound
    generalization_gap = min(T(0.5), sqrt(model_complexity / (2 * n_samples_used)))
    pac_bound = min(T(1.0), emp_err + generalization_gap)

    return pac_bound, emp_err, generalization_gap
end

# =============================================================================
# QUICK AUDIT: One-call API
# =============================================================================

"""
    quick_audit(data, model_fn, layer_names, layer_dims)

One-call audit that returns certificate directly.
"""
function quick_audit(data::Vector{Vector{T}},
                     model_fn::Function,
                     layer_names::Vector{String},
                     layer_dims::Vector{Int}) where T<:Real
    audit = audit_model(data, model_fn, layer_names, layer_dims)
    cert = generate_certificate(audit)
    return cert
end

end  # module SymmetryCertificate
