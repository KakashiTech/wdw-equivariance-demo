# TIER 3 — EXPERIMENTAL: Unified pipeline state machine (sheaf → quiver → MERA → Krylov)
module UnifiedWDW

using LinearAlgebra
using Random
using Printf
using Statistics
using ..WDW.Knowledge: TopSpace, Partial, glue_partial, int, cl
using ..WDW.Sheaves: ConstantSheaf, glue, sections_to_partials
using ..WDW.Algebra: Quiver, QuiverLayer, apply_quiver_walks, adjacency_matrix
using ..WDW.Quantum: FinitePermGroup, dihedral_group, project_equivariant, act, is_equivariant
using ..WDW.Tensor: haar_forward, haar_inverse, param_mera_reconstruct_truncated, param_multiscale_error, optimize_thetas, mera_compress, mera_reconstruct
using ..WDW.Krylov: lanczos_tridiagonal, krylov_spread_complexity

export WDWPipeline, UnifiedState, process, induce_rupture, recover, measure_invariants,
       RuptureMetrics, run_full_pipeline_test, cumulative_statistics, s_score

"""
    UnifiedState

Estado integrado que fluye a través de todas las fases del sistema WDW.
Contiene representaciones en múltiples niveles: cumulative stats, quiver, equivariante, comprimida.
"""
struct UnifiedState{T<:Real}
    raw_data::Vector{T}
    sheaf_sections::Vector{Partial{Int,T}}
    quiver_features::Matrix{T}
    equivariant_output::Matrix{T}
    compressed::Vector{T}
    complexity::T
    equivariance_error::T
end

"""
    WDWPipeline

Pipeline unificado que conecta todas las fases de WDW:
Cumulative Statistics → Quivers → Q-G-ENN → MERA → Krylov
"""
struct WDWPipeline{T<:Real}
    n::Int
    sheaf_space::TopSpace{Int}
    quiver::Quiver
    group::FinitePermGroup
    compression_levels::Int
    krylov_dim::Int
    soft_lambda::T          # 0 = hard projection (default), >0 = soft regularization
    cache::Dict{String, Any}
    
    function WDWPipeline(n::Int; 
                         compression_levels::Int=3,
                         krylov_dim::Int=10,
                         soft_lambda::Real=0.0)
        @assert iseven(n) "n debe ser par para MERA"
        
        # Construir espacio topológico para sheaves
        X = collect(1:n)
        opens = [collect(1:i) for i in 1:n]  # Cadena de abiertos crecientes
        space = TopSpace{Int}(X, opens)
        
        # Construir quiver en anillo (estructura cíclica)
        nodes = collect(1:n)
        edges = [(i, mod1(i+1, n)) for i in 1:n]
        # Agregar conexiones bidireccionales
        for i in 1:n
            push!(edges, (i, mod1(i-1, n)))
        end
        edges = unique(edges)
        quiver = Quiver(nodes, edges)
        
        # Grupo dihedral para equivariancia
        group = dihedral_group(n)
        
        new{Float64}(n, space, quiver, group, compression_levels, krylov_dim, Float64(soft_lambda), Dict())
    end
end

"""
    cumulative_statistics(pipeline, data)

Fase 1: Build sections of a CONSTANT SHEAF over a chain of open sets.

Each open U_i = {1, ..., i} is assigned the cumulative mean `(sum_{j≤i} data[j]) / i`.
This defines a constant sheaf: the restriction map res_{U_k → U_j} is the identity
(res(v) = v), because the sheaf value depends only on the open set, not on the
domain size. Restriction maps compose trivially:

    res_{U_j→U_i}(res_{U_k→U_j}(v_k)) = v_k = res_{U_k→U_i}(v_k)   ✓

The cumulative mean for any open is recovered directly as `section.value`.
"""
function cumulative_statistics(pipeline::WDWPipeline{T}, data::Vector{T}) where T
    n = pipeline.n
    @assert length(data) == n

    sections = Partial{Int,T}[]
    cum_sum = zero(T)
    for (i, U) in enumerate(pipeline.sheaf_space.opens)
        cum_sum += data[i]
        push!(sections, Partial{Int,T}(U, cum_sum / i))
    end

    return sections
end

"""
    quiver_propagation(pipeline, sections)

Fase 2: Propagar a través del quiver usando las cumulative statistics como features iniciales.
Edge maps are identity (no learned message passing).
"""
function quiver_propagation(pipeline::WDWPipeline{T}, sections::Vector{Partial{Int,T}}) where T
    n = pipeline.n
    # Inicializar features de nodos desde sheaf
    X = zeros(T, n, 1)
    for i in 1:n
        X[i, 1] = sections[min(i, length(sections))].value
    end
    
    # Crear capa de quiver con mapas de borde
    edge_maps = Dict{Tuple{Int,Int}, Matrix{T}}()
    for e in pipeline.quiver.edges
        edge_maps[e] = Matrix{T}(I, 1, 1)  # Mapas identidad simples
    end
    layer = QuiverLayer(pipeline.quiver, 1, 1, edge_maps=edge_maps)
    
    # Propagar con walks de profundidad 2
    Y = apply_quiver_walks(layer, X, depth=2)
    return Y
end

"""
    equivariant_projection(pipeline, features)

Fase 2 (Q-G-ENN): Decompose features into Dₙ-invariant + anti-invariant parts.

For 1D features on a transitive Dₙ action, the Reynolds operator alone collapses
to a constant (destroying spatial information). This function preserves ALL info
by returning stacked [inv, anti] columns:
  - First n_cols columns: Dₙ-invariant part (Reynolds projector)
  - Next n_cols columns: anti-invariant residual (spatial structure)

**Modo hard (soft_lambda=0, default):** Returns hcat(inv, anti), preserving
every bit of the original signal while separating equivariant structure.
**Modo soft (soft_lambda>0):** Interpolates, returns same dimensions as input.
"""
function equivariant_projection(pipeline::WDWPipeline{T}, features::Matrix{T}) where T
    n = pipeline.n
    n_rows, n_cols = size(features)
    G = pipeline.group
    
    # Reynolds operator per column: projection onto Dₙ-invariant subspace
    inv_part = zeros(T, n_rows, n_cols)
    for col in 1:n_cols
        col_data = features[:, col]
        proj_col = zeros(T, n_rows)
        for p in G.perms
            permuted = [col_data[p[i]] for i in 1:n_rows]
            proj_col .+= permuted
        end
        proj_col ./= length(G.perms)
        inv_part[:, col] = proj_col
    end
    
    # Anti-invariant residual: captures ALL spatial structure destroyed by averaging
    anti_part = features - inv_part
    eq_err = norm(anti_part) / (norm(features) + eps(T))
    
    if pipeline.soft_lambda > 0
        # Soft: interpolación entre original y proyectado
        W_out = (1 - pipeline.soft_lambda) * features + pipeline.soft_lambda * inv_part
        eq_err = equivariance_error(pipeline, W_out)
        return W_out, eq_err
    else
        # Hard: return [invariant | anti-invariant], preserving ALL information
        return hcat(inv_part, anti_part), eq_err
    end
end

"""
    equivariance_error(pipeline, features)

Measure INVARIANCE error under the dihedral group Dₙ.

For each column, computes the average deviation ||f∘g - f|| / √n over all 
g ∈ Dₙ. A value of 0 means the feature is Dₙ-invariant (constant on orbits).

For the expanded [inv | anti] representation produced by 
`equivariant_projection`, the first n_cols columns should have error ≈ 0
(invariant part) and the anti-invariant columns will have non-zero error.

Note: This checks INVARIANCE only, not full equivariance (which would require
f(g·x) vs g·f(x) with separately computed terms). For 1D features on a
transitive action, invariance error is the anti-invariant norm fraction.
"""
function equivariance_error(pipeline::WDWPipeline{T}, features::Matrix{T}) where T
    n = pipeline.n
    n_rows, n_cols = size(features)
    G = pipeline.group
    acc = zero(T)
    count = 0

    for col in 1:n_cols
        f_col = features[:, col]
        for p in G.perms
            # Check: is the feature value the same at position i and position p[i]?
            # i.e., is the feature field G-invariant?
            delta = norm(f_col[p] - f_col) / sqrt(n)
            acc += delta
            count += 1
        end
    end

    return count > 0 ? acc / count : zero(T)
end

"""
    mera_compression(pipeline, features; ratio=2)

Fase 3: Comprimir usando MERA (Haar wavelet) con truncamiento real.
El vector retornado tiene longitud n pero solo n/ratio grados de libertad
(las componentes de alta frecuencia se ponen a cero).
"""
function mera_compression(pipeline::WDWPipeline{T}, features::Matrix{T}; ratio::Int=2) where T
    n = pipeline.n
    x = vec(features)
    
    if length(x) < n
        x = [x; zeros(T, n - length(x))]
    elseif length(x) > n
        x = x[1:n]
    end
    
    # Optimizar parámetros (lightweight)
    thetas, error = optimize_thetas(x, pipeline.compression_levels,
                                     pipeline.compression_levels - 1,
                                     iters=20, step=0.5)
    
    # Actual compression: keep only n/ratio coarse coefficients
    compressed_coeffs, recon = mera_compress(x; ratio=ratio)
    
    return recon, thetas, error
end

"""
    build_hamiltonian(compressed_data)

Build a sparse graph Laplacian from the correlation matrix (RBF kernel) of compressed data.
This creates a proper Hamiltonian with a non-trivial spectrum for Lanczos iteration.
"""
function build_hamiltonian(compressed_data::Vector{T}) where T
    n = length(compressed_data)
    C = zeros(T, n, n)
    for i in 1:n
        for j in 1:n
            C[i,j] = abs(compressed_data[i] - compressed_data[j])
        end
    end
    C = exp.(-C .^ 2 / (2 * var(compressed_data) + eps(T)))
    D = Diagonal(vec(sum(C, dims=2)))
    L = D - C
    H = Symmetric(L + T(0.01) * I)
    return H
end

"""
    krylov_analysis(pipeline, compressed_data)

Fase 3/4: Análisis de complejidad via Krylov (Lanczos).
"""
function krylov_analysis(pipeline::WDWPipeline{T}, compressed_data::Vector{T}) where T
    m = length(compressed_data)
    krylov_m = min(pipeline.krylov_dim, m)
    
    # Build graph Laplacian Hamiltonian from data
    H = build_hamiltonian(compressed_data)
    
    # Random initial vector + normalize
    rng = MersenneTwister(42)
    v0 = randn(rng, T, m)
    v0 /= norm(v0)
    
    # Lanczos tridiagonalización
    T_mat, _, _ = lanczos_tridiagonal(H, v0, krylov_m)
    
    # Métrica de complejidad (spread complexity)
    complexity = krylov_spread_complexity(T_mat)
    
    return T_mat, complexity, diag(T_mat), diag(T_mat, 1)
end

"""
    process(pipeline, input_data)

Ejecutar el pipeline completo: Cumulative Statistics → Quivers → Q-G-ENN → MERA → Krylov
"""
function process(pipeline::WDWPipeline{T}, input_data::Vector{T}) where T
    n = pipeline.n
    @assert length(input_data) == n
    
    # Fase 1: Cumulative statistics
    sections = cumulative_statistics(pipeline, input_data)
    
    # Fase 2: Quiver propagation
    quiver_features = quiver_propagation(pipeline, sections)
    
    # Fase 2: Q-G-ENN projection
    equiv_features, eq_err = equivariant_projection(pipeline, quiver_features)
    
    # Fase 3: MERA compression
    compressed, thetas, compress_err = mera_compression(pipeline, equiv_features)
    
    # Fase 3/4: Krylov analysis
    T_mat, complexity, alpha, beta = krylov_analysis(pipeline, compressed)
    
    # Construir estado unificado
    state = UnifiedState{T}(
        input_data,
        sections,
        quiver_features,
        equiv_features,
        compressed,
        complexity,
        eq_err
    )
    
    return state, T_mat, thetas
end

"""
    induce_rupture(state, noise_mag)

Inducir ruptura (romper equivariancia) añadiendo ruido estructurado.
"""
function induce_rupture(state::UnifiedState{T}, noise_mag::Real) where T
    n = length(state.raw_data)
    # Ruido no equivariante: matriz diagonal que no respeta simetría
    noise = noise_mag * Diagonal(collect(1:n)) * randn(T, n, size(state.equivariant_output, 2))
    ruptured_features = state.equivariant_output + noise
    
    return ruptured_features
end

"""
    recover(pipeline, ruptured_features)

Recuperar equivariancia via proyección.
Returns only the Dₙ-invariant part (first half of columns) to prevent
dimension explosion on repeated recovery cycles, along with the invariance
error of the recovered features.
"""
function recover(pipeline::WDWPipeline{T}, ruptured_features::Matrix{T}) where T
    n_cols = size(ruptured_features, 2)
    W_full, _ = equivariant_projection(pipeline, ruptured_features)
    recovered = W_full[:, 1:n_cols]
    eq_err = equivariance_error(pipeline, recovered)
    return recovered, eq_err
end

"""
    RuptureMetrics

Métricas de un ciclo ruptura-recuperación.

**Nuevo (outcome-based):**
- `signal_preservation`: correlación entre features pre-ruptura y post-recuperación (0-1)
- `recovery_ratio`: ahora basado en signal_preservation, no en error inverso
- `task_accuracy_*`: métricas de tarea downstream (si se proveen)
"""
struct RuptureMetrics{T<:Real}
    equivariance_base::T
    equivariance_ruptured::T
    equivariance_recovered::T
    complexity_base::T
    recovery_ratio::T
    signal_preservation::T      # NUEVO: correlación signal pre/post recovery
    task_accuracy_base::T         # NUEVO: accuracy tarea antes de ruptura
    task_accuracy_ruptured::T   # NUEVO: accuracy tarea después de ruptura
    task_accuracy_recovered::T    # NUEVO: accuracy tarea después de recovery
    success::Bool
end

"""
    s_score(eq_base, eq_recovered, signal_preservation; n_bootstrap=100, task_recovery=nothing)

Computes an ad-hoc composite metric from pipeline results.

**Warning**: This is an experimental heuristic, not a theoretically grounded metric.
The sci (stability) component uses only 2 samples and should be interpreted qualitatively.

Returns (score, ci_low, ci_high) where ci are bootstrap 5th/95th percentiles.
"""
function s_score(eq_base::T, eq_recovered::T, signal_preservation::T;
                 n_bootstrap::Int=100, task_recovery::Union{T,Nothing}=nothing) where T
    # sci: relative equivariance improvement (0=no change, 1=perfect recovery)
    sci = max(zero(T), min(one(T), (eq_base - eq_recovered) / (max(eq_base, eps(T)))))
    residual_score = signal_preservation
    equivariance_score = 1.0 / (1.0 + eq_recovered)

    if task_recovery !== nothing && task_recovery > zero(T)
        S = (sci + residual_score + equivariance_score + task_recovery) / 4.0
        components = [sci, residual_score, equivariance_score, task_recovery]
    else
        S = (sci + residual_score + equivariance_score) / 3.0
        components = [sci, residual_score, equivariance_score]
    end

    # Bootstrap confidence intervals
    n_comp = length(components)
    boot_scores = zeros(T, n_bootstrap)
    for b in 1:n_bootstrap
        idxs = rand(1:n_comp, n_comp)
        boot_scores[b] = mean(components[idxs])
    end
    sort!(boot_scores)
    ci_low = boot_scores[clamp(Int(round(0.05 * n_bootstrap)), 1, n_bootstrap)]
    ci_high = boot_scores[clamp(Int(round(0.95 * n_bootstrap)), 1, n_bootstrap)]

    return S, ci_low, ci_high
end

"""
    measure_invariants(pipeline, state, ruptured, recovered; noise_mag=1.0)

Medir invariantes y construir métricas compuestas (S-score).

**Nuevo**: recovery_ratio se basa en signal_preservation, no en división por epsilon.
**Nuevo**: S-score usa correlación de signal, no solo error de equivariancia.
"""
function measure_invariants(pipeline::WDWPipeline{T}, 
                            state::UnifiedState{T},
                            ruptured::Matrix{T},
                            recovered::Matrix{T};
                            noise_mag::Real=1.0,
                            task_evaluator::Union{Function, Nothing}=nothing) where T
    # Error de equivariancia en cada fase
    eq_base = state.equivariance_error
    eq_ruptured = equivariance_error(pipeline, ruptured)
    eq_recovered = equivariance_error(pipeline, recovered)
    
    # Signal preservation: correlación entre features base y recuperadas
    # Mide cuánta señal útil se preserva, no cuán equivariante es el resultado
    base_features = vec(state.equivariant_output)
    rec_features = vec(recovered)
    if norm(base_features) > 0 && norm(rec_features) > 0
        signal_preservation = abs(dot(base_features, rec_features)) / (norm(base_features) * norm(rec_features))
    else
        signal_preservation = zero(T)
    end
    
    # Recovery ratio REVISADO: basado en signal preservation, no error inverso
    # Valor 1.0 = recuperación perfecta, 0.0 = señal completamente perdida
    recovery_ratio = signal_preservation
    
    # Task accuracy (si se provee un evaluador)
    acc_base = task_evaluator !== nothing ? task_evaluator(state.equivariant_output) : zero(T)
    acc_ruptured = task_evaluator !== nothing ? task_evaluator(ruptured) : zero(T)
    acc_recovered = task_evaluator !== nothing ? task_evaluator(recovered) : zero(T)
    
    # Complejidad base
    complexity_base = state.complexity
    
    # S-score using standalone function
    task_recovery_val = (task_evaluator !== nothing && acc_base > zero(T)) ? acc_recovered / max(acc_base, eps(T)) : nothing
    S_score, ci_low, ci_high = s_score(eq_base, eq_recovered, signal_preservation,
                                        task_recovery=task_recovery_val)
    
    # Success: recovery_ratio (signal preservation) > 0.5, no tautología de epsilon
    success = signal_preservation > 0.5 && eq_recovered <= 0.5 * eq_ruptured
    
    return RuptureMetrics{T}(
        eq_base, eq_ruptured, eq_recovered, complexity_base, 
        recovery_ratio, signal_preservation, acc_base, acc_ruptured, acc_recovered,
        success
    ), S_score, ci_low, ci_high
end

"""
    run_full_pipeline_test(n::Int=32; noise_mag=1.0, seed=0)

Ejecutar prueba completa del pipeline unificado con ciclo ruptura-recuperación.
Retorna métricas detalladas.
"""
function run_full_pipeline_test(n::Int=32; noise_mag::Real=1.0, seed::Int=0)
    Random.seed!(seed)
    T = Float64
    
    println("="^60)
    println("WDW Unified Pipeline Test")
    println("="^60)
    println("n=$n, noise_mag=$noise_mag, seed=$seed")
    println()
    
    # Construir pipeline
    pipeline = WDWPipeline(n, compression_levels=3, krylov_dim=10)
    println("Pipeline creado:")
    println("  - Grupo: dihedral, |G|=$(length(pipeline.group.perms))")
    println("  - Quivers: $(length(pipeline.quiver.edges)) aristas")
    println("  - MERA levels: $(pipeline.compression_levels)")
    println()
    
    # Datos de entrada
    input_data = randn(T, n)
    println("Input: n=$n, norm=$(round(norm(input_data), digits=3))")
    
    # Proceso completo
    println("\n--- Ejecutando pipeline ---")
    state, T_mat, thetas = process(pipeline, input_data)
    
    println("Fase 1 (Cumulative): $(length(state.sheaf_sections)) secciones")
    println("Fase 2 (Quiver): features $(size(state.quiver_features))")
    println("Fase 2 (Equivariant): error=$(@sprintf("%.3e", state.equivariance_error))")
    println("Fase 3 (MERA): compressed $(length(state.compressed)), thetas=$(length(thetas))")
    println("Fase 3 (Krylov): complexity=$(@sprintf("%.3e", state.complexity))")
    println("  T mat size: $(size(T_mat))")
    
    # Ruptura
    println("\n--- Induciendo ruptura ---")
    ruptured = induce_rupture(state, noise_mag)
    println("Ruptura aplicada: noise_mag=$noise_mag")
    
    # Recuperación
    println("\n--- Recuperando equivariancia ---")
    recovered, eq_rec = recover(pipeline, ruptured)
    println("Recuperado: error=$(@sprintf("%.3e", eq_rec))")
    
    # Métricas
    println("\n--- Métricas ---")
    metrics, S_score, _, _ = measure_invariants(pipeline, state, ruptured, recovered, noise_mag=noise_mag)
    
    println("Equivariance:")
    println("  Base:      $(@sprintf("%.3e", metrics.equivariance_base))")
    println("  Ruptured:  $(@sprintf("%.3e", metrics.equivariance_ruptured))")
    println("  Recovered: $(@sprintf("%.3e", metrics.equivariance_recovered))")
    println("  Recovery ratio: $(@sprintf("%.2e", metrics.recovery_ratio))")
    println("Complexity base: $(@sprintf("%.3e", metrics.complexity_base))")
    println("S-score (composite): $(@sprintf("%.3f", S_score))")
    println("Status: $(metrics.success ? "✓ SUCCESS" : "✗ FAILED")")
    
    println("="^60)
    
    return Dict(
        "pipeline" => pipeline,
        "state" => state,
        "metrics" => metrics,
        "S_score" => S_score,
        "T_mat" => T_mat,
        "success" => metrics.success,
        "S_ci_low" => nothing  # CI computed at call site if needed
    )
end

end  # module UnifiedWDW
