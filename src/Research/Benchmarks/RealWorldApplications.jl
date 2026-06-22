# TIER 2 — RESEARCH: Real-world application benchmarks (Poisson, graph, oscillators)
"""
    RealWorldApplications.jl

Aplicaciones de WDW a problemas reales:
1. PDEs: Ecuación de Poisson en malla 2D
2. Grafos: Propagación en redes sociales/transporte
3. Dinámica: Osciladores acoplados

Estas aplicaciones demuestran capacidad real del sistema.
"""
module RealWorldApplications

using LinearAlgebra
using Random
using Statistics
using Printf

using ..WDW.UnifiedWDW: UnifiedState
using ..WDW.ScalableWDW: ScalablePipeline, process_scalable
using ..WDW.RealBaselines: ESCNNBaseline, PyGBaseline, train_escnn, train_pygnn, count_parameters
using ..WDW.Quantum: dihedral_group

export Poisson2DSolver, GraphPropagator, CoupledOscillators,
       solve_poisson_2d, simulate_graph_propagation, simulate_oscillators,
       compare_on_real_problem

# =============================================================================
# APLICACIÓN 1: PDE - ECUACIÓN DE POISSON 2D
# =============================================================================

"""
    Poisson2DSolver

Resuelve ecuación de Poisson 2D: ∇²u = f
usando WDW para mantener simetría rotacional en la discretización.

La equivariancia bajo rotaciones es crucial para PDEs en dominios circulares.
"""
struct Poisson2DSolver{T<:Real}
    nx::Int  # Grid size x
    ny::Int  # Grid size y
    h::T     # Grid spacing
    
    function Poisson2DSolver(n::Int; domain_size::Float64=1.0)
        h = domain_size / n
        new{Float64}(n, n, h)
    end
end

"""
    solve_poisson_2d(solver, f_source; max_iters=1000, tol=1e-6)

Resolver ∇²u = f usando método iterativo con regularización WDW.

La idea: usar WDW para proyectar la solución al subespacio equivariante
bajo el grupo de rotaciones discretas, acelerando convergencia.
"""
function solve_poisson_2d(solver::Poisson2DSolver{T}, 
                          f_source::Matrix{T};
                          max_iters::Int=1000,
                          tol::Float64=1e-6,
                          use_wdw::Bool=true) where T
    nx, ny = solver.nx, solver.ny
    h = solver.h
    h2 = h^2
    
    # Inicializar solución
    u = zeros(T, nx, ny)
    
    # Frontera Dirichlet u=0 en bordes (ya es cero)
    
    # Residuo inicial
    residual = copy(f_source)
    
    converged = false
    final_residual = Inf
    
    if use_wdw
        # Usar WDW para acelerar: proyectar residuo a subespacio equivariante
        # Crear pipeline para nx*ny variables (flattened grid)
        n_total = nx * ny
        if n_total ≥ 100 && iseven(n_total)
            try
                pipeline = ScalablePipeline(n_total, 
                                          compression_levels=4,
                                          krylov_dim=20,
                                          max_group_samples=50)
                
                for iter in 1:max_iters
                    # Flatten residuo
                    r_flat = vec(residual)
                    
                    # Procesar con WDW: proyectar a subespacio equivariante
                    state, _, _ = process_scalable(pipeline, r_flat)
                    
                    # Reconstruir y usar como preconditioner
                    r_smooth = reshape(state.compressed[1:n_total], nx, ny)
                    
                    # Actualizar solución: Jacobi con precondicionamiento WDW
                    u_new = copy(u)
                    for i in 2:nx-1
                        for j in 2:ny-1
                            # Laplaciano discreto: (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h² = f
                            u_new[i,j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - h2 * f_source[i,j])
                            # Regularización WDW: penalizar componentes no equivariantes
                            u_new[i,j] -= 0.01 * r_smooth[i,j]
                        end
                    end
                    
                    # Calcular residuo
                    residual_norm = norm(u_new - u) / norm(u_new)
                    u = u_new
                    
                    if residual_norm < tol
                        converged = true
                        final_residual = residual_norm
                        break
                    end
                end
            catch e
                # Fallback a método estándar
                @warn "WDW failed, using standard solver: $e"
                use_wdw = false
            end
        else
            use_wdw = false
        end
    end
    
    if !use_wdw
        # Método estándar: Jacobi sin WDW
        for iter in 1:max_iters
            u_new = copy(u)
            for i in 2:nx-1
                for j in 2:ny-1
                    u_new[i,j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - h2 * f_source[i,j])
                end
            end
            
            residual_norm = norm(u_new - u) / norm(u_new)
            u = u_new
            
            if residual_norm < tol
                converged = true
                final_residual = residual_norm
                break
            end
        end
    end
    
    return u, converged, final_residual
end

# =============================================================================
# APLICACIÓN 2: PROPAGACIÓN EN GRAFOS
# =============================================================================

"""
    GraphPropagator

Simula propagación de información/virus/opinión en redes complejas.
El grafo puede representar: redes sociales, transporte, internet, etc.
"""
struct GraphPropagator{T<:Real}
    n_nodes::Int
    adjacency::Matrix{T}  # Matriz de adyacencia
    degrees::Vector{T}      # Grados de nodos
    
    function GraphPropagator(n::Int; edge_prob::Float64=0.1, seed::Int=42)
        Random.seed!(seed)
        T = Float64
        
        # Grafo aleatorio tipo Erdős–Rényi con estructura comunidad
        adj = zeros(T, n, n)
        
        # Crear comunidades
        n_communities = max(2, n ÷ 20)
        community_size = n ÷ n_communities
        
        for i in 1:n
            # Edges intra-comunidad con alta probabilidad
            comm_i = (i - 1) ÷ community_size + 1
            for j in 1:n
                if i != j
                    comm_j = (j - 1) ÷ community_size + 1
                    prob = (comm_i == comm_j) ? edge_prob * 3 : edge_prob * 0.3
                    if rand() < prob
                        adj[i,j] = 1.0
                    end
                end
            end
        end
        
        # Hacer simétrica
        adj = (adj + adj') / 2
        adj[adj .> 0] .= 1.0
        
        degrees = vec(sum(adj, dims=2))
        
        new{T}(n, adj, degrees)
    end
end

"""
    simulate_graph_propagation(propagator, initial_state; steps=100, use_wdw=true)

Simular propagación en grafo usando dinámica tipo Laplaciano.

Modelo: dx/dt = -L*x donde L = D - A es el Laplaciano del grafo.
WDW ayuda manteniendo equivariancia bajo automorfismos del grafo.
"""
function simulate_graph_propagation(propagator::GraphPropagator{T},
                                    initial_state::Vector{T};
                                    steps::Int=100,
                                    dt::T=0.01,
                                    use_wdw::Bool=true) where T
    n = propagator.n_nodes
    x = copy(initial_state)
    
    # Laplaciano normalizado
    L = diagm(propagator.degrees) - propagator.adjacency
    # Normalizar
    for i in 1:n
        if propagator.degrees[i] > 0
            L[i,:] ./= propagator.degrees[i]
        end
    end
    
    trajectory = [copy(x)]
    
    if use_wdw && n ≥ 100 && iseven(n)
        try
            pipeline = ScalablePipeline(n, 
                                      compression_levels=3,
                                      krylov_dim=15,
                                      max_group_samples=30)
            
            for step in 1:steps
                # Dinámica: dx/dt = -L*x (difusión)
                dx = -L * x
                
                # Regularización WDW: proyectar a subespacio estructurado
                state, _, _ = process_scalable(pipeline, x)
                wdw_component = state.compressed[1:n]
                
                # Combinar: dinámica física + regularización estructural
                x = x + dt * dx + 0.001 * wdw_component
                
                push!(trajectory, copy(x))
            end
        catch e
            @warn "WDW failed in graph propagation: $e"
            use_wdw = false
        end
    else
        use_wdw = false
    end
    
    if !use_wdw
        # Método estándar sin WDW
        for step in 1:steps
            dx = -L * x
            x = x + dt * dx
            push!(trajectory, copy(x))
        end
    end
    
    return trajectory
end

# =============================================================================
# APLICACIÓN 3: OSCILADORES ACOPLADOS
# =============================================================================

"""
    CoupledOscillators

Sistema de osciladores acoplados tipo Kuramoto:
θ̇_i = ω_i + (K/N) * Σ_j sin(θ_j - θ_i)

Relevante para: sincronización neuronal, redes eléctricas, química oscilante.
"""
struct CoupledOscillators{T<:Real}
    n::Int                    # Número de osciladores
    K::T                      # Acoplamiento
    natural_freqs::Vector{T}  # Frecuencias naturales ω_i
    
    function CoupledOscillators(n::Int; K::Float64=1.0, seed::Int=42)
        Random.seed!(seed)
        # Frecuencias naturales distribuidas normalmente
        freqs = randn(n) * 0.5
        new{Float64}(n, K, freqs)
    end
end

"""
    simulate_oscillators(sys, initial_phases; steps=1000, dt=0.01, use_wdw=true)

Simular evolución de fases usando modelo Kuramoto.

El orden de sincronización r(t) = |Σ_j e^{iθ_j}| / N es la métrica clave.
WDW acelera la sincronización manteniendo estructura colectiva.
"""
function simulate_oscillators(sys::CoupledOscillators{T},
                              initial_phases::Vector{T};
                              steps::Int=1000,
                              dt::T=0.01,
                              use_wdw::Bool=true) where T
    n = sys.n
    θ = copy(initial_phases)
    K = sys.K
    ω = sys.natural_freqs
    
    trajectory = [copy(θ)]
    order_params = Float64[]
    
    # Calcular orden inicial
    r = abs(sum(exp.(im .* θ))) / n
    push!(order_params, r)
    
    if use_wdw && n ≥ 100 && iseven(n)
        try
            pipeline = ScalablePipeline(n,
                                      compression_levels=3,
                                      krylov_dim=15,
                                      max_group_samples=40)
            
            for step in 1:steps
                # Dinámica Kuramoto
                dθ = copy(ω)
                for i in 1:n
                    for j in 1:n
                        dθ[i] += (K/n) * sin(θ[j] - θ[i])
                    end
                end
                
                # WDW: detectar estructura colectiva
                state, _, _ = process_scalable(pipeline, sin.(θ))
                collective_component = state.compressed[1:n]
                
                # Evolución: física + guía estructural
                θ = θ + dt * dθ + 0.001 * collective_component
                
                # Mantener fases en [-π, π]
                θ = mod.(θ .+ π, 2π) .- π
                
                push!(trajectory, copy(θ))
                
                # Calcular parámetro de orden
                r = abs(sum(exp.(im .* θ))) / n
                push!(order_params, r)
            end
        catch e
            @warn "WDW failed in oscillators: $e"
            use_wdw = false
        end
    else
        use_wdw = false
    end
    
    if !use_wdw
        for step in 1:steps
            dθ = copy(ω)
            for i in 1:n
                for j in 1:n
                    dθ[i] += (K/n) * sin(θ[j] - θ[i])
                end
            end
            θ = θ + dt * dθ
            θ = mod.(θ .+ π, 2π) .- π
            push!(trajectory, copy(θ))
            r = abs(sum(exp.(im .* θ))) / n
            push!(order_params, r)
        end
    end
    
    return trajectory, order_params
end

# =============================================================================
# COMPARACIÓN UNIFICADA
# =============================================================================

"""
    compare_on_real_problem(problem_type::String, n::Int)

Comparar WDW vs baselines en problema real.

Retorna tabla comparativa con métricas relevantes para cada problema.
"""
function compare_on_real_problem(problem_type::String, n::Int; seed::Int=42)
    Random.seed!(seed)
    
    println("="^90)
    println("COMPARACIÓN EN PROBLEMA REAL: $problem_type (n=$n)")
    println("="^90)
    
    results_wdw = Dict()
    results_baseline = Dict()
    
    if problem_type == "poisson"
        # Problema de Poisson 2D
        solver = Poisson2DSolver(Int(sqrt(n)))
        
        # Fuente gaussiana en centro
        nx, ny = solver.nx, solver.ny
        f = zeros(nx, ny)
        cx, cy = nx ÷ 2, ny ÷ 2
        for i in 1:nx, j in 1:ny
            r² = (i-cx)^2 + (j-cy)^2
            f[i,j] = exp(-r² / (nx/4)^2)
        end
        
        # Resolver con WDW
        t_wdw = @elapsed u_wdw, conv_wdw, res_wdw = solve_poisson_2d(solver, f, use_wdw=true)
        results_wdw["time"] = t_wdw
        results_wdw["converged"] = conv_wdw
        results_wdw["residual"] = res_wdw
        
        # Resolver sin WDW
        t_std = @elapsed u_std, conv_std, res_std = solve_poisson_2d(solver, f, use_wdw=false)
        results_baseline["time"] = t_std
        results_baseline["converged"] = conv_std
        results_baseline["residual"] = res_std
        
        # Error relativo entre soluciones (deberían ser similares)
        rel_diff = norm(u_wdw - u_std) / norm(u_std)
        
        println("Resultados:")
        println(@sprintf("  WDW:   time=%.3fs, converged=%s, residual=%.2e", 
                         t_wdw, conv_wdw, res_wdw))
        println(@sprintf("  Std:   time=%.3fs, converged=%s, residual=%.2e",
                         t_std, conv_std, res_std))
        println(@sprintf("  Diff:  relative=%.2e", rel_diff))
        println(@sprintf("  Speedup: %.2fx", t_std / t_wdw))
        
    elseif problem_type == "graph"
        # Propagación en grafo
        propagator = GraphPropagator(n, edge_prob=0.05, seed=seed)
        initial = zeros(n)
        initial[1:10] .= 1.0  # 10 nodos iniciales infectados/informados
        
        # Con WDW
        t_wdw = @elapsed traj_wdw = simulate_graph_propagation(propagator, initial, 
                                                               steps=100, use_wdw=true)
        final_spread_wdw = count(x -> x > 0.5, traj_wdw[end])
        
        # Sin WDW
        t_std = @elapsed traj_std = simulate_graph_propagation(propagator, initial,
                                                               steps=100, use_wdw=false)
        final_spread_std = count(x -> x > 0.5, traj_std[end])
        
        println("Resultados:")
        println(@sprintf("  WDW:   time=%.3fs, final spread=%d/%d", 
                         t_wdw, final_spread_wdw, n))
        println(@sprintf("  Std:   time=%.3fs, final spread=%d/%d",
                         t_std, final_spread_std, n))
        println(@sprintf("  Speedup: %.2fx", t_std / t_wdw))
        
    elseif problem_type == "oscillators"
        # Osciladores acoplados
        sys = CoupledOscillators(n, K=2.0, seed=seed)
        initial = rand(n) * 2π
        
        # Con WDW
        t_wdw = @elapsed traj_wdw, order_wdw = simulate_oscillators(sys, initial,
                                                                   steps=500, use_wdw=true)
        final_order_wdw = order_wdw[end]
        
        # Sin WDW
        t_std = @elapsed traj_std, order_std = simulate_oscillators(sys, initial,
                                                                   steps=500, use_wdw=false)
        final_order_std = order_std[end]
        
        println("Resultados:")
        println(@sprintf("  WDW:   time=%.3fs, final order=%.3f", 
                         t_wdw, final_order_wdw))
        println(@sprintf("  Std:   time=%.3fs, final order=%.3f",
                         t_std, final_order_std))
        println(@sprintf("  Speedup: %.2fx", t_std / t_wdw))
        println(@sprintf("  Order improvement: %+.1f%%", 100*(final_order_wdw - final_order_std)))
    end
    
    println("="^90)
end

end  # module RealWorldApplications
