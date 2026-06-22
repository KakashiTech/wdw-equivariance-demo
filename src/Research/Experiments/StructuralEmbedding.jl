# TIER 3 — EXPERIMENTAL: PCA-based structural fingerprint embedding
module StructuralEmbedding

using LinearAlgebra, Statistics, Random, Printf

export EmbeddingResult, structural_embedding, fingerprint_distance,
       find_nearest, embedding_summary, cluster_by_embedding

struct EmbeddingResult
    coords::Matrix{Float64}          # n_models × n_dims
    explained_var::Vector{Float64}
    model_names::Vector{String}
    measurement_names::Vector{String}
    raw_matrix::Matrix{Float64}      # n_measurements × n_models before PCA
    normalized_matrix::Matrix{Float64}
    pca_center::Vector{Float64}
    pca_rotation::Matrix{Float64}    # n_measurements × n_dims
    n_dims::Int
end

function structural_embedding(results::Vector{<:Any};
                               n_dims::Int=5,
                               model_names::Vector{String}=String[])
    n_models = length(results)
    n_models < 1 && error("Need at least 1 model")
    
    if isempty(model_names)
        model_names = ["Model_$i" for i in 1:n_models]
    end
    
    all_meas_names = String[]
    meas_vals = Vector{Float64}[]
    
    for (i, r) in enumerate(results)
        if hasproperty(r, :measurement_names) && hasproperty(r, :measurement_matrix)
            push!(meas_vals, r.measurement_matrix[:])
            if i == 1
                all_meas_names = r.measurement_names
            end
        else
            push!(meas_vals, Float64[])
        end
    end
    
    min_len = minimum(length, meas_vals)
    if min_len == 0
        error("All results must have non-empty measurement matrices")
    end
    
    X = hcat([v[1:min_len] for v in meas_vals]...)  # n_meas × n_models
    n_meas, n_models2 = size(X)
    
    μ = vec(mean(X, dims=2))
    σ = vec(std(X, dims=2))
    σ[σ .== 0.0] .= 1.0
    X_norm = (X .- μ) ./ σ
    
    # PCA via SVD
    n_dims_actual = min(n_dims, n_meas, n_models2)
    U, S, Vt = svd(X_norm)
    # Vt is always n_models × k (k = min(n_meas, n_models)), so Vt[:, 1:k] gives n_models × k
    coords = Vt[:, 1:n_dims_actual] .* S[1:n_dims_actual]' ./ sqrt(max(n_models2 - 1, 1))
    explained = (S[1:n_dims_actual].^2) ./ sum(S.^2)
    
    rot = U[:, 1:n_dims_actual]  # loadings: n_meas × n_dims
    
    return EmbeddingResult(coords, explained, model_names,
                          all_meas_names[1:min_len], X, X_norm, μ, rot, n_dims_actual)
end

function fingerprint_distance(emb::EmbeddingResult, i::Int, j::Int)
    return norm(emb.coords[i, :] - emb.coords[j, :])
end

function find_nearest(emb::EmbeddingResult, i::Int; k::Int=3)
    k = min(k, size(emb.coords, 1) - 1)
    dists = [(j, fingerprint_distance(emb, i, j)) for j in 1:size(emb.coords, 1) if j != i]
    sort!(dists, by=x->x[2])
    return dists[1:k]
end

function cluster_by_embedding(emb::EmbeddingResult; threshold::Float64=1.0)
    n = size(emb.coords, 1)
    assigned = zeros(Int, n)
    next_cluster = 1
    for i in 1:n
        if assigned[i] == 0
            assigned[i] = next_cluster
            for j in i+1:n
                if assigned[j] == 0 && fingerprint_distance(emb, i, j) < threshold
                    assigned[j] = next_cluster
                end
            end
            next_cluster += 1
        end
    end
    return assigned
end

function embedding_summary(emb::EmbeddingResult)
    println("="^72)
    println("  STRUCTURAL EMBEDDING SUMMARY")
    println("="^72)
    @printf "  Models:          %d\n" size(emb.coords, 1)
    @printf "  Measurements:    %d\n" length(emb.measurement_names)
    @printf "  Embedding dims:  %d\n" emb.n_dims
    ev_total = sum(emb.explained_var) * 100
    ev_pc1 = emb.explained_var[1] * 100
    ev_pc2 = length(emb.explained_var) > 1 ? emb.explained_var[2] * 100 : 0.0
    @printf "  Explained var:   %.1f%% (PC1=%.1f%%, PC2=%.1f%%)\n" ev_total ev_pc1 ev_pc2
    
    println("\n  ── Model coordinates (first 3 PCs) ──")
    @printf "  %-20s %10s %10s %10s\n" "Model" "PC1" "PC2" "PC3"
    println("  " * "-" ^ 55)
    for i in 1:size(emb.coords, 1)
        c1 = emb.coords[i, 1]
        c2 = emb.n_dims > 1 ? emb.coords[i, 2] : 0.0
        c3 = emb.n_dims > 2 ? emb.coords[i, 3] : 0.0
        @printf "  %-20s %10.4f %10.4f %10.4f\n" emb.model_names[i] c1 c2 c3
    end
    
    println("\n  ── Pairwise distances ──")
    for i in 1:min(5, size(emb.coords, 1))
        for j in i+1:min(5, size(emb.coords, 1))
            d = fingerprint_distance(emb, i, j)
            @printf "  %s ↔ %s: %.4f\n" emb.model_names[i] emb.model_names[j] d
        end
    end
    
    clusters = cluster_by_embedding(emb)
    n_clusters = length(unique(clusters))
    println("\n  ── Clusters (threshold=1.0) ──")
    @printf "  %d clusters detected\n" n_clusters
    for c in 1:n_clusters
        members = findall(==(c), clusters)
        @printf "  Cluster %d: %s\n" c join(emb.model_names[members], ", ")
    end
    println("="^72)
end

function export_embedding_csv(emb::EmbeddingResult, path::String)
    open(path, "w") do io
        header = "model_name," * join(["PC$i" for i in 1:emb.n_dims], ",")
        write(io, header * "\n")
        for i in 1:size(emb.coords, 1)
            row = emb.model_names[i] * "," * join([string(round(emb.coords[i, d], digits=6)) for d in 1:emb.n_dims], ",")
            write(io, row * "\n")
        end
    end
    println("  Exported to: $path")
end

function top_contributing_measurements(emb::EmbeddingResult; n::Int=5)
    println("\n  ── Top measurements contributing to PC1 ──")
    loadings = abs.(emb.pca_rotation[:, 1])
    idx = sortperm(loadings, rev=true)[1:min(n, length(loadings))]
    for (rank, i) in enumerate(idx)
        @printf "  %2d. %-50s %.4f\n" rank emb.measurement_names[i] loadings[i]
    end
    return emb.measurement_names[idx]
end

end
