# TIER 2 — RESEARCH: Haar-wavelet MERA compression and tensor networks
module Tensor
using LinearAlgebra

struct RepetitionCode
    n::Int
end

function encode(c::RepetitionCode, bit::Int)
    @assert c.n % 2 == 1 "n debe ser impar para corrección por mayoría"
    fill(bit, c.n)
end

function erase(codeword::Vector{Int}, erased::Vector{Int})
    cw = Vector{Union{Int,Nothing}}(undef, length(codeword))
    for i in eachindex(codeword)
        cw[i] = in(i, erased) ? nothing : codeword[i]
    end
    cw
end

function decode(c::RepetitionCode, recv::Vector{Union{Int,Nothing}})
    ones = count(x -> x === 1, recv)
    zeros = count(x -> x === 0, recv)
    if ones > zeros
        true, 1
    elseif zeros > ones
        true, 0
    else
        false, nothing
    end
end

function recoverable_after_erasure(c::RepetitionCode, k::Int)
    k <= (c.n - 1) ÷ 2
end

export RepetitionCode, encode, erase, decode, recoverable_after_erasure

function haar_step(x::AbstractVector)
    n = length(x)
    @assert iseven(n)
    a = similar(x, n ÷ 2)
    d = similar(x, n ÷ 2)
    for i in 1:2:n
        s = (x[i] + x[i+1]) / sqrt(2)
        t = (x[i] - x[i+1]) / sqrt(2)
        a[(i+1) ÷ 2] = s
        d[(i+1) ÷ 2] = t
    end
    a, d
end

function haar_forward(x::AbstractVector, levels::Int)
    cur = copy(x)
    ds = Vector{typeof(x)}()
    for _ in 1:levels
        a, d = haar_step(cur)
        push!(ds, d)
        cur = a
        if length(cur) < 2
            break
        end
    end
    cur, ds
end

function haar_inverse(a, ds::Vector)
    cur = copy(a)
    for k in length(ds):-1:1
        d = ds[k]
        n = length(d)
        x = similar(d, 2*n)
        for i in 1:n
            s = cur[i]
            t = d[i]
            x[2*i - 1] = (s + t) / sqrt(2)
            x[2*i] = (s - t) / sqrt(2)
        end
        cur = x
    end
    cur
end

function mera_reconstruct_truncated(x::AbstractVector, levels::Int, keep_levels::Int)
    a, ds = haar_forward(x, levels)
    k = clamp(keep_levels, 0, length(ds))
    ds2 = [i <= k ? ds[i] : fill!(similar(ds[i]), zero(eltype(ds[i]))) for i in 1:length(ds)]
    haar_inverse(a, ds2)
end

function multiscale_error(x::AbstractVector, levels::Int, keep_levels::Int)
    y = mera_reconstruct_truncated(x, levels, keep_levels)
    nx = norm(x) + 1e-12
    norm(x - y) / nx
end

function rotate_pairs(x::AbstractVector, θ::Real)
    n = length(x)
    @assert iseven(n)
    y = similar(x)
    c = cos(θ)
    s = sin(θ)
    for i in 1:2:n
        a = x[i]
        b = x[i+1]
        y[i] = c*a - s*b
        y[i+1] = s*a + c*b
    end
    y
end

function rotate_pairs_inv(x::AbstractVector, θ::Real)
    rotate_pairs(x, -θ)
end

function param_haar_forward(x::AbstractVector, levels::Int, thetas::AbstractVector)
    cur = copy(x)
    ds = Vector{typeof(x)}()
    for ℓ in 1:levels
        θ = thetas[min(ℓ, length(thetas))]
        cur = rotate_pairs(cur, θ)
        a, d = haar_step(cur)
        push!(ds, d)
        cur = a
        if length(cur) < 2
            break
        end
    end
    cur, ds
end

function param_haar_inverse(a, ds::Vector, thetas::AbstractVector)
    cur = copy(a)
    for k in length(ds):-1:1
        d = ds[k]
        n = length(d)
        x = similar(d, 2*n)
        for i in 1:n
            s = cur[i]
            t = d[i]
            x[2*i - 1] = (s + t) / sqrt(2)
            x[2*i] = (s - t) / sqrt(2)
        end
        θ = thetas[min(k, length(thetas))]
        cur = rotate_pairs_inv(x, θ)
    end
    cur
end

function param_mera_reconstruct_truncated(x::AbstractVector, levels::Int, keep_levels::Int, thetas::AbstractVector)
    a, ds = param_haar_forward(x, levels, thetas)
    k = clamp(keep_levels, 0, length(ds))
    ds2 = [i <= k ? ds[i] : fill!(similar(ds[i]), zero(eltype(ds[i]))) for i in 1:length(ds)]
    param_haar_inverse(a, ds2, thetas)
end

function param_multiscale_error(x::AbstractVector, levels::Int, keep_levels::Int, thetas::AbstractVector)
    y = param_mera_reconstruct_truncated(x, levels, keep_levels, thetas)
    nx = norm(x) + 1e-12
    norm(x - y) / nx
end

function optimize_thetas(x::AbstractVector, levels::Int, keep_levels::Int; iters::Int=50, step::Real=0.1)
    thetas = zeros(Float64, levels)
    e = param_multiscale_error(x, levels, keep_levels, thetas)
    s = step
    for _ in 1:iters
        improved = false
        for l in 1:levels
            θ = thetas[l]
            thetas[l] = θ + s
            ep = param_multiscale_error(x, levels, keep_levels, thetas)
            thetas[l] = θ - s
            em = param_multiscale_error(x, levels, keep_levels, thetas)
            if ep < e || em < e
                improved = true
                if ep <= em && ep < e
                    thetas[l] = θ + s
                    e = ep
                elseif em < e
                    thetas[l] = θ - s
                    e = em
                else
                    thetas[l] = θ
                end
            else
                thetas[l] = θ
            end
        end
        s *= 0.9
        if !improved && s < 1e-4
            break
        end
    end
    thetas, e
end

"""
    mera_compress(x; ratio=2)

Actual MERA-style compression: forward Haar transform then keep only the
first `length(x) ÷ ratio` coarse coefficients (truncate detail bands).

Returns (compressed, full_reconstruction) where:
  - `compressed` has length `n ÷ ratio` (the true compressed representation)
  - `full_reconstruction` has length `n` (padded with zeros for downstream compat)
"""
function mera_compress(x::AbstractVector; ratio::Int=2)
    n = length(x)
    target = max(1, n ÷ ratio)
    levels = Int(ceil(log2(n)))
    a, ds = haar_forward(x, levels)
    # Build compressed: concatenate final average with kept detail coefficients
    compressed = vcat(a, vcat([ds[i] for i in 1:levels]...))
    # Truncate to target length
    compressed = compressed[1:min(target, length(compressed))]
    # Full-size reconstruction by padding truncated coefficients with zeros
    recon = mera_reconstruct(compressed, n)
    return compressed, recon
end

"""
    mera_reconstruct(compressed, original_n)

Reconstruct to `original_n` by zero-padding the compressed coefficients
and running inverse Haar.
"""
function mera_reconstruct(compressed::AbstractVector, original_n::Int)
    n = original_n
    levels = Int(ceil(log2(n)))
    # Pad compressed with zeros to match the full coefficient count
    n_coarse = n
    for _ in 1:levels
        n_coarse = n_coarse ÷ 2 + (n_coarse % 2)
    end
    full_coeffs = vcat(compressed, zeros(eltype(compressed), n - length(compressed)))
    # Reconstruct from padded coefficients by iterating inverse steps
    a = full_coeffs[1:max(1, n ÷ 2^levels)]
    ds = Vector{typeof(a)}()
    offset = length(a)
    for ℓ in 1:levels
        chunk_len = min(2^(levels - ℓ), n ÷ 2)
        chunk = full_coeffs[offset+1:min(offset+chunk_len, end)]
        push!(ds, chunk)
        offset += length(chunk)
    end
    return haar_inverse(a, ds)
end

export haar_step, haar_forward, haar_inverse, mera_reconstruct_truncated, multiscale_error, rotate_pairs, param_haar_forward, param_haar_inverse, param_mera_reconstruct_truncated, param_multiscale_error, optimize_thetas, mera_compress, mera_reconstruct

end
