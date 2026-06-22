# TIER 2 — RESEARCH: Flux.jl implementation of auto-symmetry models
module AutoSymmetryFlux

using Flux
using Zygote
using LinearAlgebra
using Statistics
using Random
# using MLDatasets  # Comentado - usando generación sintética

export LatentLieGANFlux, LieSDFlux, SymmetryGANFlux,
       train_liegan!, train_liesd!, train_symmetrygan!,
       discover_symmetries_flux, evaluate_symmetry_quality_flux,
       RotatedMNIST, load_rotated_mnist,
       WDWAutoSymmetryModel, train_wdw_model!, evaluate_wdw_model,
       BaselineMLP, BaselineCNN, train_baseline!, evaluate_baseline,
       run_full_benchmark, statistical_comparison

# =============================================================================
# PARTE 1: LATENT LIEGAN CON FLUX (Implementación Real)
# =============================================================================

"""
    LatentLieGANFlux

Implementación real de Latent LieGAN usando Flux.jl con backprop genuino.
"""
struct LatentLieGANFlux
    encoder::Chain
    decoder::Chain
    latent_dim::Int
    group_type::String
end

"""
    LatentLieGANFlux(input_dim, latent_dim; hidden_dims=[128, 64], group_type="SO(2)")

Constructor con redes Flux reales.
"""
function LatentLieGANFlux(input_dim::Int, latent_dim::Int; 
                           hidden_dims::Vector{Int}=[128, 64], 
                           group_type::String="SO(2)")
    
    # Encoder: input -> latent
    encoder_layers = []
    dims = [input_dim; hidden_dims; latent_dim]
    for i in 1:length(hidden_dims)
        push!(encoder_layers, Dense(dims[i] => dims[i+1], relu))
    end
    push!(encoder_layers, Dense(dims[end-1] => dims[end]))
    encoder = Chain(encoder_layers...)
    
    # Decoder: latent -> input
    decoder_layers = []
    dims_rev = [latent_dim; reverse(hidden_dims); input_dim]
    for i in 1:length(hidden_dims)
        push!(decoder_layers, Dense(dims_rev[i] => dims_rev[i+1], relu))
    end
    push!(decoder_layers, Dense(dims_rev[end-1] => dims_rev[end]))
    decoder = Chain(decoder_layers...)
    
    return LatentLieGANFlux(encoder, decoder, latent_dim, group_type)
end

function encode(model::LatentLieGANFlux, x)
    return model.encoder(x)
end

function decode(model::LatentLieGANFlux, z)
    return model.decoder(z)
end

"""
    apply_group_action(latent, group_type, angle)

Aplica acción de grupo en espacio latente.
"""
function apply_group_action(z::AbstractVector, group_type::String, θ::Real)
    if group_type == "SO(2)" && length(z) >= 2
        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        z_rot = R * z[1:2]
        return vcat(z_rot, z[3:end])
    else
        return z
    end
end

"""
    train_liegan!(model, data, epochs; lr=0.001, lambda_equiv=1.0, batch_size=32)

Entrena LatentLieGAN con pérdida de reconstrucción + equivariancia.
"""
function train_liegan!(model::LatentLieGANFlux, 
                       data::Vector{<:AbstractVector}, 
                       epochs::Int;
                       lr::Float64=0.001, 
                       lambda_equiv::Float64=1.0,
                       batch_size::Int=32)
    
    opt = Adam(lr)
    st_enc = Flux.setup(opt, model.encoder)
    st_dec = Flux.setup(opt, model.decoder)
    n_samples = length(data)
    
    println("Entrenando LatentLieGAN (Flux)...")
    println("  Epochs: $epochs")
    println("  Samples: $n_samples")
    println("  Latent dim: $(model.latent_dim)")
    
    for epoch in 1:epochs
        # Mezclar datos
        shuffle_indices = randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_indices = shuffle_indices[batch_start:batch_end]
            
            # Compute loss separately (Flux.gradient does not return the value)
            function compute_loss(enc, dec)
                loss = 0.0f0
                for idx in batch_indices
                    x = data[idx]
                    z = enc(x)
                    x_recon = dec(z)
                    loss_recon = sum((x_recon .- x).^2)
                    θ = 2π * rand()
                    z_rot = apply_group_action(z, model.group_type, θ)
                    x_rot = dec(z_rot)
                    z_back = enc(x_rot)
                    loss += loss_recon + lambda_equiv * sum((z_back .- z_rot).^2)
                end
                return loss / length(batch_indices)
            end
            
            batch_loss = compute_loss(model.encoder, model.decoder)
            grads = Flux.gradient(model.encoder, model.decoder) do enc, dec
                compute_loss(enc, dec)
            end
            
            epoch_loss += batch_loss
            n_batches += 1
            
            # Actualizar parámetros
            Flux.update!(st_enc, model.encoder, grads[1])
            Flux.update!(st_dec, model.decoder, grads[2])
        end
        
        avg_loss = epoch_loss / n_batches
        if epoch % 10 == 0 || epoch == 1
            println("  Epoch $epoch: loss = $(round(avg_loss, digits=4))")
        end
    end
    
    println("✓ LatentLieGAN entrenado")
    return model
end

# =============================================================================
# PARTE 2: LIEBSD CON FLUX (Descubrimiento vía Gradientes)
# =============================================================================

"""
    LieSDFlux

Lie Symmetry Discovery con red Flux real y cálculo de Jacobianas.
"""
struct LieSDFlux
    network::Chain
    max_generators::Int
    tolerance::Float64
end

function LieSDFlux(input_dim::Int, hidden_dims::Vector{Int}=[64, 32]; 
                    max_generators::Int=5,
                    tolerance::Float64=1e-4)
    
    layers = []
    dims = [input_dim; hidden_dims; input_dim]
    for i in 1:length(hidden_dims)
        push!(layers, Dense(dims[i] => dims[i+1], relu))
    end
    push!(layers, Dense(dims[end-1] => dims[end]))
    network = Chain(layers...)
    
    return LieSDFlux(network, max_generators, tolerance)
end

function network_forward(model::LieSDFlux, x)
    return model.network(x)
end

"""
    compute_jacobian_flux(model, x)

Computa Jacobiana usando Zygote (diferenciación automática real).
"""
function compute_jacobian_flux(model::LieSDFlux, x::AbstractVector)
    n = length(x)
    J = zeros(n, n)
    
    for i in 1:n
        # Gradiente de la salida i respecto a entrada
        grad = Zygote.gradient(x -> model.network(x)[i], x)[1]
        J[i, :] = grad
    end
    
    return J
end

"""
    train_liesd!(model, data, epochs; lr=0.001)

Entrena la red LieSD para que sea invariante bajo simetrías.
"""
function train_liesd!(model::LieSDFlux, 
                      data::Vector{<:AbstractVector}, 
                      epochs::Int;
                      lr::Float64=0.001,
                      batch_size::Int=32)
    
    opt = Adam(lr)
    st_net = Flux.setup(opt, model.network)
    n_samples = length(data)
    
    println("Entrenando LieSD (Flux)...")
    
    for epoch in 1:epochs
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch = data[batch_start:batch_end]
            
            loss, grads = Flux.withgradient(model.network) do net
                l = 0.0
                for x in batch
                    y = net(x)
                    l += sum((y .- x).^2)
                end
                return l / length(batch)
            end
            
            epoch_loss += loss
            n_batches += 1
            
            Flux.update!(st_net, model.network, grads[1])
        end
        
        if epoch % 10 == 0 || epoch == 1
            println("  Epoch $epoch: loss = $(round(epoch_loss/n_batches, digits=4))")
        end
    end
    
    println("✓ LieSD entrenado")
    return model
end

"""
    find_generators_liesd_flux(model, data)

Encuentra generadores resolviendo ecuaciones de conmutación con Jacobianas reales.
"""
function find_generators_liesd_flux(model::LieSDFlux, 
                                     data::Vector{<:AbstractVector})
    
    println("\nDescubriendo generadores con LieSD (Flux)...")
    println("  Muestras: $(length(data))")
    
    n = length(data[1])
    
    # Computar Jacobianas en múltiples puntos
    jacobians = [compute_jacobian_flux(model, x) for x in data[1:min(20, length(data))]]
    
    # Buscar matrices que conmuten con Jacobianas
    generators_found = Matrix{Float64}[]
    
    # Probar candidatos: generadores de rotación, traslación, etc.
    test_generators = []
    
    if n >= 2
        # Generador de rotación SO(2)
        G_rot = zeros(n, n)
        G_rot[1, 2] = -1.0
        G_rot[2, 1] = 1.0
        push!(test_generators, ("rotation", G_rot))
    end
    
    # Más generadores posibles...
    for i in 1:n
        for j in i+1:n
            G = zeros(n, n)
            G[i, j] = 1.0
            G[j, i] = -1.0
            push!(test_generators, ("rot_$(i)_$(j)", G))
        end
    end
    
    # Verificar cuáles conmutan
    for (name, G_test) in test_generators
        commutation_errors = []
        for J in jacobians
            comm = J * G_test - G_test * J
            push!(commutation_errors, norm(comm))
        end
        
        avg_error = mean(commutation_errors)
        if avg_error < model.tolerance * 10  # Tolerancia relajada
            push!(generators_found, G_test)
            println("  Generador encontrado: $name (error: $(round(avg_error, digits=6)))")
        end
        
        if length(generators_found) >= model.max_generators
            break
        end
    end
    
    return (
        generators = generators_found,
        n_generators = length(generators_found),
        jacobians_computed = length(jacobians)
    )
end

# =============================================================================
# PARTE 3: SYMMETRYGAN CON FLUX
# =============================================================================

"""
    SymmetryGANFlux

SymmetryGAN con generador y discriminador Flux reales.
"""
struct SymmetryGANFlux
    generator::Chain      # Aprende transformación de simetría
    discriminator::Chain  # Detecta si es transformación real
    group_dim::Int
end

function SymmetryGANFlux(data_dim::Int, group_dim::Int=2;
                          gen_hidden::Vector{Int}=[128, 64],
                          disc_hidden::Vector{Int}=[64, 32])
    
    # Generador: data -> transformed_data
    gen_layers = []
    dims = [data_dim; gen_hidden; data_dim]
    for i in 1:length(gen_hidden)
        push!(gen_layers, Dense(dims[i] => dims[i+1], relu))
    end
    push!(gen_layers, Dense(dims[end-1] => dims[end]))
    generator = Chain(gen_layers...)
    
    # Discriminador: (data, transformed_data) -> real/fake
    disc_layers = []
    dims_disc = [2*data_dim; disc_hidden; 1]
    for i in 1:length(disc_hidden)
        push!(disc_layers, Dense(dims_disc[i] => dims_disc[i+1], relu))
    end
    push!(disc_layers, Dense(dims_disc[end-1] => dims_disc[end], σ))
    discriminator = Chain(disc_layers...)
    
    return SymmetryGANFlux(generator, discriminator, group_dim)
end

"""
    train_symmetrygan!(model, data, epochs; lr_gen=0.001, lr_disc=0.001)

Entrena GAN con alternancia generador/discriminador.
"""
function train_symmetrygan!(model::SymmetryGANFlux,
                            data::Vector{<:AbstractVector},
                            epochs::Int;
                            lr_gen::Float64=0.001,
                            lr_disc::Float64=0.001,
                            batch_size::Int=32)
    
    opt_gen = Adam(lr_gen)
    opt_disc = Adam(lr_disc)
    st_gen = Flux.setup(opt_gen, model.generator)
    st_disc = Flux.setup(opt_disc, model.discriminator)
    
    n_samples = length(data)
    
    println("Entrenando SymmetryGAN (Flux)...")
    
    for epoch in 1:epochs
        gen_loss_epoch = 0.0
        disc_loss_epoch = 0.0
        n_batches = 0
        
        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch = data[batch_start:batch_end]
            
            # === ENTRENAR DISCRIMINADOR ===
            disc_loss, grads_disc = Flux.withgradient(model.discriminator) do disc
                loss = 0.0
                for x in batch
                    x_fake = model.generator(x)
                    
                    # Score para real y fake
                    real_score = disc(vcat(x, x))
                    fake_score = disc(vcat(x, x_fake))
                    
                    # BCE Loss simplificado
                    loss += -log(real_score[1] + 1e-8) - log(1 - fake_score[1] + 1e-8)
                end
                return loss / length(batch)
            end
            
            Flux.update!(st_disc, model.discriminator, grads_disc[1])
            disc_loss_epoch += disc_loss
            
            # === ENTRENAR GENERADOR ===
            gen_loss, grads_gen = Flux.withgradient(model.generator) do gen
                loss = 0.0
                for x in batch
                    x_fake = gen(x)
                    fake_score = model.discriminator(vcat(x, x_fake))
                    loss += -log(fake_score[1] + 1e-8)  # Queremos engañar al disc
                end
                return loss / length(batch)
            end
            
            Flux.update!(st_gen, model.generator, grads_gen[1])
            gen_loss_epoch += gen_loss
            
            n_batches += 1
        end
        
        if epoch % 10 == 0 || epoch == 1
            println("  Epoch $epoch: gen_loss=$(round(gen_loss_epoch/n_batches, digits=4)), " *
                    "disc_loss=$(round(disc_loss_epoch/n_batches, digits=4))")
        end
    end
    
    println("✓ SymmetryGAN entrenado")
    return model
end

# =============================================================================
# PARTE 4: ROTATED MNIST BENCHMARK REAL
# =============================================================================

"""
    RotatedMNIST

Dataset MNIST con rotaciones aleatorias para benchmark de descubrimiento de simetrías.
"""
struct RotatedMNIST
    X_train::Array{Float32, 3}  # 28×28×n
    Y_train::Vector{Int}
    X_test::Array{Float32, 3}
    Y_test::Vector{Int}
    angles_train::Vector{Float64}
    angles_test::Vector{Float64}
end

"""
    load_rotated_mnist(;n_train=10000, n_test=2000, max_angle=2π)

Carga MNIST sintético con rotaciones aleatorias (no requiere MLDatasets).
Genera dígitos sintéticos tipo MNIST para benchmark.
"""
function load_rotated_mnist(;n_train::Int=10000, n_test::Int=2000, max_angle::Float64=2π)
    
    println("Generando Rotated MNIST sintético...")
    
    # Generar dígitos sintéticos tipo MNIST
    # Cada dígito es una imagen 28x28 con un patrón característico
    
    X_train = Array{Float32, 3}(undef, 28, 28, n_train)
    Y_train = Vector{Int}(undef, n_train)
    X_test = Array{Float32, 3}(undef, 28, 28, n_test)
    Y_test = Vector{Int}(undef, n_test)
    
    angles_train = rand(n_train) .* max_angle
    angles_test = rand(n_test) .* max_angle
    
    # Generar dígitos base (sin rotar)
    for i in 1:n_train
        digit = rand(0:9)
        Y_train[i] = digit + 1  # 1-10 para clasificación
        X_train[:, :, i] = generate_synthetic_digit(digit)
    end
    
    for i in 1:n_test
        digit = rand(0:9)
        Y_test[i] = digit + 1
        X_test[:, :, i] = generate_synthetic_digit(digit)
    end
    
    # Aplicar rotaciones
    X_train_rotated = similar(X_train)
    X_test_rotated = similar(X_test)
    
    for i in 1:n_train
        X_train_rotated[:, :, i] = rotate_image(X_train[:, :, i], angles_train[i])
    end
    
    for i in 1:n_test
        X_test_rotated[:, :, i] = rotate_image(X_test[:, :, i], angles_test[i])
    end
    
    println("  Train: $n_train muestras con rotaciones [0, $(round(max_angle, digits=2))] rad")
    println("  Test: $n_test muestras")
    
    return RotatedMNIST(
        X_train_rotated, Y_train,
        X_test_rotated, Y_test,
        angles_train, angles_test
    )
end

"""
    generate_synthetic_digit(digit)

Genera un dígito sintético tipo MNIST (28x28) basado en el dígito 0-9.
"""
function generate_synthetic_digit(digit::Int)
    img = zeros(Float32, 28, 28)
    
    # Centro de la imagen
    cx, cy = 14.0f0, 14.0f0
    
    if digit == 0
        # Círculo
        for i in 1:28, j in 1:28
            d = sqrt((i-cx)^2 + (j-cy)^2)
            if 6 < d < 9
                img[i, j] = 0.8f0
            end
        end
    elseif digit == 1
        # Línea vertical
        img[6:22, 13:15] .= 0.8f0
    elseif digit == 2
        # Curva
        for i in 6:22
            j = round(Int, cy + 5*sin((i-6)/16*π))
            if 1 <= j <= 28
                img[i, max(1,min(28,j))] = 0.8f0
                img[i, max(1,min(28,j+1))] = 0.6f0
            end
        end
    elseif digit == 3
        # Dos semicírculos
        for i in 1:28, j in 1:28
            d = sqrt((i-cx+3)^2 + (j-cy)^2)
            if 5 < d < 8 && j > cy
                img[i, j] = 0.8f0
            end
            if 5 < d < 8 && j <= cy
                img[i, j] = 0.8f0
            end
        end
    elseif digit == 4
        # Cruz asimétrica
        img[6:20, 12:14] .= 0.8f0  # Vertical
        img[13:15, 6:20] .= 0.8f0  # Horizontal
    elseif digit == 5
        # Patrón en S
        for i in 6:13
            j = round(Int, cx + 5)
            img[i, max(1,min(28,j))] = 0.8f0
        end
        for i in 13:22
            j = round(Int, cx - 5)
            img[i, max(1,min(28,j))] = 0.8f0
        end
        img[13, cx-5:cx+5] .= 0.8f0
    elseif digit == 6
        # Círculo con cola
        for i in 1:28, j in 1:28
            d = sqrt((i-cx)^2 + (j-cy-2)^2)
            if 5 < d < 8
                img[i, j] = 0.8f0
            end
        end
        img[6:10, 14:16] .= 0.8f0
    elseif digit == 7
        # Línea diagonal
        for i in 6:22
            j = round(Int, cy + (i-6))
            if 1 <= j <= 28
                img[i, max(1,min(28,j))] = 0.8f0
            end
        end
    elseif digit == 8
        # Dos círculos
        for i in 1:28, j in 1:28
            d1 = sqrt((i-cx)^2 + (j-cy-4)^2)
            d2 = sqrt((i-cx)^2 + (j-cy+4)^2)
            if 4 < d1 < 6 || 4 < d2 < 6
                img[i, j] = 0.8f0
            end
        end
    elseif digit == 9
        # Círculo invertido
        for i in 1:28, j in 1:28
            d = sqrt((i-cx)^2 + (j-cy+2)^2)
            if 5 < d < 8
                img[i, j] = 0.8f0
            end
        end
        img[18:22, 14:16] .= 0.8f0
    end
    
    # Añadir ruido
    img .+= randn(Float32, 28, 28) .* 0.1f0
    img = clamp.(img, 0.0f0, 1.0f0)
    
    return img
end

"""
    rotate_image(img, angle)

Rota imagen 28×28 por ángulo dado (interpolación bilineal simple).
"""
function rotate_image(img::Matrix{Float32}, angle::Float64)
    n = size(img, 1)
    center = (n + 1) / 2
    
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    
    img_rotated = zeros(Float32, n, n)
    
    for i in 1:n, j in 1:n
        # Coordenadas centradas
        xy = [i - center, j - center]
        xy_rot = R * xy
        
        # Volver a coordenadas originales
        i_src = xy_rot[1] + center
        j_src = xy_rot[2] + center
        
        # Interpolación bilineal simple
        if 1 <= i_src <= n && 1 <= j_src <= n
            i0, j0 = floor(Int, i_src), floor(Int, j_src)
            i1, j1 = min(i0+1, n), min(j0+1, n)
            
            di = i_src - i0
            dj = j_src - j0
            
            # Valor interpolado
            val = (1-di)*(1-dj)*img[i0, j0] +
                  di*(1-dj)*img[i1, j0] +
                  (1-di)*dj*img[i0, j1] +
                  di*dj*img[i1, j1]
            
            img_rotated[i, j] = val
        end
    end
    
    return img_rotated
end

# =============================================================================
# PARTE 5: MODELO WDW AUTOSYMMETRY COMPLETO
# =============================================================================

"""
    WDWAutoSymmetryModel

Modelo WDW completo: descubrimiento + clasificación.
"""
struct WDWAutoSymmetryModel
    liegan::LatentLieGANFlux
    classifier::Chain
    n_classes::Int
end

function WDWAutoSymmetryModel(input_dim::Int, latent_dim::Int, n_classes::Int;
                               liegan_hidden::Vector{Int}=[128, 64],
                               classifier_hidden::Vector{Int}=[64, 32])
    
    liegan = LatentLieGANFlux(input_dim, latent_dim; hidden_dims=liegan_hidden)
    
    # Clasificador: latent -> clases
    layers = []
    dims = [latent_dim; classifier_hidden; n_classes]
    for i in 1:length(classifier_hidden)
        push!(layers, Dense(dims[i] => dims[i+1], relu))
    end
    push!(layers, Dense(dims[end-1] => dims[end]))
    push!(layers, softmax)
    classifier = Chain(layers...)
    
    return WDWAutoSymmetryModel(liegan, classifier, n_classes)
end

function train_wdw_model!(model::WDWAutoSymmetryModel,
                          X_train::Vector{<:AbstractVector},
                          Y_train::Vector{Int},
                          epochs::Int;
                          lr::Float64=0.001,
                          batch_size::Int=32)
    
    println("\nEntrenando WDW AutoSymmetry Model (Flux)...")
    
    # FASE 1: Entrenar LieGAN
    println("\n[1/2] Entrenando descubridor de simetrías (LieGAN)...")
    train_liegan!(model.liegan, X_train, div(epochs, 2), lr=lr, batch_size=batch_size)
    
    # FASE 2: Entrenar clasificador
    println("\n[2/2] Entrenando clasificador...")
    
    opt = Adam(lr)
    st_clf = Flux.setup(opt, model.classifier)
    n_samples = length(X_train)
    
    # Convertir etiquetas a one-hot
    Y_onehot = [Flux.onehot(y, 1:model.n_classes) for y in Y_train]
    
    for epoch in 1:div(epochs, 2)
        shuffle_indices = randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_indices = shuffle_indices[batch_start:batch_end]
            
            loss, grads = Flux.withgradient(model.classifier) do clf
                l = 0.0
                for idx in batch_indices
                    x = X_train[idx]
                    y = Y_onehot[idx]
                    
                    # Forward: x -> latent -> class
                    z = encode(model.liegan, x)
                    pred = clf(z)
                    
                    # Cross-entropy loss
                    l -= sum(y .* log.(pred .+ 1e-8))
                end
                return l / length(batch_indices)
            end
            
            epoch_loss += loss
            n_batches += 1
            
            Flux.update!(st_clf, model.classifier, grads[1])
        end
        
        if epoch % 5 == 0 || epoch == 1
            println("  Epoch $epoch: loss = $(round(epoch_loss/n_batches, digits=4))")
        end
    end
    
    println("✓ WDW Model entrenado")
    return model
end

function evaluate_wdw_model(model::WDWAutoSymmetryModel,
                            X_test::Vector{<:AbstractVector},
                            Y_test::Vector{Int})
    
    correct = 0
    for (x, y) in zip(X_test, Y_test)
        z = encode(model.liegan, x)
        pred = model.classifier(z)
        pred_class = argmax(pred)
        if pred_class == y
            correct += 1
        end
    end
    
    accuracy = correct / length(Y_test)
    return accuracy
end

# =============================================================================
# PARTE 6: BASELINES REALES
# =============================================================================

"""
    BaselineMLP

MLP estándar para comparación.
"""
struct BaselineMLP
    model::Chain
end

function BaselineMLP(input_dim::Int, n_classes::Int;
                      hidden_dims::Vector{Int}=[128, 64, 32])
    layers = []
    dims = [input_dim; hidden_dims; n_classes]
    for i in 1:length(hidden_dims)
        push!(layers, Dense(dims[i] => dims[i+1], relu))
    end
    push!(layers, Dense(dims[end-1] => dims[end]))
    push!(layers, softmax)
    
    return BaselineMLP(Chain(layers...))
end

"""
    BaselineCNN

CNN simple para comparación en imágenes.
"""
struct BaselineCNN
    model::Chain
end

function BaselineCNN(n_classes::Int=10)
    # CNN simple: 28x28 -> conv -> flatten -> dense -> classes
    model = Chain(
        Conv((3, 3), 1 => 16, relu, pad=1),   # 28x28x1 -> 28x28x16
        MaxPool((2, 2)),                       # -> 14x14x16
        Conv((3, 3), 16 => 32, relu, pad=1),  # -> 14x14x32
        MaxPool((2, 2)),                       # -> 7x7x32
        Flux.flatten,                          # -> 1568
        Dense(1568 => 128, relu),
        Dense(128 => n_classes),
        softmax
    )
    
    return BaselineCNN(model)
end

function train_baseline!(baseline::Union{BaselineMLP, BaselineCNN},
                         X_train, Y_train::Vector{Int},
                         epochs::Int;
                         lr::Float64=0.001,
                         batch_size::Int=32)
    
    opt = Adam(lr)
    st_base = Flux.setup(opt, baseline.model)
    n_samples = length(Y_train)
    n_classes = size(baseline.model.layers[end].weight, 1)
    
    println("Entrenando baseline...")
    
    for epoch in 1:epochs
        shuffle_indices = randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_indices = shuffle_indices[batch_start:batch_end]
            
            loss, grads = Flux.withgradient(baseline.model) do m
                l = 0.0
                for idx in batch_indices
                    x = X_train[idx]
                    y = Flux.onehot(Y_train[idx], 1:n_classes)
                    
                    pred = m(x)
                    l -= sum(y .* log.(pred .+ 1e-8))
                end
                return l / length(batch_indices)
            end
            
            epoch_loss += loss
            n_batches += 1
            
            Flux.update!(st_base, baseline.model, grads[1])
        end
        
        if epoch % 5 == 0 || epoch == 1
            println("  Epoch $epoch: loss = $(round(epoch_loss/n_batches, digits=4))")
        end
    end
    
    return baseline
end

function evaluate_baseline(baseline::Union{BaselineMLP, BaselineCNN},
                           X_test, Y_test::Vector{Int})
    
    n_classes = size(baseline.model.layers[end].weight, 1)
    correct = 0
    
    for (x, y) in zip(X_test, Y_test)
        pred = baseline.model(x)
        pred_class = argmax(pred)
        if pred_class == y
            correct += 1
        end
    end
    
    return correct / length(Y_test)
end

# =============================================================================
# PARTE 7: BENCHMARK COMPLETO CON ESTADÍSTICAS
# =============================================================================

"""
    run_full_benchmark(;n_runs=30, epochs=50)

Ejecuta benchmark completo con múltiples runs para intervalos de confianza.
"""
function run_full_benchmark(;n_runs::Int=30, epochs::Int=50, 
                            n_train::Int=5000, n_test::Int=1000)
    
    println("\n" * "="^80)
    println("BENCHMARK COMPLETO: WDW AutoSymmetry vs Baselines")
    println("="^80)
    println("  Número de runs: $n_runs (para intervalos 95%)")
    println("  Epochs por run: $epochs")
    println("  Dataset: Rotated MNIST ($n_train train, $n_test test)")
    
    # Cargar datos
    dataset = load_rotated_mnist(n_train=n_train, n_test=n_test)
    
    # Preparar datos como vectores
    X_train = [vec(dataset.X_train[:, :, i]) for i in 1:size(dataset.X_train, 3)]
    X_test = [vec(dataset.X_test[:, :, i]) for i in 1:size(dataset.X_test, 3)]
    Y_train = dataset.Y_train
    Y_test = dataset.Y_test
    
    input_dim = 28 * 28
    n_classes = 10
    
    # Resultados de cada run
    wdw_accuracies = Float64[]
    mlp_accuracies = Float64[]
    cnn_accuracies = Float64[]
    
    for run in 1:n_runs
        println("\n" * "-"^60)
        println("RUN $run/$n_runs")
        println("-"^60)
        
        # Set seed para reproducibilidad
        Random.seed!(run * 42)
        
        # === WDW AutoSymmetry ===
        println("\n[WDW AutoSymmetry]")
        wdw_model = WDWAutoSymmetryModel(input_dim, 16, n_classes,
                                          liegan_hidden=[256, 128],
                                          classifier_hidden=[64, 32])
        train_wdw_model!(wdw_model, X_train, Y_train, epochs, lr=0.001)
        acc_wdw = evaluate_wdw_model(wdw_model, X_test, Y_test)
        push!(wdw_accuracies, acc_wdw)
        println("  Accuracy: $(round(acc_wdw*100, digits=2))%")
        
        # === MLP Baseline ===
        println("\n[Baseline MLP]")
        mlp = BaselineMLP(input_dim, n_classes, hidden_dims=[256, 128, 64])
        train_baseline!(mlp, X_train, Y_train, epochs, lr=0.001)
        acc_mlp = evaluate_baseline(mlp, X_test, Y_test)
        push!(mlp_accuracies, acc_mlp)
        println("  Accuracy: $(round(acc_mlp*100, digits=2))%")
        
        # === CNN Baseline ===
        println("\n[Baseline CNN]")
        cnn = BaselineCNN(n_classes)
        # Para CNN, datos deben ser 4D: (H, W, C, N) -> (28, 28, 1, N)
        X_train_cnn = [reshape(X_train[i], 28, 28, 1, 1) for i in 1:length(X_train)]
        X_test_cnn = [reshape(X_test[i], 28, 28, 1, 1) for i in 1:length(X_test)]
        train_baseline!(cnn, X_train_cnn, Y_train, epochs, lr=0.001)
        acc_cnn = evaluate_baseline(cnn, X_test_cnn, Y_test)
        push!(cnn_accuracies, acc_cnn)
        println("  Accuracy: $(round(acc_cnn*100, digits=2))%")
        
        # Liberar memoria
        GC.gc()
    end
    
    # Análisis estadístico
    println("\n" * "="^80)
    println("RESULTADOS ESTADÍSTICOS ($n_runs runs)")
    println("="^80)
    
    stats = statistical_comparison(wdw_accuracies, mlp_accuracies, cnn_accuracies)
    
    return stats
end

"""
    statistical_comparison(wdw_acc, mlp_acc, cnn_acc)

Calcula estadísticas con intervalos de confianza 95%.
"""
function statistical_comparison(wdw_acc::Vector{Float64},
                                mlp_acc::Vector{Float64},
                                cnn_acc::Vector{Float64})
    
    n = length(wdw_acc)
    
    # Medias
    mean_wdw = mean(wdw_acc)
    mean_mlp = mean(mlp_acc)
    mean_cnn = mean(cnn_acc)
    
    # Desviaciones estándar
    std_wdw = std(wdw_acc)
    std_mlp = std(mlp_acc)
    std_cnn = std(cnn_acc)
    
    # Intervalos de confianza 95% (aproximación normal)
    ci95_wdw = 1.96 * std_wdw / sqrt(n)
    ci95_mlp = 1.96 * std_mlp / sqrt(n)
    ci95_cnn = 1.96 * std_cnn / sqrt(n)
    
    # Test t de Student (pareado)
    # H0: mean_wdw == mean_mlp
    t_stat_mlp = (mean_wdw - mean_mlp) / (std(wdw_acc - mlp_acc) / sqrt(n))
    t_stat_cnn = (mean_wdw - mean_cnn) / (std(wdw_acc - cnn_acc) / sqrt(n))
    
    # Cohen's d (tamaño de efecto)
    pooled_std_mlp = sqrt((std_wdw^2 + std_mlp^2) / 2)
    pooled_std_cnn = sqrt((std_wdw^2 + std_cnn^2) / 2)
    cohen_d_mlp = (mean_wdw - mean_mlp) / pooled_std_mlp
    cohen_d_cnn = (mean_wdw - mean_cnn) / pooled_std_cnn
    
    println("\n| Modelo | Mean ± 95% CI | Std | vs WDW Cohen's d |")
    println("|--------|---------------|-----|------------------|")
    println("| WDW    | $(round(mean_wdw*100, digits=2))% ± $(round(ci95_wdw*100, digits=2))% | $(round(std_wdw*100, digits=2))% | - |")
    println("| MLP    | $(round(mean_mlp*100, digits=2))% ± $(round(ci95_mlp*100, digits=2))% | $(round(std_mlp*100, digits=2))% | $(round(cohen_d_mlp, digits=2)) |")
    println("| CNN    | $(round(mean_cnn*100, digits=2))% ± $(round(ci95_cnn*100, digits=2))% | $(round(std_cnn*100, digits=2))% | $(round(cohen_d_cnn, digits=2)) |")
    
    println("\n**Interpretación Cohen's d:**")
    if abs(cohen_d_mlp) < 0.2
        println("  WDW vs MLP: Efecto despreciable")
    elseif abs(cohen_d_mlp) < 0.5
        println("  WDW vs MLP: Efecto pequeño")
    elseif abs(cohen_d_mlp) < 0.8
        println("  WDW vs MLP: Efecto mediano")
    else
        println("  WDW vs MLP: Efecto GRANDE ✓")
    end
    
    println("\n**Claims verificables:**")
    if mean_wdw > mean_mlp + ci95_mlp
        println("  ✓ WDW > MLP (95% confianza)")
    else
        println("  ✗ WDW no supera a MLP significativamente")
    end
    
    if mean_wdw > mean_cnn + ci95_cnn
        println("  ✓ WDW > CNN (95% confianza)")
    else
        println("  ✗ WDW no supera a CNN significativamente")
    end
    
    return Dict(
        :wdw => Dict(:mean => mean_wdw, :std => std_wdw, :ci95 => ci95_wdw),
        :mlp => Dict(:mean => mean_mlp, :std => std_mlp, :ci95 => ci95_mlp),
        :cnn => Dict(:mean => mean_cnn, :std => std_cnn, :ci95 => ci95_cnn),
        :cohen_d_mlp => cohen_d_mlp,
        :cohen_d_cnn => cohen_d_cnn,
        :n_runs => n
    )
end

end  # module AutoSymmetryFlux
