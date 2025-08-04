###############################################################
# XOR task with an Echo State Network (ESN)
# - Fixed delay for XOR labels (delay = 10)
# - Fixed noise standard deviation (noise_std = 1e-8)
# - Sweep the input-weight standard deviation and the spectral
#   radius of the recurrent matrix; evaluate r^2 over seeds
# - Save results as 2D arrays (rows: spectral radius, columns: input std)
###############################################################

using LinearAlgebra, Random, Statistics, SparseArrays, Distributions, ProgressMeter

include("io_array.jl")

# Generate the target sequence for the XOR task
function compute_xor_target(input_data::Vector{Int}, delay::Int)
    N = length(input_data)
    targets = zeros(Int, N)
    for i in delay+1:N
        targets[i] = (input_data[i] == input_data[i-delay] ? 0 : 1)
    end
    return targets
end

# Generate input-weight matrix (input dimension is fixed to 1)
function generate_input_weights(num_nodes::Int, input_std::Float64, seed::Int)
    Random.seed!(seed)
    scale = sqrt(3) * input_std
    return rand(num_nodes, 1) .* (2 * scale) .- scale
end

# Generate recurrent (reservoir) matrix as a sparse matrix
function generate_recurrent_matrix(num_nodes::Int, connectivity::Float64, spec_radius::Float64, seed::Int)
    Random.seed!(seed)
    W = zeros(num_nodes, num_nodes)
    for i in 1:num_nodes, j in 1:num_nodes
        if rand() < connectivity
            W[i, j] = 2.0 * rand() - 1.0
        end
    end
    current_radius = maximum(abs.(eigvals(W)))
    W = spec_radius * W / current_radius
    return sparse(W)
end

# Simulation of the XOR task with an ESN
function run_xor_simulation(recurrent_matrix::SparseMatrixCSC, num_nodes::Int, input_std::Float64, noise_std::Float64, xor_delay::Int, seed::Int)
    # Simulation steps
    num_discard_steps = 200                                  # discard for warmup
    num_training_steps = 5 * num_nodes + num_discard_steps   # total training steps
    num_testing_steps = 2 * num_nodes + num_discard_steps    # total testing steps

    # Generate input weights (1-D input)
    input_weights = generate_input_weights(num_nodes, input_std, seed)
    noise_dist = Normal(0, noise_std)

    # ===== Training phase =====
    Random.seed!(seed)
    u_train = rand(0:1, num_training_steps)   # binary input for training
    x = rand(num_nodes)                       # initial reservoir state
    training_states = zeros(num_nodes, num_training_steps - num_discard_steps)

    for t in 1:num_training_steps
        noise_vector = rand(noise_dist, num_nodes)
        pre_activation = input_weights * u_train[t] + recurrent_matrix * x + noise_vector
        x = tanh.(pre_activation)
        if t > num_discard_steps
            training_states[:, t-num_discard_steps] = x
        end
    end

    # Build training targets
    train_targets_full = compute_xor_target(collect(u_train), xor_delay)
    train_targets = train_targets_full[num_discard_steps+1:end]

    # Linear regression with a bias term to learn readout weights
    X_train = vcat(ones(1, size(training_states, 2)), training_states)
    readout_weights = X_train' \ train_targets

    # ===== Testing phase =====
    Random.seed!(seed + 1000)  # change seed for testing
    u_test = rand(0:1, num_testing_steps)    # binary input for testing
    x = rand(num_nodes)                      # re-initialize reservoir state
    testing_states = zeros(num_nodes, num_testing_steps - num_discard_steps)
    for t in 1:num_testing_steps
        noise_vector = rand(noise_dist, num_nodes)
        pre_activation = input_weights * u_test[t] + recurrent_matrix * x + noise_vector
        x = tanh.(pre_activation)
        if t > num_discard_steps
            testing_states[:, t-num_discard_steps] = x
        end
    end

    test_targets_full = compute_xor_target(collect(u_test), xor_delay)
    test_targets = test_targets_full[num_discard_steps+1:end]

    X_test = vcat(ones(1, size(testing_states, 2)), testing_states)
    model_output = X_test' * readout_weights

    # Metrics: RMSE and squared Pearson correlation (r^2)
    target = Float64.(test_targets)
    predictions = vec(model_output)
    rmse = sqrt(mean((target .- predictions) .^ 2))
    r2 = cor(target, predictions)^2

    return rmse, r2
end

# --- Main ---

# Grid parameters
input_std_grid = 10.0 .^ (-3:0.05:0.5)   # grid of input-weight standard deviations
spectral_radius_grid = 0.1:0.025:1.2     # grid of spectral radii for the recurrent matrix

# Fixed ESN parameters
num_reservoir_nodes = 200
connectivity = 0.1
noise_std = 1e-8
xor_delay = 10

# Buffers for results (rows: spectral radius, columns: input std)
rmse_results = zeros(length(spectral_radius_grid), length(input_std_grid))
r2_results = zeros(length(spectral_radius_grid), length(input_std_grid))

# Run simulations for each grid point and each seed
for seed in 1:10
    println("seed = ", seed)
    @showprogress for (i, spec_radius) in enumerate(spectral_radius_grid)
        for (j, input_std) in enumerate(input_std_grid)
            rec_matrix = generate_recurrent_matrix(num_reservoir_nodes, connectivity, spec_radius, seed)
            rmse, r2 = run_xor_simulation(rec_matrix, num_reservoir_nodes, input_std, noise_std, xor_delay, seed)
            rmse_results[i, j] = rmse
            r2_results[i, j] = r2
        end
    end
    # --- Write out the r^2 map for this seed (2D array) ---
    write_to_file("r2_map_N=$(num_reservoir_nodes)_seed=$(seed).txt", r2_results)
end
