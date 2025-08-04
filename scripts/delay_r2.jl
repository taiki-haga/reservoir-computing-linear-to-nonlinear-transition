###############################################################
# Delay task with an Echo State Network (ESN)
# - Fix the spectral radius (spec_radius = 0.8)
# - Fix the noise standard deviation (noise_std = 1e-8 or as set below)
# - Sweep the input-weight standard deviation and evaluate r^2
#   (correlation coefficient squared); run multiple seeds
# - Save each seed's r^2 values as a 1D array to a text file
###############################################################

using LinearAlgebra, Random, Statistics, SparseArrays, Distributions, ProgressMeter

include("io_array.jl")

# Generate the target sequence for the delay task
function compute_delay_target(input_data::Vector{Int}, delay::Int)
    N = length(input_data)
    targets = zeros(Int, N)
    for i in delay+1:N
        targets[i] = input_data[i-delay]
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

# Simulation of the delay task with an ESN
function run_delay_simulation(recurrent_matrix::SparseMatrixCSC, num_nodes::Int, input_std::Float64, noise_std::Float64, delay::Int, seed::Int)
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
    train_targets_full = compute_delay_target(u_train, delay)
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

    test_targets_full = compute_delay_target(u_test, delay)
    test_targets = test_targets_full[num_discard_steps+1:end]

    X_test = vcat(ones(1, size(testing_states, 2)), testing_states)
    model_output = X_test' * readout_weights

    # Metrics: squared Pearson correlation (r^2)
    target = Float64.(test_targets)
    predictions = vec(model_output)
    r2 = cor(target, predictions)^2

    return r2
end

# --- Main ---

# Grid for input-weight standard deviation
input_std_grid = 10.0 .^ (-3:0.05:1.0)

# Fixed ESN parameters
num_reservoir_nodes = 600
connectivity = 0.1
noise_std = 1e-8
spec_radius = 0.8
delay = 30

# Buffer to store per-seed r^2 across the grid
r2_results = zeros(length(input_std_grid))

for seed in 1:10
    println("    seed = ", seed)
    @showprogress for (i, input_std) in enumerate(input_std_grid)
        rec_matrix = generate_recurrent_matrix(num_reservoir_nodes, connectivity, spec_radius, seed)
        r2 = run_delay_simulation(rec_matrix, num_reservoir_nodes, input_std, noise_std, delay, seed)
        r2_results[i] = r2
    end
    # --- Write results for this seed ---
    write_to_file("r2_N=$(num_reservoir_nodes)_seed=$(seed).txt", r2_results)
end
