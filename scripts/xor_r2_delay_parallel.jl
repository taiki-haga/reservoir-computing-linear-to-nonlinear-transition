###############################################################
# XOR task with an Echo State Network (ESN) — multi-delay readouts
# - Fixed spectral radius (spec_radius = 0.8) and noise (noise_std = 1e-8)
# - Sweep the input-weight standard deviation (sigma_in)
# - Train a single reservoir and learn multiple readouts in parallel
#   for XOR targets with delays k = 1..25 (shared reservoir states)
# - Evaluate r^2 on a test sequence; repeat over seeds
# - Save, for each delay, a 1D array of r^2 values across sigma_in
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

# ESN simulation for the XOR task with multiple delays learned in parallel
function run_xor_simulation_delay_parallel(recurrent_matrix::SparseMatrixCSC, num_nodes::Int, input_std::Float64, noise_std::Float64, xor_delay_list::Vector{Int}, seed::Int)
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

    # Compute Moore–Penrose pseudoinverse once (shared across delays)
    X_train = vcat(ones(1, size(training_states, 2)), training_states)
    inv_X_train = pinv(X_train')

    # Learn readout weights for each XOR delay
    readout_weights_list = []
    for xor_delay in xor_delay_list
        # Build training targets
        train_targets_full = compute_xor_target(collect(u_train), xor_delay)
        train_targets = train_targets_full[num_discard_steps+1:end]
        # Linear regression via pseudoinverse
        readout_weights = inv_X_train * train_targets
        push!(readout_weights_list, readout_weights)
    end

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

    # Compute r^2 for each XOR delay
    r2_list = Float64[]
    for (i, xor_delay) in enumerate(xor_delay_list)
        # Build test targets for this delay
        test_targets_full = compute_xor_target(collect(u_test), xor_delay)
        test_targets = test_targets_full[num_discard_steps+1:end]
        # Model output for this delay
        X_test = vcat(ones(1, size(testing_states, 2)), testing_states)
        model_output = X_test' * readout_weights_list[i]

        # Performance: squared Pearson correlation r^2
        target = Float64.(test_targets)
        predictions = vec(model_output)
        r2 = cor(target, predictions)^2
        push!(r2_list, r2)
    end

    return r2_list
end

# --- Main ---

# Grid for input-weight standard deviation
input_std_grid = 10.0 .^ (-4:0.05:0.0)

# Fixed ESN parameters
num_reservoir_nodes = 1000
connectivity = 0.1
spec_radius = 0.8
noise_std = 1e-8

xor_delay_list = collect(1:25)

println("N = ", num_reservoir_nodes)
for seed in 1:10
    println("  seed = ", seed)
    r2_list = zeros(length(xor_delay_list), length(input_std_grid))
    rec_matrix = generate_recurrent_matrix(num_reservoir_nodes, connectivity, spec_radius, seed)
    @showprogress for (i, input_std) in enumerate(input_std_grid)
        r2_list[:, i] = run_xor_simulation_delay_parallel(rec_matrix, num_reservoir_nodes, input_std, noise_std, xor_delay_list, seed)
    end
    # --- Save results ---
    for (i, xor_delay) in enumerate(xor_delay_list)
        # Create output directory if not exists
        dir = "data//r2_finite_size_scaling//N=$(num_reservoir_nodes)//delay=$(xor_delay)"
        if isdir(dir) != 1
            mkdir(dir)
        end
        # Write r^2 values (as a 1D array over sigma_in) for this delay
        write_to_file("data//r2_finite_size_scaling//N=$(num_reservoir_nodes)//delay=$(xor_delay)//r2_N=$(num_reservoir_nodes)_delay=$(xor_delay)_seed=$(seed).txt", r2_list[i, :])
    end
end
