###############################################################
# One-step prediction of the Lorenz system with an Echo State Network (ESN)
# - Fixed noise standard deviation (noise_std = 1e-8)
# - Sweep the input-weight standard deviation and the spectral
#   radius of the recurrent matrix; evaluate r^2 over seeds
# - Save results as 2D arrays (rows: spectral radius, columns: input std)
###############################################################

using LinearAlgebra, Random, Statistics, SparseArrays, Distributions, DifferentialEquations, SpecialFunctions, ProgressMeter

include("io_array.jl")

# Lorenz system ODE
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Generate a noisy time series from the Lorenz system
function generate_noisy_lorenz(steps::Int, seed::Int; dt=0.2, noise_std=0.1, p=(10.0, 28.0, 8 / 3))
    Random.seed!(seed)
    u0 = randn(3) # initial condition
    discard_steps = 1000
    tspan = (0.0, dt * (steps + discard_steps - 1))
    prob = ODEProblem(lorenz!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=dt)

    X = hcat(sol.u...)'                       # (steps × 3) matrix
    X = X[discard_steps+1:end, :]             # remove initial transient
    X .+= noise_std .* randn(size(X))         # add observation noise
    x_normal = (X[:, 1] .- mean(X[:, 1])) / std(X[:, 1])
    return x_normal
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

function run_lorenz_simulation(recurrent_matrix::SparseMatrixCSC, num_nodes::Int, input_std::Float64, noise_std::Float64, seed::Int)
    # Simulation steps
    num_discard_steps = 200                                  # discard for warmup
    num_training_steps = 5 * num_nodes + num_discard_steps   # total training steps
    num_testing_steps = 2 * num_nodes + num_discard_steps    # total testing steps

    # Generate input weights (1-D input)
    input_weights = generate_input_weights(num_nodes, input_std, seed)
    noise_dist = Normal(0, noise_std)

    # ===== Training phase =====
    lorenz_series = generate_noisy_lorenz(num_training_steps + 1, seed)
    u_train = lorenz_series[1:end-1]   # input sequence
    x = rand(num_nodes)                # initial reservoir state
    training_states = zeros(num_nodes, num_training_steps - num_discard_steps)

    for t in 1:num_training_steps
        noise_vector = rand(noise_dist, num_nodes)
        pre_activation = input_weights * u_train[t] + recurrent_matrix * x + noise_vector
        x = tanh.(pre_activation)
        if t > num_discard_steps
            training_states[:, t-num_discard_steps] = x
        end
    end

    # Build training targets (one-step-ahead)
    train_targets_full = lorenz_series[2:end]     # one step ahead of u_train
    train_targets = train_targets_full[num_discard_steps+1:end]

    # Linear regression with a bias term to learn readout weights
    X_train = vcat(ones(1, size(training_states, 2)), training_states)
    readout_weights = X_train' \ train_targets

    # ===== Testing phase =====
    lorenz_series = generate_noisy_lorenz(num_testing_steps + 1, seed + 1000)
    u_test = lorenz_series[1:end-1]   # input sequence
    x = rand(num_nodes)               # re-initialize reservoir state
    testing_states = zeros(num_nodes, num_testing_steps - num_discard_steps)
    for t in 1:num_testing_steps
        noise_vector = rand(noise_dist, num_nodes)
        pre_activation = input_weights * u_test[t] + recurrent_matrix * x + noise_vector
        x = tanh.(pre_activation)
        if t > num_discard_steps
            testing_states[:, t-num_discard_steps] = x
        end
    end

    test_targets_full = lorenz_series[2:end]     # one step ahead of u_test
    test_targets = test_targets_full[num_discard_steps+1:end]

    X_test = vcat(ones(1, size(testing_states, 2)), testing_states)
    model_output = X_test' * readout_weights

    # Metrics: squared Pearson correlation (r^2)
    r2 = cor(test_targets, model_output)^2

    return r2
end

# --- Main ---

# Grid parameters
input_std_grid = 10.0 .^ (-4.5:0.05:0.0)    # grid of input-weight standard deviations
spectral_radius_grid = 0.1:0.025:1.2        # grid of spectral radii for the recurrent matrix

# Fixed ESN parameters
num_reservoir_nodes = 500
connectivity = 0.1
noise_std = 1e-8

# Buffer for results (rows: spectral radius, columns: input std)
r2_results = zeros(length(spectral_radius_grid), length(input_std_grid))

for seed in 1:10
    println("seed = ", seed)
    @showprogress for (i, spec_radius) in enumerate(spectral_radius_grid)
        for (j, input_std) in enumerate(input_std_grid)
            rec_matrix = generate_recurrent_matrix(num_reservoir_nodes, connectivity, spec_radius, seed)
            r2 = run_lorenz_simulation(rec_matrix, num_reservoir_nodes, input_std, noise_std, seed)
            r2_results[i, j] = r2
        end
    end
    # --- Write out the r^2 map for this seed (2D array) ---
    write_to_file("r2_map_N=$(num_reservoir_nodes)_seed=$(seed).txt", r2_results)
end
