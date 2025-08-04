###############################################################
# Compute the Lyapunov exponent with an Echo State Network (ESN)
# - Simulate the reservoir and record pre-activation states
# - Use the derivative of tanh (1 - tanh(x)^2) to build the
#   time-varying propagation matrix and accumulate its product
# - Sweep input-weight std and spectral radius
# - Save results as 2D arrays (rows: spectral radius, columns: input std)
###############################################################

using LinearAlgebra, Random, Statistics, SparseArrays, Distributions, ProgressMeter

include("io_array.jl")

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

# Simulate reservoir dynamics and record pre-activation states
function simulate_reservoir_for_lyapunov(recurrent_matrix::SparseMatrixCSC, num_nodes::Int, input_std::Float64, noise_std::Float64, num_steps::Int, seed::Int)
    discard_steps = 200
    total_steps = num_steps + discard_steps
    # Generate a binary input sequence
    Random.seed!(seed)
    u = rand(0:1, total_steps)
    # Input weights
    input_weights = generate_input_weights(num_nodes, input_std, seed)
    noise_dist = Normal(0, noise_std)
    # Initial reservoir state
    x = rand(num_nodes)
    # Pre-activation states
    x_pre_matrix = zeros(num_nodes, num_steps)
    for t in 1:total_steps
        noise_vector = rand(noise_dist, num_nodes)
        x_pre = input_weights * u[t] + recurrent_matrix * x + noise_vector
        x = tanh.(x_pre)
        if t > discard_steps
            x_pre_matrix[:, t-discard_steps] = x_pre
        end
    end
    return x_pre_matrix
end

# Compute the Lyapunov exponent
# x_pre: matrix of pre-activation states (size: num_nodes x num_steps)
function compute_lyapunov_exponent(x_pre::Matrix{Float64}, recurrent_matrix::SparseMatrixCSC)
    num_nodes = size(x_pre, 1)
    time_steps = size(x_pre, 2)
    # D: diagonal matrix whose entries are the local derivatives at each time
    D = zeros(Float64, num_nodes, num_nodes)
    # E: product of perturbation propagation matrices (initialized as identity)
    E = Matrix{Float64}(I, num_nodes, num_nodes)
    temp = similar(E)
    for t in 1:time_steps
        for i in 1:num_nodes
            # Derivative of tanh is 1 - tanh(x)^2
            D[i, i] = 1 - tanh(x_pre[i, t])^2
        end
        mul!(temp, D * recurrent_matrix, E)
        E .= temp
    end
    # Average over multiple random initial perturbations
    num_trials = 20
    lamda_trials = zeros(Float64, num_trials)
    x0 = zeros(Float64, num_nodes)
    for trial in 1:num_trials
        Random.seed!(trial)
        rand!(x0)
        lamda_trials[trial] = (1 / time_steps) * log(norm(E * x0) / norm(x0))
    end
    return mean(lamda_trials)
end

# --- Main ---

# Grid parameters
input_std_grid = 10.0 .^ (-3:0.05:0.5)   # grid of input-weight standard deviations
spectral_radius_grid = 0.9:0.01:1.3      # grid of spectral radii for the recurrent matrix

# Fixed ESN parameters
num_reservoir_nodes = 1000
connectivity = 0.10
noise_std = 1e-8
simulation_steps = 100 # number of effective steps for Lyapunov calculation
discard_steps = 200

# Buffer for results (rows: spectral radius, columns: input std)
lyapunov_results = zeros(length(spectral_radius_grid), length(input_std_grid))

for seed in 1:5
    println("seed = ", seed)
    @showprogress for (i, spec_radius) in enumerate(spectral_radius_grid)
        for (j, input_std) in enumerate(input_std_grid)
            rec_matrix = generate_recurrent_matrix(num_reservoir_nodes, connectivity, spec_radius, seed)
            x_pre_matrix = simulate_reservoir_for_lyapunov(rec_matrix, num_reservoir_nodes, input_std, noise_std, simulation_steps, seed)
            lyapunov_results[i, j] = compute_lyapunov_exponent(x_pre_matrix, rec_matrix)
        end
    end
    # --- Write results for this seed ---
    write_to_file("lyapunov_map_N=$(num_reservoir_nodes)_seed=$(seed).txt", lyapunov_results)
end
