# reservoir-computing-linear-to-nonlinear-transition

This repository contains code, data, and notebooks for our study of phase transitions from linear to nonlinear information processing in echo-state networks (ESNs).

- **Paper (PDF in this repo):** [paper.pdf](https://github.com/taiki-haga/reservoir-computing-linear-to-nonlinear-transition/blob/main/paper.pdf)

- **External link:**  https://arxiv.org/abs/2505.13003

------

### Scripts

The `scripts/` directory contains Julia scripts to reproduce data used in the paper.

- **`io_array.jl`** : Utilities for reading/writing numeric arrays to text/CSV.
- **`xor_r2.jl`** : Evaluate XOR task performance (squared Pearson correlation $r^2$) over a grid of input-weight standard deviations $\sigma_{\text{in}}$.
- **`xor_r2_map.jl`** : Sweep both $\sigma_{\text{in}}$ and spectral radius $\rho$; save 2-D $r^2$ maps.
- **`xor_r2_delay_parallel.jl`** : Train one reservoir and learn multiple XOR readouts with different delays in parallel; save $r^2$ vs $\sigma_{\text{in}}$ for each delay.
- **`lyapunov_map.jl`** : Simulate the reservoir and estimate the largest Lyapunov exponent across $\sigma_{\text{in}}$ and $\rho$.
- **`narma_r2.jl`** : Evaluate NARMA task performance; save $r^2$ vs $\sigma_{\text{in}}$.
- **`narma_r2_map.jl`** : Sweep both $\sigma_{\text{in}}$ and $\rho$; save 2-D $r^2$ maps.
- **`delay_r2.jl`** : Evaluate delay task performance; save $r^2$ vs $\sigma_{\text{in}}$.
- **`delay_r2_map.jl`** : Sweep both $\sigma_{\text{in}}$ and $\rho$; save 2-D $r^2$ maps.
- **`lorenz_r2.jl`** — Generate a noisy Lorenz time series and evaluate one-step prediction $x(t)\!\to\!x(t+1)$; save $r^2$ vs $\sigma_{\text{in}}$.
- **`lorenz_r2_map.jl`** : Sweep both $\sigma_{\text{in}}$ and $\rho$; save 2-D $r^2$ maps.

----

### Data

The `data/` directory contains data used in the paper.

#### XOR task

- **`r2.zip`** : $r^2$ vs $\sigma_{\text{in}}$ with different node numbers.
- **`r2_map.zip`** : 2-D $r^2$ maps with respect to $\sigma_{\text{in}}$ and $\rho$.
- **`r2_noise_scaling.zip`** : $r^2$ vs $\sigma_{\text{in}}$ with different noise intensities.
- **`r2_finite_size_scaling.zip`** : $r^2$ vs $\sigma_{\text{in}}$ with different node numbers and delays.
- **`lyapunov_map.zip`** : Largest Lyapunov exponent with respect to $\sigma_{\text{in}}$ and $\rho$.

#### NARMA task

- **`narma_r2.zip`** : $r^2$ vs $\sigma_{\text{in}}$ with different node numbers.
- **`narma_r2_map.zip`** : 2-D $r^2$ maps with respect to $\sigma_{\text{in}}$ and $\rho$.
- **`narma_r2_noise_scaling.zip`** : $r^2$ vs $\sigma_{\text{in}}$ with different noise intensities.

#### Delay task

- **`delay_r2.zip`** : $r^2$ vs $\sigma_{\text{in}}$ with different node numbers.
- **`delay_r2_map.zip`** : 2-D $r^2$ maps with respect to $\sigma_{\text{in}}$ and $\rho$.
- **`delay_r2_noise_scaling.zip`** : $r^2$ vs $\sigma_{\text{in}}$ with different noise intensities.

#### Lorenz task

- **`lorenz_r2.zip`** : $r^2$ vs $\sigma_{\text{in}}$ with different node numbers.
- **`lorenz_r2_map.zip`** : 2-D $r^2$ maps with respect to $\sigma_{\text{in}}$ and $\rho$.
- **`lorenz_r2_noise_scaling.zip`** : $r^2$ vs $\sigma_{\text{in}}$ with different noise intensities.

#### Appendix

- **`activation_erf_noise_scaling.zip`** : $r^2$ vs $\sigma_{\text{in}}$ for the XOR task with the error function activation.
- **`activation_fifth_noise_scaling.zip`** : $r^2$ vs $\sigma_{\text{in}}$ for the XOR task with a fifth-order activation function.
- **`activation_PL_noise_scaling.zip`** : $r^2$ vs $\sigma_{\text{in}}$ for the XOR task with a piecewise linear activation function.
- **`ridge_regression.zip`** : $r^2$ vs $\sigma_{\text{in}}$ for the XOR task using the ridge regression.
- **`rounding_error_test.zip`** : $r^2$ vs $\sigma_{\text{in}}$ for the XOR task with different numerical precisions.

----

### Notebooks

The `notebooks/` directory contains analysis/visualization notebooks (e.g., phase diagrams, scaling plots).

- **`fig2_r2_map_plot.ipynb`** : Plot the 2-D $r^2$ maps with respect to $\sigma_{\text{in}}$ and $\rho$.
- **`fig3_noise_scaling_plot.ipynb`** : Plot $r^2$ vs $\sigma_{\text{in}}$ with different noise intensities.
- **`fig4_finite_size_scaling_plot.ipynb`** : Plot the transition point $\sigma_{\text{in},0}$ and the sharpness of transition $\Delta r^2_{\text{max}}$ with respect to the node number $N$.
- **`fig5_narma_delay_plot.ipynb`** : Plot the 2-D $r^2$ maps for the NARMA, delay, and Lorenz tasks.
- **`fig6_ridge_noise_scaling_plot.ipynb`** : Plot $r^2$ vs $\sigma_{\text{in}}$ using the ridge regression with different regularization parameters.
- **`fig7_narma_delay_noise_scaling_plot.ipynb`** : Plot $r^2$ vs $\sigma_{\text{in}}$ for the NARMA, delay, and Lorenz tasks with different noise intensities.
- **`fig8_activation_noise_scaling_plot.ipynb`** : Plot $r^2$ vs $\sigma_{\text{in}}$ for different activation functions and noise intensities.
- **`fig9_rounding_error_plot.ipynb`** : Plot $r^2$ vs $\sigma_{\text{in}}$ for different numerical precisions.

----






