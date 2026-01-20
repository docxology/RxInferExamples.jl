# Agents

This directory contains the **Kalman Filter Agent**, responsible for executing and analyzing Kalman filtering and smoothing experiments.

## Responsibilities
-   **Orchestration**: `KalmanFilterApp.jl` manages the lifecycle of all experiments.
-   **Inference**: `Experiments.jl` and `Models.jl` define and run the probabilistic inference using `RxInfer`.
-   **Verification**: The agent verifies its own output by checking convergence and calculating statistical error metrics (RMSE, MAE).
-   **Reporting**: Detailed execution logs and statistical reports are generated automatically.

## Files
-   `Kalman filtering and smoothing.jl`: Entry point script.
-   `src/`: Source code for the modular agent components.
-   `config.toml`: Configuration file.
-   `meta.jl`: Metadata for the example.

## Interaction
The agent is designed to be run as a standalone script or integrated into a larger pipeline via `run_all()`. It relies on the `RxInfer` ecosystem and follows the project's standard modular architecture.
