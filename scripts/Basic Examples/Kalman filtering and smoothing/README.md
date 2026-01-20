# Kalman Filtering and Smoothing

This is a comprehensive example of Kalman Filtering and Smoothing using `RxInfer.jl`. It demonstrates modular design, configuration management, and advanced diagnostics.

## Models
The example covers:
1.  **Rotating SSM**: A linear Gaussian state-space model with rotating dynamics.
2.  **System Identification**:
    -   Linear: $y = x + w$
    -   Nonlinear: $y = \min(x, w)$
3.  **Reactive Identification**: Online inference for the nonlinear model.
4.  **Smoothing**: Handling missing data in a noisy sine wave signal.

## Architecture
The code is organized using the **Local Package Pattern**:
-   `src/KalmanFilterApp.jl`: Main application entry point and orchestrator.
-   `src/Experiments.jl`: Experiment definitions and execution logic.
-   `src/Models.jl`: RxInfer model specifications.
-   `src/DataGeneration.jl`: Synthetic data generation.
-   `src/Visualization.jl`: Plotting and visualization.
-   `src/Analysis.jl`: Statistical analysis and diagnostics.
-   `src/LoggingUtils.jl`: Execution logging.
-   `src/Reporting.jl`: Report generation.

## Usage
Run the main script to execute all examples:
```bash
julia --project=. "Kalman filtering and smoothing.jl"
```

Configuration can be adjusted in `config.toml`.
Outputs (plots, logs, reports) are saved to the `output/` directory.
