# Coin Toss Model (Beta-Bernoulli)

A fundamental example of Bayesian inference using **RxInfer.jl**. This script demonstrates how to estimate the bias of a coin (probability of heads $\theta$) given a sequence of coin flips.

## Model

 The model consists of:

- **Prior**: $\theta \sim \text{Beta}(a, b)$
- **Likelihood**: $y_i \sim \text{Bernoulli}(\theta)$

 This is a **conjugate model**, meaning the posterior is also a Beta distribution:
 $$ P(\theta | y) = \text{Beta}(a + \sum y_i, b + n - \sum y_i) $$

## Features

- **Modular Architecture**: Clean separation of data generation, inference, and reporting.
- **Configurable**: All parameters are controlled via `config.toml`.
- **Advanced Visualization**:
  - Static posterior plot (`inference_results.png`).
  - Dynamic belief update animation (`posterior_animation.gif`).
- **Automated Reporting**: Generates a Markdown analysis report (`analysis_report.txt`).
- **Validation**: Automated checks against ground truth values.
- **Logging**: Structured logging to console and file.

## Usage

 Run the script from the project root or the script directory:

 ```bash
 julia --project="scripts/Basic Examples/Coin Toss Model" "scripts/Basic Examples/Coin Toss Model/Coin Toss Model.jl"
 ```

 You can also specify a custom configuration file:

 ```bash
 julia --project="scripts/Basic Examples/Coin Toss Model" "scripts/Basic Examples/Coin Toss Model/Coin Toss Model.jl" --config path/to/config.toml
 ```

## Configuration (`config.toml`)

 | Section | Key | Description |
 | :--- | :--- | :--- |
 | **model** | `n_samples` | Number of coin flips. |
 | | `theta_real` | True probability of heads. |
 | | `prior_a`, `prior_b` | Hyperparameters for the Beta prior. |
 | **output** | `save_plots` | Save static plots? |
 | | `save_animation` | Generate GIF? |
 | | `save_report` | Generate report? |
 | | `directory` | Output directory name. |
 | **validation** | `error_threshold` | Max allowed error for pass/fail. |
