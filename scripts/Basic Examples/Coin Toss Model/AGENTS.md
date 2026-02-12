# AGENTS.md - Coin Toss Model

## Context

 Is a basic example of using `RxInfer.jl` for conjugate Bayesian inference. It serves as a template for "Maximal Modularity," demonstrating recommended patterns for structure, configuration, and logging in this repository.

## Key Files

- `Coin Toss Model.jl`: Main entry point.
- `config.toml`: Configuration parameters.
- `Project.toml`: Local dependencies.
- `meta.jl`: Metadata for the examples gallery.

## Patterns Used

- **Support Module**: Uses `../../../../support/src/julia/Support.jl` for shared utilities.
- **ConfigUtils**: For TOML loading.
- **LoggingUtils**: For structured logging.
- **ValidationUtils**: For automated correctness checks.
- **ReportingUtils**: For markdown report generation.

## Maintainer Notes

- When updating `RxInfer` versions, verify that the `infer` call signature remains compatible.
- Ensure `Support` module path is correctly resolved relative to the script.
