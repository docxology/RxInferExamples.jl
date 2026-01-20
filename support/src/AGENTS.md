# Source Modules - Agent Guide

Reusable modules imported by orchestration scripts.

## Architecture

- **scripts/** → thin orchestrators (CLI parsing, workflow control)
- **src/** → core functionality (testable, reusable methods)
  - **julia/** → Julia modules
  - **python/** → Python modules

## Julia Modules

| Module | Location | Used By |
|--------|----------|---------|
| `CommandUtils.jl` | `julia/` | `scripts/setup/` |
| `EnvironmentSetup.jl` | `julia/` | `scripts/setup/` |
| `NotebookConversion.jl` | `julia/` | `scripts/notebooks/` |

## Python Modules

| Module | Location | Used By |
|--------|----------|---------|
| `gnn_utils.py` | `python/` | `scripts/gnn/` |
| `server_utils.py` | `python/` | `scripts/server/` |
