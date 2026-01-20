# Scripts - Agent Guide

Utility scripts for RxInferExamples.jl support operations.

## Domains

- **[gnn/](gnn/)** - GNN integration (Python)
- **[notebooks/](notebooks/)** - Notebook conversion (Julia)
- **[server/](server/)** - RxInfer client (Python)
- **[setup/](setup/)** - Environment setup (Julia)

## Key Entry Points

| Task | Command |
|------|---------|
| Full Setup | `julia support/scripts/setup/setup.jl --convert --verify` |
| Convert Notebooks | `julia support/scripts/notebooks/notebooks_to_scripts.jl --skip-existing` |
| Clone GNN | `python support/scripts/gnn/clone_and_setup_gnn.py` |

## Language Distribution

- **Julia**: `setup.jl`, `notebooks_to_scripts.jl`
- **Python**: `clone_and_setup_gnn.py`, `RxInferClient.py`
