# Support Scripts

Executable utility scripts organized by domain.

## Structure

```
scripts/
├── gnn/           # GNN integration tools
├── notebooks/     # Notebook conversion
├── server/        # Python server client  
└── setup/         # Environment setup
```

## Quick Start

```bash
# Full setup with conversion
julia support/scripts/setup/setup.jl --convert --verify

# Convert notebooks only
julia support/scripts/notebooks/notebooks_to_scripts.jl --skip-existing

# Clone GNN repository
python support/scripts/gnn/clone_and_setup_gnn.py
```

## Subdirectories

| Directory | Purpose | Primary Script |
|-----------|---------|----------------|
| [gnn/](gnn/README.md) | GNN repository integration | `clone_and_setup_gnn.py` |
| [notebooks/](notebooks/) | Notebook → script conversion | `notebooks_to_scripts.jl` |
| [server/](server/) | RxInfer Python client | `RxInferClient.py` |
| [setup/](setup/) | Environment orchestration | `setup.jl` |

## Adding New Scripts

1. Create appropriate subdirectory or use existing
2. Add `README.md` if creating new domain
3. Follow existing patterns for CLI flags and logging
4. Update parent AGENTS.md with new script references
