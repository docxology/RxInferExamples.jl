# GNN Integration for RxInferExamples.jl

This directory provides utilities for integrating the [Generalized Notation Notation (GNN)](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation) project with RxInferExamples.jl.

## Overview

GNN provides a standardized notation system for Active Inference generative models, enabling:

- **Text-Based Models**: Human-readable GNN Markdown with mathematical notation
- **Graphical Models**: Factor graphs and network visualizations  
- **Executable Models**: Code generation for PyMDP, **RxInfer.jl**, JAX, and more

## Quick Start

```bash
# Clone and setup GNN repository
python support/gnn/clone_and_setup_gnn.py

# With options
python support/gnn/clone_and_setup_gnn.py --target-dir ./external/gnn --branch main
```

## Scripts

### `clone_and_setup_gnn.py`

Clones and configures the GNN repository for integration.

| Option | Description |
|--------|-------------|
| `--target-dir DIR` | Target directory (default: `./GeneralizedNotationNotation`) |
| `--branch BRANCH` | Git branch to checkout (default: `main`) |
| `--dry-run` | Show actions without executing |
| `--force` | Remove existing and re-clone |
| `--quiet` | Reduce output verbosity |

## RxInfer.jl Bridge

GNN can generate executable RxInfer.jl code from standardized model specifications. The integration enables:

1. **Model Parsing**: Convert GNN notation to RxInfer factor graph definitions
2. **Inference Execution**: Run generated models through RxInfer message passing
3. **Result Analysis**: Process inference outputs with GNN visualization tools

### Example Workflow

```julia
# 1. Parse GNN model specification
# 2. Generate RxInfer.jl model code  
# 3. Execute inference
# 4. Export results back to GNN format
```

## Directory Structure

After running `clone_and_setup_gnn.py`:

```
gnn/
├── clone_and_setup_gnn.py           # Setup script
├── README.md                         # This file
└── GeneralizedNotationNotation/     # Cloned GNN repository
    ├── src/                          # GNN source code
    ├── output/                       # Generated outputs
    └── ...
```

## Related Resources

- [GNN Repository](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation)
- [RxInfer.jl Documentation](https://docs.rxinfer.com)
- [Active Inference Institute](https://activeinference.org)
