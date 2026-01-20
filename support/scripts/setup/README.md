# Environment Setup

Julia script for comprehensive environment setup and validation.

## Usage

```bash
# Recommended: Full setup with conversion and verification
julia support/scripts/setup/setup.jl --convert --verify

# Quick setup without examples/docs
julia support/scripts/setup/setup.jl --no-examples --no-docs

# Clean build first
julia support/scripts/setup/setup.jl --clean --convert --verify
```

## Options

| Flag | Description |
|------|-------------|
| `--convert` | Run notebook-to-script conversion (incremental) |
| `--convert-all` | Force conversion of all notebooks |
| `--verify` | Verify conversion integrity |
| `--clean` | Clean build cache before setup |
| `--no-examples` | Skip building examples |
| `--no-docs` | Skip building documentation |
| `--no-preview` | Don't start documentation preview server |
| `--force` | Continue on non-critical errors |
| `--quiet` | Reduce output verbosity |

## What It Does

1. **Environment Check**: Verifies running from repository root
2. **Julia Update**: Updates Julia via juliaup (if available)
3. **Package Installation**: Installs required packages (Weave, ImageInTerminal, JSON)
4. **Project Setup**: Initializes and updates `examples/` environment
5. **Notebook Conversion**: Optionally converts notebooks to scripts
6. **Build Examples**: Executes `make examples` (unless skipped)
7. **Build Documentation**: Executes `make docs` (unless skipped)
8. **Preview**: Optionally starts documentation preview server

## Prerequisites

- Julia 1.6+
- juliaup (recommended for Julia version management)
- Git (for repository operations)
