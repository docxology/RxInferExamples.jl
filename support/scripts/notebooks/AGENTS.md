# Notebook Conversion - Agent Guide

Julia script for converting Jupyter notebooks to executable scripts.

## File

- `notebooks_to_scripts.jl` - Main conversion engine

## Quick Command

```bash
julia support/scripts/notebooks/notebooks_to_scripts.jl --skip-existing --verify
```

## Key Flags

| Flag | Purpose |
|------|---------|
| `--skip-existing` | Incremental mode |
| `--force` | Reconvert all |
| `--verify` | Check 1:1 mapping |
| `--filter STR` | Filter by name |
