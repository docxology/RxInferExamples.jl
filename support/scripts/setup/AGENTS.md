# Environment Setup - Agent Guide

Julia script for comprehensive environment setup and validation.

## File

- `setup.jl` - Main orchestration script

## Quick Command

```bash
julia support/scripts/setup/setup.jl --convert --verify
```

## Key Flags

| Flag | Purpose |
|------|---------|
| `--convert` | Run notebook conversion |
| `--verify` | Verify conversion integrity |
| `--clean` | Clean build cache first |
| `--no-examples` | Skip building examples |
| `--no-docs` | Skip building docs |
