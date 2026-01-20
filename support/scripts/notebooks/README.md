# Notebook Conversion

Julia script for converting Jupyter notebooks to executable Julia scripts.

## Usage

```bash
# Convert with incremental mode (skip up-to-date)
julia support/scripts/notebooks/notebooks_to_scripts.jl --skip-existing

# Force convert all notebooks
julia support/scripts/notebooks/notebooks_to_scripts.jl --force

# Verify 1:1 mapping after conversion
julia support/scripts/notebooks/notebooks_to_scripts.jl --verify

# Convert specific notebooks
julia support/scripts/notebooks/notebooks_to_scripts.jl --filter "Coin Toss" --force
```

## Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Show planned conversions without executing |
| `--skip-existing` | Skip files newer than source notebooks |
| `--force` | Reconvert all targeted notebooks |
| `--filter SUBSTR` | Convert only notebooks containing SUBSTR |
| `--verify` | Verify 1:1 mapping after conversion |
| `--quiet` | Reduce output verbosity |
| `--list` | List notebooks that would be processed |

## How It Works

1. Recursively walks `examples/` directory
2. Parses notebook JSON structure
3. Extracts code cells (filters empty/comment-only)
4. Generates `.jl` files in mirrored `scripts/` structure
5. Copies `Project.toml` and `meta.jl` files
6. Optionally verifies 1:1 notebook/script mapping

## Output

- Scripts written to: `scripts/<category>/<example>/`
- Preserves directory hierarchy from `examples/`
- Includes auto-generated header with source reference
