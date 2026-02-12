# RxInferExamples.jl Support Utilities

Enhanced fork utilities for environment management, workflow automation, and research development.

## Quick Start

```bash
# Full setup with notebook conversion
julia support/scripts/setup/setup.jl --convert --verify

# Convert notebooks only
julia support/scripts/notebooks/notebooks_to_scripts.jl --skip-existing
```

## Structure

The support sidecar is organized by responsibility to ensure modularity and ease of testing:

- **`scripts/`**: **Thin Orchestrators**. Domain-organized executable scripts (CLI, workflow) that handle user input and orchestration. They should remain "thin" and delegate logic to `src/`.
- **`src/`**: **Core Modules**. testable, reusable library code.
  - **`julia/`**: Julia utilities (Command, File, Config, Stats, etc.)
  - **`python/`**: Python utilities (File, Config, Stats, Logging, etc.)
- **`docs/`**: Reference documentation for RxInfer.jl patterns.
- **`tests/`**: Suite-level verification for all support modules.
- **`notes/`**: Temporary research notes and scratchpad.

```mermaid
graph TD
    support/ --> scripts/
    support/ --> src/
    support/ --> docs/
    support/ --> tests/
    support/ --> notes/
end
```

| Feature | Script | Module |
| ----------- | ------ | ------ |
| **Environment Setup** | [setup.jl](scripts/setup/setup.jl) | `utils/CommandUtils`, `environment/EnvironmentSetup` |
| **Notebook Conversion** | [notebooks_to_scripts.jl](scripts/notebooks/notebooks_to_scripts.jl) | `environment/NotebookConversion` |
| **GNN Integration** | [clone_and_setup_gnn.py](scripts/gnn/clone_and_setup_gnn.py) | `integration/gnn_utils` |
| **Server Client** | [RxInferClient.py](scripts/server/RxInferClient.py) | `integration/server_utils` |

## Usage

### Julia

```julia
include("support/src/julia/Support.jl")
using .Support

# Use utilities
ensure_directory("output")
config = load_toml_config("config.toml")
logger = setup_logger("output", "config.toml", 1234)
```

### Python

```python
from support.src.python import utils, statistics, logging

# Use utilities
utils.ensure_directory("output")
config = utils.load_toml("config.toml")
logger = logging.setup_logger("output", "config.toml", 1234)
```

## Documentation Hierarchy

- [Root README](../README.md) - Repository overview
- [AGENTS.md](AGENTS.md) - Agent signposting
- [src/README.md](src/README.md) - Source modules overview
- [src/julia/README.md](src/julia/README.md) - Julia modules
- [src/python/README.md](src/python/README.md) - Python modules

## Resources

- [RxInfer.jl Documentation](https://docs.rxinfer.com)
- [RxInfer Examples](https://examples.rxinfer.com)
- [Contribution Guide](https://examples.rxinfer.com/how_to_contribute/)
