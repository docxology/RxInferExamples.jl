# Support Source Modules

Reusable modules for RxInferExamples.jl support utilities.

Organized into language-specific subfolders for better modularity and expansion.

## Directory Structure

```
src/
├── julia/           # Julia modules
│   ├── CommandUtils.jl
│   ├── EnvironmentSetup.jl
│   └── NotebookConversion.jl
├── python/          # Python modules
│   ├── __init__.py
│   ├── gnn_utils.py
│   └── server_utils.py
├── AGENTS.md
└── README.md
```

## Julia Modules

| Module | Purpose | Exports |
|--------|---------|---------|
| [CommandUtils.jl](julia/CommandUtils.jl) | Command execution | `run_command`, `parse_flag_args`, `log_info` |
| [EnvironmentSetup.jl](julia/EnvironmentSetup.jl) | Environment management | `install_required_packages`, `setup_environment_vars` |
| [NotebookConversion.jl](julia/NotebookConversion.jl) | Notebook → script | `extract_code_from_notebook`, `convert_notebook` |

## Python Modules

| Module | Purpose | Functions |
|--------|---------|-----------|
| [gnn_utils.py](python/gnn_utils.py) | GNN integration | `clone_repository`, `verify_clone`, `setup_integration` |
| [server_utils.py](python/server_utils.py) | RxInfer server | `create_client`, `ping_server`, `create_model` |

## Usage

### Julia
```julia
include("support/src/julia/NotebookConversion.jl")
using .NotebookConversion
```

### Python
```python
import sys
sys.path.insert(0, "support/src/python")
from gnn_utils import clone_repository
```
