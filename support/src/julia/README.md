# Julia Modules

Reusable Julia modules for RxInferExamples.jl support utilities.

## Modules

| Module | Purpose |
|--------|---------|
| `CommandUtils.jl` | Shell command execution and argument parsing |
| `EnvironmentSetup.jl` | Environment configuration and package management |
| `NotebookConversion.jl` | Jupyter notebook to Julia script conversion |

## Usage

```julia
include("support/src/julia/NotebookConversion.jl")
using .NotebookConversion

# Convert a notebook
convert_notebook("example.ipynb", "example.jl")
```
