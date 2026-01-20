# Julia Support Modules

Reusable Julia modules for RxInferExamples.jl support utilities.

## Structure

```
julia/
├── Support.jl           # Umbrella module
├── utils/
│   ├── CommandUtils.jl  # Shell command execution
│   ├── FileUtils.jl     # File system operations
│   └── ConfigUtils.jl   # TOML configuration handling
├── statistics/
│   ├── AnalysisUtils.jl # Statistical metrics (RMSE, MAE, VAF)
│   └── ValidationUtils.jl # Validation framework
├── visualization/
│   ├── PlottingUtils.jl # Plotting helpers
│   └── AnimationUtils.jl # Animation creation
├── logging/
│   ├── LoggingUtils.jl  # Logging infrastructure
│   └── ReportingUtils.jl # Report generation
├── environment/
│   ├── EnvironmentSetup.jl # Environment configuration
│   └── NotebookConversion.jl # Notebook to script conversion
└── README.md
```

## Usage

### Load All Modules

```julia
include("support/src/julia/Support.jl")
using .Support
```

### Load Individual Submodules

```julia
include("support/src/julia/utils/FileUtils.jl")
using .FileUtils
```

## Module Categories

| Category | Modules | Purpose |
|----------|---------|---------|
| **utils** | CommandUtils, FileUtils, ConfigUtils | General utilities |
| **statistics** | AnalysisUtils, ValidationUtils | Statistical analysis |
| **visualization** | PlottingUtils, AnimationUtils | Plotting and animation |
| **logging** | LoggingUtils, ReportingUtils | Logging infrastructure |
| **environment** | EnvironmentSetup, NotebookConversion | Environment management |
