# Support Source Modules

Reusable modules for RxInferExamples.jl support utilities.

Organized into language-specific subfolders with categorical submodules.

## Directory Structure

```
src/
├── julia/                # Julia modules
│   ├── Support.jl        # Umbrella module
│   ├── utils/            # General utilities
│   ├── statistics/       # Statistical analysis
│   ├── visualization/    # Plotting and animation
│   ├── logging/          # Logging infrastructure
│   └── environment/      # Environment management
├── python/               # Python modules
│   ├── utils/            # File and config utilities
│   ├── statistics/       # Statistical analysis
│   ├── visualization/    # Matplotlib helpers
│   ├── logging/          # Logging infrastructure
│   └── integration/      # External integrations
├── AGENTS.md
└── README.md
```

## Quick Start

### Julia

```julia
include("support/src/julia/Support.jl")
using .Support
```

### Python

```python
from support.src.python import utils, statistics, visualization
```

## Documentation

- [Julia Modules](julia/README.md)
- [Python Modules](python/README.md)
