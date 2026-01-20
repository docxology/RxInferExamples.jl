# Python Support Modules

Reusable Python modules for RxInferExamples.jl support utilities.

## Structure

```
python/
├── __init__.py
├── utils/
│   ├── file_utils.py    # File system operations
│   └── config_utils.py  # Configuration handling
├── statistics/
│   ├── analysis_utils.py # Statistical metrics
│   └── validation_utils.py # Validation framework
├── visualization/
│   └── plotting_utils.py # Matplotlib helpers
├── experiment_logging/
│   └── logging_utils.py # Logging infrastructure
├── integration/
│   ├── gnn_utils.py     # GNN repository integration
│   └── server_utils.py  # RxInfer server client
└── README.md
```

## Usage

### Import All Submodules

```python
from support.src.python import utils, statistics, visualization, logging, integration
```

### Import Specific Functions

```python
from support.src.python.statistics import calculate_rmse, ValidationResult
from support.src.python.utils import load_toml, ensure_directory
```

## Module Categories

| Category | Modules | Purpose |
|----------|---------|---------|
| **utils** | file_utils, config_utils | File and config utilities |
| **statistics** | analysis_utils, validation_utils | Statistical analysis |
| **visualization** | plotting_utils | Matplotlib helpers |
| **experiment_logging** | logging_utils | Logging infrastructure |
| **integration** | gnn_utils, server_utils | External integrations |
