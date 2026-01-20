# Python Modules

Reusable Python modules for RxInferExamples.jl support utilities.

## Modules

| Module | Purpose |
|--------|---------|
| `gnn_utils.py` | GNN repository cloning and integration |
| `server_utils.py` | RxInfer server client utilities |

## Usage

```python
import sys
sys.path.insert(0, "support/src/python")
from gnn_utils import clone_repository, verify_clone
from server_utils import create_client, ping_server
```
