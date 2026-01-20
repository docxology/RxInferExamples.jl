# Server Integration

Python client utilities for RxInfer server integration.

## Prerequisites

```bash
pip install rxinferclient
```

## Usage

```python
from rxinferclient import RxInferClient

# Initialize with default settings (auto-generates API key)
client = RxInferClient()

# Check server status
response = client.server.ping_server()
print(response.status)  # 'ok'

# Create a model instance
response = client.models.create_model_instance({
    "model_name": "BetaBernoulli-v1"
})
instance_id = response.instance_id

# Delete when done
client.models.delete_model_instance(instance_id=instance_id)
```

## Custom Configuration

```python
# Custom server URL
client = RxInferClient(server_url="http://localhost:8000/v1")

# Custom API key
client = RxInferClient(api_key="your-api-key")
```

## Resources

- [RxInferServer.jl](https://github.com/lazydynamics/RxInferServer) - Server implementation
- [RxInferClient.py](https://github.com/lazydynamics/RxInferClient.py) - Python SDK
- [Server Documentation](https://server.rxinfer.com)
