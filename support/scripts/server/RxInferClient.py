#!/usr/bin/env python3
"""
RxInfer Server Client Example

This script demonstrates using the server_utils module from support/src/
to interact with RxInfer servers.

Prerequisites:
    pip install rxinferclient

Usage:
    python support/scripts/server/RxInferClient.py
"""

import sys
from pathlib import Path

# Setup path for importing from src/
script_dir = Path(__file__).parent
support_dir = script_dir.parent.parent
src_dir = support_dir / "src"
sys.path.insert(0, str(src_dir))

from server_utils import create_client, ping_server, create_model, delete_model


def main():
    """Demonstrate RxInfer server client usage."""
    print("RxInfer Server Client Demo")
    print("=" * 40)
    
    try:
        # Create client
        client = create_client()
        print("✅ Client created")
        
        # Ping server
        if ping_server(client):
            print("✅ Server is responding")
        else:
            print("❌ Server not responding")
            return 1
        
        # Create a model instance
        instance_id = create_model(client, "BetaBernoulli-v1")
        if instance_id:
            print(f"✅ Created model instance: {instance_id}")
            
            # Clean up
            if delete_model(client, instance_id):
                print("✅ Deleted model instance")
        
    except ImportError as e:
        print(f"❌ {e}")
        print("Install with: pip install rxinferclient")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("=" * 40)
    print("Demo complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())