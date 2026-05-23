"""
Simple example demonstrating how to use CDAT (Controlled Dynamics Attractor Transformer)
for graph classification tasks.
"""

import os
import sys
import jax
import jax.numpy as jnp
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import GraphCDAT, init_model, get_graph, get_nparams


def main():
    # Configuration
    data_name = "MUTAG"  # Small dataset for quick testing
    k = 15  # Positional embedding dimension
    embed_type = "eigen"  # Positional embedding type: "eigen" or "svd"
    task_level = "graph"  # Task level: "graph" or "node"
    max_num_nodes = 500
    
    # Model configuration
    embed_dim = 128
    out_dim = 2  # Number of classes for MUTAG dataset
    nheads = 4
    alpha = 0.1
    depth = 2
    block = 2
    head_dim = 64
    multiplier = 4.0
    
    print(f"Using devices: {jax.local_devices()}")
    
    # Load dataset (will auto-download if not present)
    print(f"Loading {data_name} dataset...")
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_data = TUDataset(root=data_root, name=data_name, use_node_attr=True)
    print(f"Dataset loaded: {len(train_data)} graphs")
    
    # Initialize model
    print("Initializing CDAT model...")
    model = GraphCDAT(
        embed_dim=embed_dim,
        out_dim=out_dim,
        nheads=nheads,
        alpha=alpha,
        depth=depth,
        block=block,
        head_dim=head_dim,
        multiplier=multiplier,
        num_tokens=500,
        dtype=jnp.float32,
        kernel_size=[3, 3],
        kernel_dilation=[1, 1],
        compute_corr=True,
        vary_noise=False,
        chn_atype="relu",
    )
    
    # Initialize model parameters
    key = jax.random.PRNGKey(42)
    init_loader = DataLoader(train_data, batch_size=1)
    params = init_model(init_loader, key, model, k, embed_type, task_level)
    print(f"Model parameters: {get_nparams(params):,}")
    
    # Example: Forward pass on a single batch
    print("\nRunning forward pass on a sample batch...")
    sample_loader = DataLoader(train_data, batch_size=2)
    sample_batch = next(iter(sample_loader))
    
    # Prepare graph data
    from functools import partial
    get_data_fn = partial(
        get_graph,
        max_num_nodes=max_num_nodes,
        k=k,
        embed_type=embed_type,
        task_level=task_level,
        to_device=False,
    )
    
    X, A, P, targets = get_data_fn(sample_batch)
    
    # Forward pass
    key, subkey = jax.random.split(key)
    output = model.apply(
        {'params': params},
        X, A, P,
        key=subkey,
        return_stats=False,
        training=False,
    )
    
    print(f"Input shape: {X.shape}")
    print(f"Output CLS token shape: {output['CLS'].shape}")
    print(f"Output node features shape: {output['X'].shape}")
    print(f"Output adjacency shape: {output['A'].shape}")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
