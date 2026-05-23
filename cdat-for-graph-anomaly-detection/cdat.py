import math
from collections import deque

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import sparse as sp
from scipy.sparse.csgraph import shortest_path

from graph_transformer_layer import GraphTransformerLayer


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

    return g





class GraphTransformerNet(nn.Module):
    

    def __init__(self, in_feats, h_feats, num_classes, graph, num_heads, n_layers, layer_norm, batch_norm, residual, dropout, ffn, args, subsample_flag):
        super().__init__()

        in_dim_node = in_feats
        hidden_dim = h_feats
        out_dim = h_feats
        n_classes = num_classes
        #num_heads = 2
        #in_feat_dropout = net_params['in_feat_dropout']
        dropout = dropout
        #n_layers = 2

        #self.readout = net_params['readout']
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        #self.dropout = dropout
        self.n_classes = n_classes
        self.alpha=0.1
        self.r: int = 4
        self.t: int = 4
 



        self.embedding_h = nn.Linear(in_dim_node, hidden_dim) # node feat is an integer
        self.pos_embedding = nn.Embedding(graph.number_of_nodes(), hidden_dim)
        self.c = nn.Parameter(torch.rand(1))


        self.layers = GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout, ffn, self.layer_norm, self.batch_norm, self.residual)
        self.n_layers = n_layers - 1 
        self.layers_out = GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, ffn, self.layer_norm, self.batch_norm, self.residual)
        
        # Restore low-rank parameter form to avoid NxN explosion
        self.P = nn.Parameter(torch.rand(self.r, graph.number_of_nodes()))
        self.Q = nn.Parameter(torch.randn(self.r, self.r) * (1.0/self.r)**0.5)
        # Cache
        self._gaussian_cache = None
        self._cached_graph_hash = None
        self._norm_adj_cache = None
        self._cached_graph_sig = None

        
        #self.MLP_layer = nn.Linear(2 *out_dim, n_classes)
        if args.homo:
            if subsample_flag:
                self.MLP_layer = nn.Linear(out_dim, n_classes)
            else:
                self.MLP_layer = nn.Linear(2* out_dim, n_classes)
        else:
            self.MLP_layer = nn.Linear(out_dim, n_classes)
            #self.MLP_layer = nn.Linear(out_dim *len(graph.canonical_etypes), n_classes)
        
        self.args = args

        # Sampling default configuration
        # use_neighbor_sampling: whether to use subgraph batching; batch_size: number of seeds per batch; num_neighbors: number of neighbors per hop (string "1024,1024")
        self.use_neighbor_sampling = getattr(self.args, 'use_neighbor_sampling', True)
        self.batch_size = getattr(self.args, 'batch_size', 1024)
        num_neighbors_str = getattr(self.args, 'num_neighbors', "1024,1024")
        self.fanouts = [int(x) if int(x) > 0 else -1 for x in num_neighbors_str.split(',')]

    def suppression(self, x):
        # c decreases with time step t, exponential decay
        c = self.c 
        return c * x

    def gen_noise(self, key, t: int, std: float = 0.02, gamma: float = 0.55, shape=[1]):
        """
        Generate noise method, converted from JAX version to PyTorch version
        """
        if key is None:
            return key, 0
        
        if getattr(self, 'vary_noise', False):
            std = gamma / pow(1 + t, gamma)
        
        # Generate normal distribution noise using PyTorch
        noise = torch.normal(mean=0.0, std=std, size=shape)
        if hasattr(self, 'device'):
            noise = noise.to(self.device)
        elif next(self.parameters()).is_cuda:
            noise = noise.cuda()
            
        return key, noise
    
       
    
    
    def compute_gaussian_distance_matrix(self, g, J_0=1.0, a=0.5, threshold=0.01, max_hops=4):
        """
        Compute Gaussian distribution matrix based on shortest path distances in graph structure (limited to max_hops range)
        
        Args:
            g: DGL graph object
            J_0: Amplitude parameter of Gaussian distribution
            a: Standard deviation parameter of Gaussian distribution
            threshold: Truncation threshold, elements smaller than this will be set to 0
            max_hops: Maximum hop count, node pairs beyond this distance will not compute Gaussian values
            
        Returns:
            torch.Tensor: Symmetric matrix where element (i,j) is the Gaussian distance from node i to node j
        """
        
        # Get adjacency matrix
        adj_matrix = g.adjacency_matrix(scipy_fmt="csr")
        
        # Compute shortest path distances between all node pairs
        # Using scipy's shortest_path function
        distance_matrix = shortest_path(adj_matrix, directed=False, unweighted=True)
        
        # Set infinite distances (disconnected nodes) and distances beyond max_hops to infinity
        distance_matrix[np.isinf(distance_matrix)] = float('inf')
        distance_matrix[distance_matrix > max_hops] = float('inf')
        
        # Convert to torch tensor
        distance_matrix = torch.from_numpy(distance_matrix).float()
        
        # Move matrix to GPU if model is on GPU
        if next(self.parameters()).is_cuda:
            distance_matrix = distance_matrix.cuda()
        
        # Initialize Gaussian matrix as all zeros
        gaussian_matrix = torch.zeros_like(distance_matrix)
        
        # Only compute Gaussian values for node pairs within max_hops range
        valid_mask = ~torch.isinf(distance_matrix)
        
        # Apply Gaussian kernel function: J(x, x') = (J_0 / sqrt(2πa)) * exp[-(d)^2 / (2*a^2)]
        sqrt_term = torch.sqrt(torch.tensor(2 * math.pi * a**2, device=distance_matrix.device))
        gaussian_matrix[valid_mask] = (J_0 / sqrt_term) * torch.exp(-distance_matrix[valid_mask]**2 / (2 * a**2))
        
        # Handle diagonal elements (distance from self to self is 0)
        diagonal_mask = torch.eye(gaussian_matrix.shape[0], device=gaussian_matrix.device, dtype=torch.bool)
        gaussian_matrix[diagonal_mask] = J_0 / sqrt_term   # Value when d=0
        
        
        return gaussian_matrix

    # Low-rank multiplication to avoid explicit construction of P_Q_P (N×N)
    def _low_rank_mul(self, P: torch.Tensor, Q: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # P: (r, N), Q: (r, r), X: (N, d) -> returns (N, d)
        P = P.to(X.device, dtype=X.dtype)
        Q = Q.to(X.device, dtype=X.dtype)
        tmp = P @ X            # (r, d)
        tmp = Q @ tmp          # (r, d)
        out = P.t() @ tmp      # (N, d)
        return out

    # Get normalized adjacency sparse tensor (COO), cached for reuse
    def _get_normalized_adj(self, g, device):
        sig = (g.number_of_nodes(), g.number_of_edges())
        if self._norm_adj_cache is not None and self._cached_graph_sig == sig and self._norm_adj_cache.device == device:
            return self._norm_adj_cache

        # Build using SciPy COO, then convert to torch.sparse_coo_tensor
        adj_coo = g.adjacency_matrix(scipy_fmt="coo")
        row = torch.from_numpy(adj_coo.row.astype(np.int64))
        col = torch.from_numpy(adj_coo.col.astype(np.int64))
        n = g.number_of_nodes()

        deg = g.in_degrees().float().clamp(min=1.0)
        # Normalize weights 1/sqrt(d_u d_v)
        w = (deg[row] * deg[col]).sqrt().reciprocal()

        indices = torch.stack([row, col], dim=0)
        A_norm = torch.sparse_coo_tensor(indices, w, (n, n), dtype=torch.float32, device=device).coalesce()

        self._norm_adj_cache = A_norm
        self._cached_graph_sig = sig
        return A_norm

    # Approximate Gaussian kernel effect on graph using multi-hop sparse multiplication (without constructing NxN)
    def _apply_gaussian_filter(self, g, X: torch.Tensor, J_0=1.0, a=1.0, max_hops=4) -> torch.Tensor:
        # X: (N, d)
        A = self._get_normalized_adj(g, X.device)
        sqrt_term = math.sqrt(2.0 * math.pi * (a ** 2))

        out = (J_0 / sqrt_term) * X  # d=0 term
        Z = X
        for d in range(1, max_hops + 1):
            Z = torch.sparse.mm(A, Z)  # A_norm @ Z
            coeff = (J_0 / sqrt_term) * math.exp(-(d ** 2) / (2.0 * (a ** 2)))
            out = out + coeff * Z
        return out
    
    def forward(self, g, h, subsample_flag):
        """
        If use_neighbor_sampling is enabled, batch construct subgraphs for all nodes and forward,
        returns full graph logits (but internally concatenates subgraph batch results).
        """
        # Embed initial features
        H = self.embedding_h(h)

        if not self.use_neighbor_sampling:
            # Keep original full graph forward (for small graphs or debugging) - note we have restored low-rank terms
            if self.args.homo:
                h_cur = H
                h_orig = H
                for i in range(self.n_layers):
                    grad = self.layers(g, h_cur)
                    _, noise = self.gen_noise(None, i+1)
                    # Low-rank term: on full graph
                    lowrank_h = self._low_rank_mul(self.P, self.Q, h_cur)
                    h_cur = h_cur - self.alpha * grad \
                            - self.alpha * (h_cur + self.suppression(h_cur)) \
                            + self.alpha * lowrank_h \
                            + (self.alpha ** 0.5) * noise
                h_cur = self.layers_out(g, h_cur)
                if not subsample_flag:
                    h_cur = torch.cat((h_cur, h_orig), dim=-1)
                logits = self.MLP_layer(h_cur)
                return logits
            else:
                # Heterogeneous graph branch, keep as is
                h_all = []
                h1 = H
                for relation in g.canonical_etypes:
                    h_0 = H
                    for i in range(self.n_layers):
                        h_0 = self.layers(g[relation], h_0)
                    h_0 = self.layers_out(g[relation], h_0)
                    h_all.append(torch.nan_to_num(h_0))
                h_all.append(h1)
                h_rst = torch.max(torch.stack(h_all), dim=0)[0]
                logits = self.MLP_layer(h_rst)
                return logits

        # Subgraph batching: batch sample neighbor subgraphs for all nodes, forward batch by batch
        device = H.device
        N = g.number_of_nodes()
        all_nodes = torch.arange(N, device=device, dtype=torch.long)
        batch_size = max(1, int(self.batch_size))
        fanouts = self.fanouts if len(self.fanouts) > 0 else [1024, 1024]

        logits_full = torch.zeros(N, self.n_classes, device=device, dtype=torch.float32)

        # Iterate over all seed batches
        for i in range(0, N, batch_size):
            seeds = all_nodes[i: i + batch_size]
            # Construct k-hop subgraph
            subg, seed_pos = self._build_khop_subgraph(g, seeds, fanouts)
            # Align features with subgraph node order
            global_ids = subg.ndata[dgl.NID]
            H_sub = H[global_ids]

            # Run structure layers on subgraph (same logic as original, but low-rank term uses P sliced to subgraph)
            if self.args.homo:
                h_cur = H_sub
                h_orig = H_sub
                # Slice low-rank P to subgraph nodes
                P_slice = self.P[:, global_ids]
                for t in range(self.n_layers):
                    grad = self.layers(subg, h_cur)
                    _, noise = self.gen_noise(None, t+1)
                    lowrank_h = self._low_rank_mul(P_slice, self.Q, h_cur)
                    h_cur = h_cur - self.alpha * grad \
                            - self.alpha * (h_cur + self.suppression(h_cur)) \
                            + self.alpha * lowrank_h \
                            + (self.alpha ** 0.5) * noise
                h_cur = self.layers_out(subg, h_cur)
                if not subsample_flag:
                    h_cur = torch.cat((h_cur, h_orig), dim=-1)
                logits_sub = self.MLP_layer(h_cur)
                # Only write seed node outputs back to full graph
                logits_full[seeds] = logits_sub[seed_pos]
            else:
                # Heterogeneous graph: if subgraph batching is needed, sample subgraphs for each relation separately, keep full graph approach here or extend later
                h_all = []
                h1 = H_sub
                for relation in g.canonical_etypes:
                    h_0 = H_sub
                    for t in range(self.n_layers):
                        h_0 = self.layers(subg[relation], h_0)
                    h_0 = self.layers_out(subg[relation], h_0)
                    h_all.append(torch.nan_to_num(h_0))
                h_all.append(h1)
                h_rst = torch.max(torch.stack(h_all), dim=0)[0]
                logits_sub = self.MLP_layer(h_rst)
                logits_full[seeds] = logits_sub[seed_pos]

        return logits_full

    # Construct k-hop neighbor subgraph, return subgraph and seed positions in subgraph
    def _build_khop_subgraph(self, g: dgl.DGLGraph, seeds: torch.Tensor, fanouts: list):
        import dgl
        from dgl import sampling
        device = seeds.device
        frontier = seeds
        nodes_set = set(seeds.detach().cpu().tolist())
        for fanout in fanouts:
            sg = sampling.sample_neighbors(g, frontier, fanout=fanout, replace=False)
            # sg is the sampled subgraph, extract global node IDs
            if dgl.NID in sg.ndata:
                nids = sg.ndata[dgl.NID].detach().cpu().tolist()
                for nid in nids:
                    nodes_set.add(int(nid))
            else:
                # Fallback: if version doesn't have NID, fall back to merging src/dst indices (usually won't reach here)
                src, dst = sg.edges()
                for v in torch.cat([src, dst]).detach().cpu().tolist():
                    nodes_set.add(int(v))
            frontier = torch.tensor(list(nodes_set), device=device, dtype=torch.long)
        # Construct final regular subgraph (maintain NID mapping)
        all_nodes = torch.tensor(sorted(list(nodes_set)), device=device, dtype=torch.long)
        sub = dgl.node_subgraph(g, all_nodes)
        # Compute seed positions in subgraph node order
        global_ids = sub.ndata[dgl.NID]  # (M,)
        pos_map = {int(nid.item()): i for i, nid in enumerate(global_ids)}
        seed_pos = torch.tensor([pos_map[int(s.item())] for s in seeds], device=device, dtype=torch.long)
        return sub, seed_pos
    




        