import jax, flax, tqdm, torch, functools, numpy as np, jax.numpy as jnp

from typing import (
    Callable,
    List,
)

from torch_geometric.utils import unbatch, to_dense_adj

from .tools_utils import to_device_split, get_pos

from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold


def get_adjs(graph, use_edge_attr: bool = True):
    adj = edge_attr = None
    adj = to_dense_adj(graph.edge_index, graph.batch)

    # add CLS
    adj = torch.nn.functional.pad(adj, (1, 0, 1, 0), "constant", 1.0).numpy()

    if not (use_edge_attr):
        return adj, None

    try:
        edge_attr = graph.edge_attr
    except:
        edge_attr = None
    else:
        edge_attr = to_dense_adj(
            graph.edge_index, graph.batch, edge_attr=graph.edge_attr
        )

        arr = [0 for _ in range(len(edge_attr.shape) * 2)]

        if len(edge_attr.shape) == 3:
            arr[0] = 1
            arr[2] = 1
        else:
            arr[2] = 1
            arr[4] = 1

        edge_attr = torch.nn.functional.pad(edge_attr, arr, "constant", 1.0).numpy()
        edge_attr = edge_attr[..., None] if len(edge_attr.shape) == 3 else edge_attr

    return adj, edge_attr


def batchify(
    src: List[torch.Tensor],
    adjs: List[torch.Tensor],
    max_num_nodes: int,
    pad_value: float,
):
    """Find the maximnum number of nodes in the list then pad and concatenate
    to generate a tensor of shape B x N x d

    Args:
        src: a list of tensors
    """

    def pad_shape(maxl, minl, size):
        arr = [0 for _ in range(size * 2)]
        arr[-1] = maxl - minl
        return arr

    max_nodes = max_num_nodes

    pad_fn = lambda x, ub: torch.nn.functional.pad(
        x, pad_shape(ub, x.size(0), len(x.shape)), "constant", pad_value
    )

    if adjs is not None:
        adj, edge_attr = adjs
    else:
        adj = edge_attr = None

    batch = None
    for i, tsr in enumerate(src):
        if tsr.size(0) >= max_nodes:
            # no padding is required
            tsr = tsr[:max_nodes]

        else:
            # padding is required
            size = tsr.size(0)
            tsr = pad_fn(tsr, max_nodes)

            # provide padded entries as nodes connect to actual nodes of the graph
            if adjs is not None:
                adj[i, size + 1 :, : size + 1] = 1.0

                if edge_attr is not None:
                    edge_attr[i, size + 1 :, : size + 1] = 1.0

        if batch == None:
            batch = tsr[None]
        else:
            batch = torch.cat([batch, tsr[None]], 0)

    if adj is not None:
        adj = adj[:, : max_nodes + 1, : max_nodes + 1]

    if edge_attr is not None:
        edge_attr = edge_attr[:, : max_nodes + 1, : max_nodes + 1]

    return batch.numpy(), adj, edge_attr


def get_graph(
    graph,
    max_num_nodes: int = 500,
    k: int = 10,
    embed_type: str = "eigen",
    task_level: str = "graph",
    to_device: bool = False,
    flip_sign: bool = True,
    use_edge_attr: bool = True,
    include_key: bool = True,
):
    nodes = unbatch(graph.x, graph.batch)
    adj, edge_attr = get_adjs(graph, use_edge_attr)
    max_num_nodes = min(adj.shape[1] - 1, max_num_nodes)
    nodes, adj, edge_attr = batchify(nodes, [adj, edge_attr], max_num_nodes, 0)
    nodes = torch.from_numpy(nodes).float()

    try:
        pos_feats = unbatch(graph.pos, graph.batch)
    except:
        pos_feats = None
    else:
        pos_feats, _, _ = batchify(pos_feats, None, max_num_nodes, 0)
        pos_feats = torch.from_numpy(pos_feats).float().to(nodes.device)
        nodes = nodes.to(pos_feats.device)
        nodes = torch.concatenate([nodes, pos_feats], dim=-1)
    nodes = nodes.cpu().numpy()

    if task_level == "graph":
        """Graph Classification Task"""
        labels = graph.y.numpy()

    else:
        """Node Classification Task"""
        labels = unbatch(graph.y, graph.batch)
        labels, _, _ = batchify(labels, None, max_num_nodes, -1)

    batch_size = nodes.shape[0]
    pos_embeddings = get_pos(embed_type, k, flip_sign)(adj)

    if to_device:
        nodes, labels, adj, edge_attr, pos_embeddings = map(
            to_device_split, (nodes, labels, adj, edge_attr, pos_embeddings)
        )

        batch_size = nodes.shape[0] * nodes.shape[1]

    key = None
    if include_key:
        key = jax.random.split(
            jax.random.PRNGKey(np.random.randint(nodes.shape[1], nodes.shape[1] * 2))
        )[0]

    return {
        "X": nodes,
        "Y": labels,
        "A": adj[..., None] if not (use_edge_attr) else edge_attr,
        "P": pos_embeddings,
        "batch_size": batch_size,
        "key": flax.jax_utils.replicate(key) if to_device else key,
    }


def init_model(
    train_loader,
    key,
    model,
    k: int = 10,
    embed_type: str = "eigen",
    task_level: str = "graph",
):
    for graph in train_loader:
        data = get_graph(graph, 500, k, embed_type, task_level)

        params = model.init(
            key, data["X"], data["A"], data["P"], data["key"], False, False
        )["params"]

        return params


def train_step(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    A: jnp.ndarray,
    P: jnp.ndarray,
    key: jnp.ndarray,
    state,
    partial_loss_fn: functools.partial,
):
    grad_fn = jax.value_and_grad(partial_loss_fn, has_aux=True)

    (loss, accuracy), grads = grad_fn(
        state.params, {"X": X, "Y": Y, "A": A, "P": P, "key": key}
    )

    loss, accuracy, grads = map(
        lambda x: jax.lax.pmean(x, "batch"), (loss, accuracy, grads)
    )

    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss, "accuracy": accuracy}


def eval_loop(
    state,
    loader,
    get_data: Callable,
    eval_fn: Callable,
    return_metrics: bool = False,
):
    Loss = 0
    count = 0
    Accuracy = 0

    if not (return_metrics):
        loader = tqdm.tqdm(loader)

    for graph in loader:
        data = get_data(graph)
        m = data.pop("batch_size")
        loss, accuracy = eval_fn(state.params, data)

        Loss += loss.mean().item()
        Accuracy += accuracy.mean().item()
        count += 1

    if return_metrics:
        return Loss / count, Accuracy / count
    print("J:", Loss / count, "Accuracy:", Accuracy / count)


def train_loop(
    state,
    loaders: dict,
    get_data: dict,
    train_fn: Callable,
    eval_fn: Callable,
    epochs: int = 20,
    path: str = "model_stats.npy",
    key=None,
):
    stats = np.zeros([epochs, 4])
    trainer = tqdm.tqdm(range(1, epochs + 1))
    train_loader, valid_loader = loaders["train"], loaders["valid"]
    get_train_data, get_valid_data = get_data["train"], get_data["valid"]

    update_step = jax.pmap(
        functools.partial(train_step, partial_loss_fn=train_fn),
        axis_name="batch",
    )

    if key is None:
        key = jax.random.PRNGKey(epochs + np.random.randint(0, 100000, size=()))
    for i in trainer:
        trainer.set_description()

        running_metrics = {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "valid_loss": 0.0,
            "valid_accuracy": 0.0,
        }

        count = 0
        for graph in train_loader:
            data = get_train_data(graph)

            state, metrics = update_step(
                data["X"], data["Y"], data["A"], data["P"], data["key"], state
            )

            running_metrics["train_loss"] += metrics["loss"].mean().item()
            running_metrics["train_accuracy"] += metrics["accuracy"].mean().item()
            count += 1

        running_metrics["train_loss"] /= count
        running_metrics["train_accuracy"] /= count

        stats[i - 1, 0] = running_metrics["train_loss"]
        stats[i - 1, 1] = running_metrics["train_accuracy"]

        valid_loss = valid_accuracy = 0.0
        if valid_loader is not None:
            valid_loss, valid_accuracy = eval_loop(
                state, valid_loader, get_valid_data, eval_fn, return_metrics=True
            )
            stats[i - 1, 2] = valid_loss
            stats[i - 1, 3] = valid_accuracy

        trainer.set_postfix(
            train_loss=stats[i - 1, 0],
            train_accuracy=stats[i - 1, 1],
            val_loss=stats[i - 1, 2],
            val_accuracy=stats[i - 1, 3],
        )

    _file = open(path, "ab")
    np.savetxt(_file, stats)
    _file.close()
    print("Saved statistics to {0}".format(path))

    return key, state


def eval_nfold(
    dataset,
    model,
    create_state_fn: Callable,
    get_data: Callable,
    train_fn: Callable,
    eval_fn: Callable,
    nfolds: int = 10,
    epochs: int = 20,
    batch_size: int = 64,
    key=None,
    k: int = 10,
    embed_type: str = "eigen",
    task_level: str = "graph",
    random_state: int = 42,
    stats_path: str = "nfold_stats.npy",
):
    if key is None:
        key = jax.random.PRNGKey(random_state)

    kfold = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    fold_losses, fold_accuracies = [], []

    for fold_idx, (train_index, test_index) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold_idx + 1}/{nfolds}")

        train_data = dataset[train_index.tolist()]
        test_data = dataset[test_index.tolist()]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        key, subkey = jax.random.split(key)
        init_loader = DataLoader(train_data, batch_size=1)
        params = init_model(init_loader, subkey, model, k, embed_type, task_level)

        state = create_state_fn(params)

        fold_stats_path = stats_path.replace(".npy", f"_fold{fold_idx}.npy")
        key, subkey = jax.random.split(key)
        key, state = train_loop(
            state,
            loaders={"train": train_loader, "valid": None},
            get_data={"train": get_data, "valid": get_data},
            train_fn=train_fn,
            eval_fn=eval_fn,
            epochs=epochs,
            path=fold_stats_path,
            key=subkey,
        )

        loss, accuracy = eval_loop(
            state, test_loader, get_data, eval_fn, return_metrics=True
        )
        fold_losses.append(loss)
        fold_accuracies.append(accuracy)
        print(f"  Fold {fold_idx + 1} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    mean_loss = np.mean(fold_losses)
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"N-Fold Result: Accuracy = {mean_accuracy:.4f} +/- {std_accuracy:.4f}")

    return mean_loss, mean_accuracy, std_accuracy
