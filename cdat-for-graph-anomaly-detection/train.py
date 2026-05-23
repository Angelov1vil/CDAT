import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score

from dataset import Dataset
from cdat import GraphTransformerNet


# ------------------ Threshold Search Function ------------------ #
def get_best_f1(labels, probs):
    best_f1, best_thre = 0.0, 0.0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average="macro")
        if mf1 > best_f1:
            best_f1, best_thre = mf1, thres
    return best_f1, best_thre


# ------------------ Subgraph Sampling Training ------------------ #
def train(g, args, device, model, subsample_flag):
    sample_ratio = 0.05  # Sampling ratio per iteration

    if args.dataset == "tsocial":
        sample_ratio = 0.001

    # Original graph data
    features_all = g.ndata["feature"]
    labels_all = g.ndata["label"]

    # Data splitting
    all_idx = list(range(len(labels_all)))
    if args.dataset == "amazon":
        all_idx = list(range(3305, len(labels_all)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(
        all_idx,
        labels_all[all_idx],
        stratify=labels_all[all_idx],
        train_size=args.train_ratio,
        random_state=2,
        shuffle=True,
    )

    idx_valid, idx_test, y_valid, y_test = train_test_split(
        idx_rest, y_rest, stratify=y_rest, test_size=0.67, random_state=2, shuffle=True
    )

    train_mask_all = torch.zeros(len(labels_all), dtype=torch.bool)
    val_mask_all = torch.zeros_like(train_mask_all)
    test_mask_all = torch.zeros_like(train_mask_all)

    train_mask_all[idx_train] = True
    val_mask_all[idx_valid] = True
    test_mask_all[idx_test] = True

    print(
        f"train/dev/test samples: {train_mask_all.sum().item()}, {val_mask_all.sum().item()}, {test_mask_all.sum().item()}"
    )

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pos_weight = (1 - labels_all[train_mask_all]).sum().item() / labels_all[
        train_mask_all
    ].sum().item()
    print(f"cross entropy weight: {pos_weight:.2f}")

    best_f1 = 0.0
    final_trec = final_tpre = final_tmf1 = final_tauc = 0.0
    tic = time.time()

    # ============================== Training Loop ============================== #
    for epoch in range(args.epoch):
        # -------- Sample Subgraph -------- #
        idx_train_sample = np.random.choice(
            idx_train, int(sample_ratio * len(idx_train)), replace=False
        )
        idx_valid_sample = np.random.choice(
            idx_valid, int(sample_ratio * len(idx_valid)), replace=False
        )
        idx_test_sample = np.random.choice(
            idx_test, int(sample_ratio * len(idx_test)), replace=False
        )

        sub_nodes = (
            list(idx_train_sample) + list(idx_valid_sample) + list(idx_test_sample)
        )
        sg = g.subgraph(sub_nodes)  # Subgraph
        sg = dgl.add_self_loop(sg)  # Add self-loops
        features = sg.ndata["feature"].to(device)
        labels = sg.ndata["label"].to(device)

        # Subgraph masks corresponding to new node order
        train_sample_size = len(idx_train_sample)
        valid_sample_size = len(idx_valid_sample)
        test_sample_size = len(idx_test_sample)

        train_mask = torch.zeros(
            train_sample_size + valid_sample_size + test_sample_size,
            dtype=torch.bool,
            device=device,
        )
        val_mask = torch.zeros_like(train_mask)
        test_mask = torch.zeros_like(train_mask)

        train_mask[:train_sample_size] = True
        val_mask[train_sample_size : train_sample_size + valid_sample_size] = True
        test_mask[-test_sample_size:] = True

        # -------- Forward + Backward -------- #
        model.train()
        logits = model(sg.to(device), features, subsample_flag)
        loss = F.cross_entropy(
            logits[train_mask],
            labels[train_mask],
            weight=torch.tensor([1.0, pos_weight], device=device),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------- Validation + Testing -------- #
        model.eval()
        with torch.no_grad():
            probs = logits.softmax(dim=1)

        f1_val, threshold = get_best_f1(
            labels[val_mask].cpu().numpy(), probs[val_mask].cpu().numpy()
        )

        preds_np = (probs[:, 1] > threshold).int().cpu().numpy()
        labels_np = labels.cpu().numpy()
        trec = recall_score(
            labels_np[test_mask.cpu().numpy()], preds_np[test_mask.cpu().numpy()]
        )
        tpre = precision_score(
            labels_np[test_mask.cpu().numpy()], preds_np[test_mask.cpu().numpy()]
        )
        tmf1 = f1_score(
            labels_np[test_mask.cpu().numpy()],
            preds_np[test_mask.cpu().numpy()],
            average="macro",
        )
        tauc = roc_auc_score(
            labels_np[test_mask.cpu().numpy()],
            probs[:, 1].cpu().numpy()[test_mask.cpu().numpy()],
        )

        if f1_val > best_f1:
            best_f1 = f1_val
            final_trec, final_tpre, final_tmf1, final_tauc = trec, tpre, tmf1, tauc

        print(
            f"Epoch {epoch:03d} | loss {loss.item():.4f} | val MF1 {f1_val:.4f} (best {best_f1:.4f})"
        )

        del logits, probs
        torch.cuda.empty_cache()

    # ============================== Training Complete ============================== #
    print(f"time cost: {time.time() - tic:.1f}s")
    print(
        f"Test: REC {final_trec * 100:.2f} PRE {final_tpre * 100:.2f} MF1 {final_tmf1 * 100:.2f} AUC {final_tauc * 100:.2f}"
    )
    return final_tmf1, final_tauc


# ------------------ Main Entry Point ------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Controlled Dynamics Attractor Transformer (CDAT) for Graph Anomaly Detection"
    )
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--homo", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--layer_norm", type=bool, default=False)
    parser.add_argument("--batch_norm", type=bool, default=False)
    parser.add_argument("--residual", type=bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ffn", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    
    # Controlled dynamics iteration core parameters (externally controllable)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Dynamics iteration step size (default: 0.1)")
    parser.add_argument("--suppression_coef", type=float, default=1.0,
                        help="Coefficient for suppression term (default: 1.0)")
    parser.add_argument("--noise_std", type=float, default=0.02,
                        help="Base noise standard deviation, set to 0 to disable noise (default: 0.02)")
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    graph = Dataset(args.dataset, args.homo).graph
    in_feats = graph.ndata["feature"].shape[1]
    num_class = 2
    subsample_flag = 1  # Subgraph sampling mode

    mf1s, aucs = [], []
    for r in range(args.run):
        model = GraphTransformerNet(
            in_feats,
            args.hid_dim,
            num_class,
            graph,
            args.num_heads,
            args.n_layers,
            args.layer_norm,
            args.batch_norm,
            args.residual,
            args.dropout,
            args.ffn,
            args,
            subsample_flag,
        ).to(device)

        mf1, auc = train(graph, args, device, model, subsample_flag)
        mf1s.append(mf1)
        aucs.append(auc)

    mf1s = np.array(mf1s)
    aucs = np.array(aucs)
    print(
        f"MF1-mean: {100 * mf1s.mean():.2f}  MF1-std: {100 * mf1s.std():.2f} "
        + f"AUC-mean: {100 * aucs.mean():.2f}  AUC-std: {100 * aucs.std():.2f}"
    )


if __name__ == "__main__":
    main()
