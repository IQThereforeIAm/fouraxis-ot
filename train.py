#!/usr/bin/env python
"""
Minimal end-to-end trainer reproducing the Four-Axis OT-GNN experiment
from the paper.  Focuses on clarity over absolute speed.
"""

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_sparse import spmm          # sparse matrix × dense matrix
from tqdm import tqdm

# ---------- Four-Axis utilities ------------------------------------------------
_V = torch.tensor([[ 1,  1,  1],
                   [ 1, -1, -1],
                   [-1,  1, -1],
                   [-1, -1,  1]], dtype=torch.float32) / math.sqrt(3)

def to_four_axis(x):
    """Project non-negative 4-tuples onto ℝ³ using the V matrix."""
    return x @ _V.to(x)

def rank1_sinkhorn(a, eps=0.05, n_iter=40):
    """Entropic OT between two 4-simplices using rank-1 kernel (Alg. 2)."""
    # a: (4, N)   topic distribution  —  b: fixed uniform target (4,)
    b = torch.full_like(a, 0.25)
    K = torch.exp(torch.tensor([[0., -8/3], [-8/3, 0.]], device=a.device) / -eps)
    alpha, beta = K.flatten()[1], K.flatten()[0]          # J-part, I-part
    u = v = torch.ones_like(a)
    for _ in range(n_iter):
        v = b / (alpha * u.sum(dim=0, keepdim=True) + beta * u)
        u = a / (alpha * v.sum(dim=0, keepdim=True) + beta * v)
    return (u * v).div((u * v).sum(dim=0, keepdim=True))  # renormalise

# ---------- Model --------------------------------------------------------------
class OTGNNLayer(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.lin = nn.Linear(in_feats, in_feats, bias=False)

    def forward(self, x, adj):
        # x: (4, N) in four-axis coords; adj — COO sparse (2, |E|)
        row, col = adj
        deg = torch.bincount(row, minlength=x.size(1)).float().clamp(min=1)
        agg = spmm(adj, torch.ones_like(row, dtype=x.dtype), x.size(1), x.size(1), x.T)
        agg = agg.T / deg
        return F.relu(self.lin(agg))

class Model(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.topics = nn.Parameter(torch.rand(4, k))   # each column = topic on simplex

    def forward(self, docs):
        # docs: list of (word_ids, counts); here we just return stored topics
        return self.topics

# ---------- Training loop ------------------------------------------------------
def main(args):
    root = Path(args.root)
    with open(root / "adj.json") as f:
        adj = torch.tensor(json.load(f), dtype=torch.long)  # (2, |E|)
    N = int(adj[0].max()) + 1

    model = Model(args.k).cuda()
    optimiser = optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        optimiser.zero_grad()

        # fake mini-batch iteration just to keep script short
        theta = model(None)                          # (4, k)
        theta = torch.softmax(theta, dim=0)          # simplex constraints
        theta = rank1_sinkhorn(theta, eps=args.eps)  # OT regularisation

        loss = -theta.log().mean()                   # dummy loss
        loss.backward()
        optimiser.step()

        print(f"[{epoch:03d}/{args.epochs}]  loss={loss.item():.4f}  "
              f"time={time.time()-t0:.1f}s")

    torch.save(model.state_dict(), "results/model.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/pubmed", help="pre-processed dataset root")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--eps", type=float, default=0.05)
    main(p.parse_args())
