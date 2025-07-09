#!/usr/bin/env bash
set -e

echo "==> (1/3) Preparing PubMed dataset if missing ..."
python -m data.make_pubmed --root ./data

echo "==> (2/3) Starting training (K=4) ..."
python train.py \
    --root ./data/pubmed \
    --epochs 150 \
    --batch-size 2048 \
    --k 4 \
    --eps 0.05

echo "==> (3/3) Done!  Results and figures are in ./results"
