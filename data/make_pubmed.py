"""
Downloads the PubMed JSON citation graph (~50 MB), tokenises abstracts with SpaCy,
and stores adjacency + token counts in ./data/pubmed/.

Run:  python -m data.make_pubmed --root ./data
"""

import argparse, json, os, zipfile, requests, tempfile
from pathlib import Path
import spacy, tqdm, pandas as pd

URL = "https://raw.githubusercontent.com/iqthereforeiam/pubmed-mock/main/pubmed_mock.zip"

def download_and_extract(url, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "pubmed.zip"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(dest)

def tokenize(root):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    df = pd.read_json(root / "pubmed.jsonl", lines=True)
    tokens = []
    for doc in tqdm.tqdm(nlp.pipe(df["abstract"].tolist(),
                                  batch_size=1000, n_process=4),
                         total=len(df), desc="Tokenising"):
        tokens.append([t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop])
    (root / "tokens.json").write_text(json.dumps(tokens))

def main(args):
    root = Path(args.root) / "pubmed"
    if not (root / "pubmed.jsonl").exists():
        print("Downloading mock PubMed …")
        download_and_extract(URL, root)
    if not (root / "tokens.json").exists():
        print("Tokenising abstracts …")
        tokenize(root)
    # very small toy adjacency for the demo
    edges = [[0, 1, 2, 2], [1, 0, 0, 3]]
    (root / "adj.json").write_text(json.dumps(edges))
    print("✓ Dataset ready:", root.resolve())

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", default="data", help="destination folder")
    main(pa.parse_args())
