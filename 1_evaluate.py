#!/usr/bin/env python3
# evaluate.py  (count class dist + accuracy)

import json
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------ CONFIG ------------
ARTIFACTS_DIR = "artifacts_custom"
SPLITS        = ["train", "validation", "test"]
BATCH_SIZE    = 32
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------

def has_tokenizer_files(dir_path: Path) -> bool:
    needed = {"tokenizer.json", "tokenizer_config.json"}
    return needed.issubset({p.name for p in dir_path.glob("*")})

def load_tokenizer(model_dir: Path, tok_dir: Path):
    if has_tokenizer_files(tok_dir):
        return AutoTokenizer.from_pretrained(tok_dir)
    if has_tokenizer_files(model_dir):
        return AutoTokenizer.from_pretrained(model_dir)
    return AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else model_dir)

def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch]) for k in keys if k != "label"}
    if "label" in keys:
        out["labels"] = torch.tensor([b["label"] for b in batch])
    return out

def accuracy(model, ds):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(DEVICE)
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            preds  = model(**batch).logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

def label_dist(ds):
    if "label" not in ds.features:
        return None
    cnt = Counter(int(x) for x in ds["label"])
    # จัดรูปแบบ "class:value" เรียงตาม key
    return ", ".join(f"{k}:{cnt[k]}" for k in sorted(cnt.keys()))

def main():
    art = Path(ARTIFACTS_DIR)
    meta = json.loads((art / "meta.json").read_text())

    model_dir = Path(meta["model_dir"])
    tok_dir   = Path(meta["tokenizer_dir"])
    data_dir  = Path(meta["dataset_dir"])

    tokenizer = load_tokenizer(model_dir, tok_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)

    ds_all = load_from_disk(str(data_dir))

    for sp in SPLITS:
        if sp not in ds_all:
            print(f"{sp}: missing split")
            continue

        ds = ds_all[sp]
        dist = label_dist(ds)
        if dist is not None:
            print(f"{sp} label dist -> {dist}")

        if "label" in ds.features:
            acc = accuracy(model, ds)
            print(f"Accuracy ({sp}): {acc:.4f}")
        else:
            print(f"{sp}: no label, skip accuracy.")

if __name__ == "__main__":
    main()
