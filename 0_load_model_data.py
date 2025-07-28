#!/usr/bin/env python3
# load_model_data.py  (no argparse, fixed concatenate_datasets)

from pathlib import Path
import json
from collections import Counter

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================== CONFIG ==================
MODEL_NAME      = "facebook/opt-125m"
NUM_LABELS      = 2
OUT_DIR         = "artifacts_custom"

TOTAL_SAMPLES   = 30000              # total examples (from train+validation with labels)
SPLITS          = [0.96, 0.02, 0.02]  # train / val / test (sum = 1.0)
MAX_LEN         = 128
SEED            = 42
# ============================================

def check_splits(splits):
    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"SPLITS must sum to 1.0, got {splits}")

def print_shapes(encoded_ds):
    print("\nDataset shapes:")
    for split in encoded_ds.keys():
        n = len(encoded_ds[split])
        seq_len = len(encoded_ds[split][0]["input_ids"])
        labels = [int(l) for l in encoded_ds[split]["label"]]
        dist = Counter(labels)
        print(f"  {split:<10}: samples = {n}, seq_len = {seq_len}")
        print(f"              label dist = {dict(dist)}")
    print()

def main():
    check_splits(SPLITS)

    out_dir   = Path(OUT_DIR)
    model_dir = out_dir / "model"
    tok_dir   = out_dir / "tokenizer"
    data_dir  = out_dir / "dataset_sst2"
    meta_file = out_dir / "meta.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    print("Loading SST-2 dataset (labeled only: train+validation)...")
    raw_ds = load_dataset("glue", "sst2")
    # use helper instead of .concatenate()
    labeled = concatenate_datasets([raw_ds["train"], raw_ds["validation"]])

    if TOTAL_SAMPLES > len(labeled):
        raise ValueError(f"TOTAL_SAMPLES={TOTAL_SAMPLES} > available labeled={len(labeled)}")

    labeled = labeled.shuffle(seed=SEED).select(range(TOTAL_SAMPLES))

    r_train, r_val, r_test = SPLITS
    n_train = int(TOTAL_SAMPLES * r_train)
    n_val   = int(TOTAL_SAMPLES * r_val)
    n_test  = TOTAL_SAMPLES - n_train - n_val

    train_ds = labeled.select(range(0, n_train))
    val_ds   = labeled.select(range(n_train, n_train + n_val))
    test_ds  = labeled.select(range(n_train + n_val, TOTAL_SAMPLES))

    ds = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    def preprocess(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    print("Tokenizing...")
    # remove_columns needs existing names
    remove_cols = [c for c in ["sentence", "idx"] if c in ds["train"].column_names]
    encoded_ds = ds.map(preprocess, batched=True, remove_columns=remove_cols)
    encoded_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print_shapes(encoded_ds)

    print("Saving to disk...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tok_dir)
    encoded_ds.save_to_disk(str(data_dir))

    meta = {
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tok_dir),
        "dataset_dir": str(data_dir),
        "max_len": MAX_LEN,
        "num_labels": NUM_LABELS,
        "model_name": MODEL_NAME,
        "total_samples": TOTAL_SAMPLES,
        "splits": SPLITS,
        "seed": SEED
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done!")
    print(f"  Model     -> {model_dir}")
    print(f"  Tokenizer -> {tok_dir}")
    print(f"  Dataset   -> {data_dir}")
    print(f"  Meta      -> {meta_file}")

if __name__ == "__main__":
    main()
