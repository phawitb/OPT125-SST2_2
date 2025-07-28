#!/usr/bin/env python3
# train_iter.py  -- iteration-based training using prepared artifacts + config.BACKPROP

import os, json, csv, shutil, random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    default_data_collator,
    set_seed,
)

import config
from datasets import load_from_disk

# ---------------- LOAD CONFIG ----------------
BP = getattr(config, "FULL_BP", {})
SEED         = BP.get("SEED", getattr(config, "SEED", 42))
BATCH_SIZE   = BP.get("BATCH_SIZE", 32)
LR           = BP.get("LEARNING_RATE", 5e-5)
WEIGHT_DECAY = BP.get("WEIGHT_DECAY", 0.0)
MAX_STEPS    = BP.get("MAX_STEPS", 1000)
EVAL_EVERY   = BP.get("EVAL_EVERY", 100)
LOG_EVERY    = BP.get("LOG_EVERY", 50)
SAVE_EVERY   = BP.get("SAVE_EVERY", 500)
GRAD_ACCUM   = BP.get("GRAD_ACCUM", 1)
WARMUP_STEPS = BP.get("WARMUP_STEPS", 100)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ----------- SEED -----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# ----------- ARTIFACT PATHS -----------
ART_DIR   = Path("artifacts_custom")
META      = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR = Path(META["model_dir"])
TOK_DIR   = Path(META["tokenizer_dir"])
DATA_DIR  = Path(META["dataset_dir"])

# ----------- OUTPUT DIRS -----------
base_output_dir = Path("output/full_bp")
base_output_dir.mkdir(parents=True, exist_ok=True)

runs = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in runs], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
(run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
shutil.copy("config.py", run_dir / "config.py")

log_csv = run_dir / "log_full_bp_train.csv"
with open(log_csv, "w", newline="") as f:
    writer = csv.DictWriter(f,
        fieldnames=["step","loss","lr","train_acc","val_acc","test_acc","timestamp"])
    writer.writeheader()

# ----------- LOAD MODEL / DATA -----------
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

ds = load_from_disk(str(DATA_DIR))
train_ds, val_ds, test_ds = ds["train"], ds["validation"], ds["test"]
train_for_test_ds = ds["train"].shuffle(seed=SEED).select(range(600)) 

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=default_data_collator)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=default_data_collator)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=default_data_collator)
train_for_test_loader = DataLoader(train_for_test_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=default_data_collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
)

@torch.inference_mode()
def evaluate(loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        correct += (logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total

# ----------- TRAIN LOOP -----------
step = 0
loss_buf = 0.0
model.train()

while step < MAX_STEPS:
    for batch in train_loader:
        step += 1
        labels = batch.pop("labels").to(DEVICE)
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}

        out  = model(**batch, labels=labels)
        loss = out.loss / GRAD_ACCUM
        loss.backward()

        loss_buf += loss.item()

        if step % GRAD_ACCUM == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # log
        if step % LOG_EVERY == 0:
            avg_loss = loss_buf / LOG_EVERY
            loss_buf = 0.0
            print(f"[step {step}] loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        # eval
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            train_acc = evaluate(train_for_test_loader)
            val_acc   = evaluate(val_loader)
            test_acc  = evaluate(test_loader)

            with open(log_csv, "a", newline="") as f:
                writer = csv.DictWriter(f,
                    fieldnames=["step","loss","lr","train_acc","val_acc","test_acc","timestamp"])
                writer.writerow({
                    "step": step,
                    "loss": avg_loss if 'avg_loss' in locals() else float(loss.item()),
                    "lr": scheduler.get_last_lr()[0],
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "timestamp": datetime.now().isoformat(timespec="seconds")
                })
            print(f"- eval @step {step} | train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f}")

        # save
        if step % SAVE_EVERY == 0 or step == MAX_STEPS:
            ckpt = run_dir / "ckpt" / f"step_{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt, safe_serialization=True)
            tokenizer.save_pretrained(ckpt)
            print(f"saved checkpoint -> {ckpt}")

        if step >= MAX_STEPS:
            break

print("Training complete.")
print(f"CSV log -> {log_csv}")
