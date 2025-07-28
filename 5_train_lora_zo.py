#!/usr/bin/env python3
# train_iter_zo_lora.py  -- iteration-based ZO-SGD training on LoRA adapters

import os
import json
import csv
import shutil
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

import config
from datasets import load_from_disk

# ---------------- CONFIG ----------------
ZO = getattr(config, "LORA_ZO", {})
SEED         = ZO.get("SEED", getattr(config, "SEED", 42))
BATCH_SIZE   = ZO.get("BATCH_SIZE", 32)
LR           = ZO.get("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = ZO.get("WEIGHT_DECAY", 0.0)
MAX_STEPS    = ZO.get("MAX_STEPS", 1000)
EVAL_EVERY   = ZO.get("EVAL_EVERY", 100)
LOG_EVERY    = ZO.get("LOG_EVERY", 50)
SAVE_EVERY   = ZO.get("SAVE_EVERY", 500)
GRAD_ACCUM   = ZO.get("GRAD_ACCUM", 1)
MU           = ZO.get("MU", 1e-3)
P            = ZO.get("P", 5)

# LoRA-specific config
LORA_R        = ZO.get("LORA_R", 16)
LORA_ALPHA    = ZO.get("LORA_ALPHA", 32)
LORA_DROPOUT  = ZO.get("LORA_DROPOUT", 0.1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- SEED ------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# ------------- ARTIFACT PATHS -----------
ART_DIR   = Path("artifacts_custom")
META_PATH = ART_DIR / "meta.json"
if not META_PATH.exists():
    raise FileNotFoundError("artifacts_custom/meta.json not found. Run load_model_data.py first.")
META      = json.loads(META_PATH.read_text())
MODEL_DIR = Path(META["model_dir"])
TOK_DIR   = Path(META["tokenizer_dir"])
DATA_DIR  = Path(META["dataset_dir"])

# ------------- OUTPUT DIRS --------------
base_output_dir = Path("output/lora_zo")
base_output_dir.mkdir(exist_ok=True)
runs = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in runs], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
(run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
shutil.copy("config.py", run_dir / "config.py")

log_csv = run_dir / "log_full_zo_lora_train.csv"
with open(log_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["step", "loss", "lr", "train_acc", "val_acc", "test_acc", "timestamp"]
    )
    writer.writeheader()

# ------------- LOAD MODEL & WRAP LoRA ----------
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)

# load base model (frozen)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# configure LoRA adapters
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT
)
# inject LoRA and freeze base parameters
model = get_peft_model(base_model, lora_config).to(DEVICE)
# only adapter params require grad
model.print_trainable_parameters()

# load dataset
_ds = load_from_disk(str(DATA_DIR))
train_ds, val_ds, test_ds = _ds["train"], _ds["validation"], _ds["test"]
train_for_test_ds = _ds["train"].shuffle(seed=SEED).select(range(600)) 

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=default_data_collator
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=default_data_collator
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=default_data_collator
)
train_for_test_loader = DataLoader(train_for_test_ds, batch_size=BATCH_SIZE, shuffle=True,collate_fn=default_data_collator)


# ------------- FLATTEN HELPERS (LoRA only) ----------
# get only trainable LoRA params
params_list  = [p for p in model.parameters() if p.requires_grad]
param_sizes  = [p.numel() for p in params_list]
param_shapes = [p.shape for p in params_list]
offsets      = np.cumsum([0] + param_sizes)

def params_to_vector(params):
    return torch.cat([p.detach().view(-1) for p in params])

def vector_to_params(vec):
    return [vec[offsets[i]:offsets[i+1]].view(param_shapes[i]) for i in range(len(params_list))]

def update_lora_params(vec):
    with torch.no_grad():
        for p, new in zip(params_list, vector_to_params(vec)):
            p.copy_(new)

@torch.no_grad()
def compute_loss(batch):
    model.eval()
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    return model(**batch).loss.item()

@torch.inference_mode()
def evaluate(loader):
    model.eval()
    total = correct = 0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        correct += (logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total

# ------------- TRAIN LOOP ---------------
print("\nZO-SGD on LoRA adapters (iteration-based)")

# initialize flat vector of LoRA params
w = params_to_vector(params_list).to(DEVICE)
step = 0
loss_buf = 0.0
train_iter = iter(train_loader)

while step < MAX_STEPS:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    step += 1
    base_w = w.clone()

    grad_est = torch.zeros_like(w)
    losses   = []

    for _ in range(P):
        u = torch.randn_like(w)
        w_pos = base_w + MU * u
        w_neg = base_w - MU * u

        update_lora_params(w_pos)
        L_pos = compute_loss(batch)

        update_lora_params(w_neg)
        L_neg = compute_loss(batch)

        grad_est += ((L_pos - L_neg) / (2 * MU)) * u
        losses.append((L_pos + L_neg) / 2.0)

    grad_est /= P
    avg_loss = float(np.mean(losses))
    loss_buf += avg_loss

    if WEIGHT_DECAY > 0.0:
        grad_est += WEIGHT_DECAY * base_w

    if step % GRAD_ACCUM == 0:
        w = base_w - LR * grad_est
        update_lora_params(w)

    if step % LOG_EVERY == 0:
        avg_loss_step = loss_buf / LOG_EVERY
        loss_buf = 0.0
        norm = grad_est.norm().item()
        print(f"[step {step}] loss={avg_loss_step:.4f} | grad_norm={norm:.2f}")

    if step % EVAL_EVERY == 0 or step == MAX_STEPS:
        train_acc = evaluate(train_for_test_loader)
        val_acc   = evaluate(val_loader)
        test_acc  = evaluate(test_loader)

        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "loss", "lr", "train_acc", "val_acc", "test_acc", "timestamp"]
            )
            writer.writerow({
                "step": step,
                "loss": avg_loss_step if 'avg_loss_step' in locals() else avg_loss,
                "lr": LR,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "timestamp": datetime.now().isoformat(timespec="seconds")
            })
        print(f"- eval @step {step} | train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f}")

    if step % SAVE_EVERY == 0 or step == MAX_STEPS:
        ckpt = run_dir / "ckpt" / f"step_{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        # save full model + LoRA adapter
        model.save_pretrained(ckpt, safe_serialization=True)
        tokenizer.save_pretrained(ckpt)
        torch.save({"lora_w": w.cpu()}, ckpt / "lora_weights.pt")
        print(f"saved checkpoint -> {ckpt}")

print("Training complete.")

# final save
final_dir = run_dir / "opt-sst2-zo-lora-finetuned"
final_dir.mkdir(exist_ok=True)
model.save_pretrained(final_dir, safe_serialization=True)
tokenizer.save_pretrained(final_dir)
print(f"Model + LoRA adapters saved to {final_dir}")
print(f"Logs saved to {log_csv}")
