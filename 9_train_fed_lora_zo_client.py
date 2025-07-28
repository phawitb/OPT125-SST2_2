#!/usr/bin/env python3
# zo_lora_client.py <cid> [--use_cpu]
# ZO-SGD Client with LoRA adapters: recv GLOBAL_ADAPTER -> ZO update adapters -> send LOCAL_ADAPTER

import sys
import socket
import json
import torch
import random
import numpy as np
import time
import os
import csv
import psutil
import shutil
from pathlib import Path
from datetime import datetime

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from zo_comm import send_json, recv_json, send_file, recv_file
import config

# usage
if len(sys.argv) < 2:
    print("Usage: python zo_lora_client.py <client_id> [--use_cpu]")
    sys.exit(1)
CID     = int(sys.argv[1])
USE_CPU = "--use_cpu" in sys.argv

# ZO + LoRA config
global config_ZO
config_ZO    = config.ZO_LORA        # single shared dict for ZO+LoRA
SEED         = config_ZO.get("SEED", config.SEED + CID)
LR           = config_ZO.get("LEARNING_RATE", 1e-4)
MU           = config_ZO.get("MU", 1e-3)
P            = config_ZO.get("P", 5)
LOCAL_STEPS  = config_ZO.get("LOCAL_STEPS", 10)
PRINT_EVERY  = config_ZO.get("LOG_EVERY", 2)

# LoRA hyperparams
LORA_R       = config_ZO.get("LORA_R", 16)
LORA_ALPHA   = config_ZO.get("LORA_ALPHA", 32)
LORA_DROPOUT = config_ZO.get("LORA_DROPOUT", 0.1)

# connectivity
SERVER_IP   = config.SERVER_IP
SERVER_PORT = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS
DEVICE      = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

# seeds
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

# artifacts
ART_DIR    = Path("artifacts_custom")
META       = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR  = Path(META["model_dir"])
TOK_DIR    = Path(META["tokenizer_dir"])
DATA_DIR   = Path(META["dataset_dir"])

# prepare data
tokenizer    = AutoTokenizer.from_pretrained(TOK_DIR)
ds           = load_from_disk(str(DATA_DIR))
train_ds     = ds["train"].shard(num_shards=NUM_CLIENTS, index=CID)
train_loader = DataLoader(
    train_ds,
    batch_size=config_ZO.get("BATCH_SIZE", 4),
    shuffle=True,
    collate_fn=default_data_collator
)

# output directory
base_output_dir = Path(f"output/fed_lora_zo/client_{CID}")
base_output_dir.mkdir(parents=True, exist_ok=True)
runs = [d for d in base_output_dir.iterdir()
        if d.is_dir() and d.name.startswith("train_")
        and len(d.name.split("_")) == 2 and d.name.split("_")[1].isdigit()]
next_id = max([int(d.name.split("_")[1]) for d in runs], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)

log_path = run_dir / f"client_{CID}_fed_zo_lora.csv"
if not log_path.exists():
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["round","avg_loss","timestamp","comp_time","comm_time","memory_mb"]
        )
        writer.writeheader()

cfg_dst = run_dir / "config_client_copy.py"
if not cfg_dst.exists() and Path("config.py").exists():
    shutil.copy("config.py", cfg_dst)

# model init: base + LoRA adapters
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
lora_cfg   = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT
)
model = get_peft_model(base_model, lora_cfg).to(DEVICE)
model.print_trainable_parameters()

# helpers
def filter_adapter_state(sd):
    return {k: v for k, v in sd.items() if "lora_" in k}

def count_params(sd):
    return sum(v.numel() for v in sd.values())

def mb(b):
    return f"{b/1024/1024:.2f} MB"

def apply_adapter_weights(model, flat_w, keys):
    sd = filter_adapter_state(model.state_dict())
    ptr = 0
    new_sd = {}
    for k in keys:
        num = sd[k].numel()
        new_sd[k] = flat_w[ptr:ptr+num].view_as(sd[k])
        ptr += num
    model.load_state_dict({**model.state_dict(), **new_sd}, strict=False)

# ZO update on adapters only
def zo_update(batch):
    sd = filter_adapter_state(model.state_dict())
    keys = list(sd.keys())
    w0 = torch.cat([sd[k].view(-1) for k in keys]).to(DEVICE)
    grad_est = torch.zeros_like(w0)
    losses = []

    for i in range(P):
        u = torch.randn_like(w0)
        w_pos = w0 + MU * u
        w_neg = w0 - MU * u
        apply_adapter_weights(model, w_pos, keys)
        L_pos = compute_loss(model, batch)
        apply_adapter_weights(model, w_neg, keys)
        L_neg = compute_loss(model, batch)
        grad_est += ((L_pos - L_neg)/(2*MU)) * u
        losses.append((L_pos + L_neg)/2.0)

    grad_est /= P
    w1 = w0 - LR * grad_est
    apply_adapter_weights(model, w1, keys)
    return np.mean(losses)

@torch.no_grad()
def compute_loss(model, batch):
    out = model(**{k: v.to(DEVICE) for k, v in batch.items()})
    return out.loss.item()

# main loop
if __name__ == "__main__":
    print(f"[Client {CID}] Device: {DEVICE}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))
    proc = psutil.Process(os.getpid())

    loader_iter = iter(train_loader)
    while True:
        msg = recv_json(sock)
        cmd = msg.get("cmd")
        if cmd == "DONE":
            print(f"[Client {CID}] DONE and disconnected.")
            break

        if cmd == "GLOBAL_ADAPTER":
            rnd = msg.get("round")
            print(f"\n[Client {CID}] GLOBAL_ADAPTER r{rnd}")

            # receive global adapter
            sz_in, t_in = recv_file(sock, "global_adapter.pt")
            adapter_sd = torch.load("global_adapter.pt", map_location=DEVICE)
            num_in = count_params(adapter_sd)
            print(f"[recv] params={num_in} size={mb(sz_in)} in {t_in:.2f}s")
            model.load_state_dict(adapter_sd, strict=False)

            # multiple ZO local steps
            total_loss = 0.0
            t0 = time.perf_counter()
            for step in range(1, LOCAL_STEPS + 1):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    batch = next(loader_iter)
                loss = zo_update(batch)
                total_loss += loss
                if step % PRINT_EVERY == 0 or step == LOCAL_STEPS:
                    print(f"[Client {CID}] step {step}/{LOCAL_STEPS} | loss={loss:.4f}")
            comp_time = time.perf_counter() - t0
            avg_loss = total_loss / LOCAL_STEPS

            # send local adapter
            local_sd = filter_adapter_state(model.state_dict())
            num_out = count_params(local_sd)
            out_path = run_dir / f"local_c{CID}_r{rnd}_adapter.pt"
            torch.save(local_sd, out_path)

            send_json(sock, {"cmd": "LOCAL_ADAPTER", "round": rnd})
            sz_out, t_out = send_file(sock, str(out_path))
            print(f" [send] params={num_out} size={mb(sz_out)} in {t_out:.2f}s")

            # log
            mem = proc.memory_info().rss / 1024 / 1024
            with open(log_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["round","avg_loss","timestamp","comp_time","comm_time","memory_mb"]
                )
                writer.writerow({
                    "round": rnd,
                    "avg_loss": avg_loss,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "comp_time": f"{comp_time:.4f}",
                    "comm_time": f"{(t_in + t_out):.4f}",
                    "memory_mb": f"{mem:.2f}"
                })

    sock.close()
