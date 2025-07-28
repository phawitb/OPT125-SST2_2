#!/usr/bin/env python3
# bp_lora_client.py <cid>  -- client trains LoRA adapters after receiving GLOBAL_ADAPTER
# Logs to output/train_full_bp_lora/client_{cid}_fed_bp_lora.csv

import sys
import socket
import json
import torch
import random
import numpy as np
import time
import csv
import psutil
import os
import shutil
from pathlib import Path
from datetime import datetime

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from bp_comm import send_json, recv_json, send_file, recv_file
import config

def count_params(state_dict):
    """Count total number of elements in a state_dict of tensors."""
    return sum(v.numel() for v in state_dict.values())

# usage
if len(sys.argv) != 2:
    print("Usage: python bp_lora_client.py <cid>")
    sys.exit(1)
CID = int(sys.argv[1])

# server connection
SERVER_IP   = config.SERVER_IP
SERVER_PORT = config.SERVER_PORT

# config sections
BP_LORA = config.BACKPROP_LORA
SEED         = BP_LORA.get("SEED", 42) + CID
BATCH_SIZE   = BP_LORA.get("BATCH_SIZE", 32)
LR           = BP_LORA.get("LEARNING_RATE", 5e-5)
LOCAL_STEPS  = BP_LORA.get("LOCAL_STEPS", 50)
PRINT_EVERY  = BP_LORA.get("LOG_EVERY", 10)

# LoRA hyperparams
LORA_R       = BP_LORA.get("LORA_R", 16)
LORA_ALPHA   = BP_LORA.get("LORA_ALPHA", 32)
LORA_DROPOUT = BP_LORA.get("LORA_DROPOUT", 0.1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# artifact paths
ART_DIR   = Path("artifacts_custom")
META      = json.loads((ART_DIR / "meta.json").read_text())
TOK_DIR   = Path(META["tokenizer_dir"])
MODEL_DIR = Path(META["model_dir"])
DATA_DIR  = Path(META["dataset_dir"])

# prepare tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
ds = load_from_disk(str(DATA_DIR))
# split train shard for this client
train_shard = ds["train"].shard(num_shards=config.NUM_CLIENTS, index=CID)
loader = DataLoader(
    train_shard,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=default_data_collator
)

# run dir & logging
base_output_dir = Path("output/fed_lora_bp/client")
base_output_dir.mkdir(exist_ok=True)
existing = [
    d for d in base_output_dir.iterdir()
    if d.name.startswith("train_") and d.name.split("_")[1].isdigit()
]
if existing:
    run_dir = sorted(existing, key=lambda p: int(p.name.split("_")[1]))[-1]
else:
    run_dir = base_output_dir / "train_0"
    run_dir.mkdir(parents=True, exist_ok=True)

client_log = run_dir / f"client_{CID}_fed_bp_lora.csv"
if not client_log.exists():
    with open(client_log, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round","loss","lr","timestamp",
                "computation_time","communication_time","memory_mb"
            ]
        )
        writer.writeheader()

cfg_dst = run_dir / "config_client_copy.py"
if not cfg_dst.exists() and Path("config.py").exists():
    shutil.copy("config.py", cfg_dst)

# model init: load base + LoRA adapters
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True
)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,  # train adapters
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT
)
model = get_peft_model(base_model, lora_config).to(DEVICE)
model.print_trainable_parameters()

# helper to filter only adapter weights
def filter_adapter_state(sd):
    # match any key containing "lora_"
    return {k: v for k, v in sd.items() if "lora_" in k}

# training function
def local_train():
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    running_loss = 0.0
    it = iter(loader)
    for step in range(1, LOCAL_STEPS + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        labels = batch.pop("labels").to(DEVICE)
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        if step % PRINT_EVERY == 0 or step == LOCAL_STEPS:
            avg = running_loss / (PRINT_EVERY if step % PRINT_EVERY == 0 else step % PRINT_EVERY)
            print(f"[Client {CID}] step {step}/{LOCAL_STEPS} | loss={avg:.4f}")
            running_loss = 0.0
    return running_loss / LOCAL_STEPS

# main loop
if __name__ == "__main__":
    proc = psutil.Process(os.getpid())
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((SERVER_IP, SERVER_PORT))
        print(f"[Client {CID}] connected to {SERVER_IP}:{SERVER_PORT}")
        while True:
            msg = recv_json(sock)
            cmd = msg.get("cmd")

            if cmd == "GLOBAL_ADAPTER":
                r = msg.get("round")

                # receive global adapter
                size_recv, t_recv = recv_file(sock, "global_adapter.pt")
                global_sd = torch.load("global_adapter.pt", map_location=DEVICE)
                num_recv = count_params(global_sd)
                print(f"[Client {CID}] recv adapter r{r} | params={num_recv} | size={size_recv/1024/1024:.4f}MB in {t_recv:.3f}s")

                # load into model
                model.load_state_dict(global_sd, strict=False)

                # local training
                t0 = time.perf_counter()
                avg_loss = local_train()
                comp_t = time.perf_counter() - t0

                # extract and save local adapter
                local_sd = filter_adapter_state(model.state_dict())
                num_send = count_params(local_sd)
                out_path = f"local_c{CID}_r{r}.pt"
                torch.save(local_sd, out_path)

                # send local adapter
                send_json(sock, {"cmd":"LOCAL_ADAPTER","round":r})
                size_send, t_send = send_file(sock, out_path)
                print(f"[Client {CID}] sent adapter r{r} | params={num_send} | size={size_send/1024/1024:.4f}MB in {t_send:.3f}s")

                # log
                mem_mb = proc.memory_info().rss / (1024**2)
                with open(client_log, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "round","loss","lr","timestamp",
                            "computation_time","communication_time","memory_mb"
                        ]
                    )
                    writer.writerow({
                        "round": r,
                        "loss": avg_loss,
                        "lr": LR,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "computation_time": f"{comp_t:.4f}",
                        "communication_time": f"{(t_recv + t_send):.4f}",
                        "memory_mb": f"{mem_mb:.2f}"
                    })

            elif cmd == "DONE":
                print(f"[Client {CID}] DONE")
                break

            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
