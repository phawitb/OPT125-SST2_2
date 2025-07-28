#!/usr/bin/env python3
# zo_lora_server.py -- ZO-SGD Server with LoRA adapters
# Send GLOBAL adapter -> recv LOCAL adapter -> avg -> eval -> log

import socket
import torch
import json
import time
import os
import shutil
import csv
import psutil
from pathlib import Path
from datetime import datetime

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
from zo_comm import send_json, recv_json, send_file, recv_file
import config

# ---------------- SETUP ----------------
HOST        = "0.0.0.0"
PORT        = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS
# ROUNDS      = config.ROUNDS
ROUNDS      = config.ZO_LORA['ROUNDS']
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LR          = config.ZO.get("LEARNING_RATE", 1e-4)

ART_DIR     = Path("artifacts_custom")
META        = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR   = Path(META["model_dir"])
DATA_DIR    = Path(META["dataset_dir"])

# ---------------- OUTPUT DIRECTORY ----------------
base_output_dir = Path("output/fed_lora_zo/server")
base_output_dir.mkdir(parents=True, exist_ok=True)
# only consider directories named train_<number>
runs = [d for d in base_output_dir.iterdir()
        if d.is_dir() and d.name.startswith("train_")
        and len(d.name.split("_")) == 2 and d.name.split("_")[1].isdigit()]
next_id = max([int(d.name.split("_")[1]) for d in runs], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)

# copy config
shutil.copy("config.py", run_dir / "config.py")

# prepare log CSV
log_path = run_dir / "log_fed_zo_lora.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "round", "train_acc", "val_acc", "test_acc", "val_loss", "timestamp",
        "computation_time", "communication_time", "memory_mb"
    ])
    writer.writeheader()

# ---------------- DATA ----------------
ds = load_from_disk(str(DATA_DIR))
train_loader = DataLoader(ds["train"], batch_size=64, shuffle=False,
                          collate_fn=default_data_collator)
val_loader   = DataLoader(ds["validation"], batch_size=64, shuffle=False,
                          collate_fn=default_data_collator)
test_loader  = DataLoader(ds["test"], batch_size=64, shuffle=False,
                          collate_fn=default_data_collator)

# ---------------- MODEL ----------------
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=True,
    r=config.ZO.get("LORA_R", 16),
    lora_alpha=config.ZO.get("LORA_ALPHA", 32),
    lora_dropout=config.ZO.get("LORA_DROPOUT", 0.1)
)
model = get_peft_model(base_model, peft_config).to(DEVICE)

@torch.inference_mode()
def evaluate(loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**inputs, labels=labels)
        correct += (out.logits.argmax(-1) == labels).sum().item()
        loss_sum += out.loss.item() * labels.size(0)
        total += labels.size(0)
    return correct/total, loss_sum/total

# helper to filter adapter state dict entries
def filter_adapter_state(sd):
    return {k: v for k, v in sd.items() if "lora_" in k}

# helper to count parameters in a state_dict
def count_params(sd): return sum(v.numel() for v in sd.values())

# ---------------- MAIN ROUNDS ----------------
def mb(b): return f"{b/1024/1024:.2f} MB"

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(NUM_CLIENTS)
    print(f"[Server] Listening on {HOST}:{PORT}")

    conns = []
    for i in range(NUM_CLIENTS):
        conn, addr = server.accept()
        print(f"[Server] Client {i} connected from {addr}")
        conns.append((i, conn))

    proc = psutil.Process(os.getpid())
    # initialize global adapter
    global_sd = filter_adapter_state(model.state_dict())
    global_path = run_dir / "global_r0_adapter.pt"
    torch.save(global_sd, global_path)

    for r in range(1, ROUNDS+1):
        print(f"\nRound {r}")
        comm_t = 0.0

        # Send global adapter
        sd_to_send = torch.load(global_path, map_location="cpu")
        num_params = count_params(sd_to_send)
        for cid, conn in conns:
            print(f"[Server→C{cid}] Sending GLOBAL_ADAPTER r{r} | params={num_params}")
            send_json(conn, {"cmd": "GLOBAL_ADAPTER", "round": r})
            sz, t_send = send_file(conn, str(global_path))
            comm_t += t_send
            print(f"[Server→C{cid}] sent ({mb(sz)}) in {t_send:.2f}s")

        # Receive local adapters
        local_sds = []
        for cid, conn in conns:
            msg = recv_json(conn)
            assert msg.get("cmd")=="LOCAL_ADAPTER" and msg.get("round")==r
            local_path = run_dir / f"local_c{cid}_r{r}_adapter.pt"
            sz, t_recv = recv_file(conn, str(local_path))
            comm_t += t_recv
            sd_local = torch.load(local_path, map_location="cpu")
            num_local = count_params(sd_local)
            print(f"[Server←C{cid}] received LOCAL_ADAPTER r{r} | params={num_local} | size={mb(sz)} in {t_recv:.2f}s")
            local_sds.append(sd_local)

        # Average and evaluate
        t0 = time.perf_counter()
        keys = local_sds[0].keys()
        avg_sd = {k: sum(sd[k] for sd in local_sds)/len(local_sds) for k in keys}
        global_path = run_dir / f"global_r{r}_adapter.pt"
        torch.save(avg_sd, global_path)
        model.load_state_dict({**model.state_dict(), **avg_sd})

        train_acc, _ = evaluate(train_loader)
        val_acc, val_loss = evaluate(val_loader)
        test_acc, _ = evaluate(test_loader)
        comp_t = time.perf_counter() - t0
        mem_mb = proc.memory_info().rss / (1024**2)

        print(f"Eval r{r} | train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}")
        print(f"comp={comp_t:.2f}s comm={comm_t:.2f}s mem={mem_mb:.1f}MB")

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "round", "train_acc", "val_acc", "test_acc", "val_loss", "timestamp",
                "computation_time", "communication_time", "memory_mb"
            ])
            writer.writerow({
                "round": r,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "val_loss": val_loss,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "computation_time": f"{comp_t:.4f}",
                "communication_time": f"{comm_t:.4f}",
                "memory_mb": f"{mem_mb:.2f}"
            })

    # cleanup
    for cid, conn in conns:
        send_json(conn, {"cmd": "DONE"})
        conn.close()
    server.close()
    print(f"Training complete. Logs at {log_path}")

if __name__ == "__main__":
    main()
