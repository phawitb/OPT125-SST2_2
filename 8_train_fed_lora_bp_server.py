#!/usr/bin/env python3
# zo_server.py -- ZO‑SGD Server: send GLOBAL → recv LOCAL → avg → eval → log

import socket
import torch
import json
import time
import os
import shutil
import csv
import psutil
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    default_data_collator,
)
from zo_comm import send_json, recv_json, send_file, recv_file
import config

# ---------------- SETUP ----------------
HOST        = "0.0.0.0"
PORT        = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS
ROUNDS      = config.ROUNDS
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LR          = config.ZO.get("LEARNING_RATE", 1e-4)

ART_DIR   = Path("artifacts_custom")
META      = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR = Path(META["model_dir"])
DATA_DIR  = Path(META["dataset_dir"])

# ---------------- OUT DIR ----------------
base_output_dir = Path("output/fed_lora_bp/server")
base_output_dir.mkdir(exist_ok=True)
runs = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in runs], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)

shutil.copy("config.py", run_dir / "config.py")
log_path = run_dir / "log_fed_zo.csv"

with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "step", "loss", "lr", "train_acc", "val_acc", "test_acc", "timestamp",
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
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

@torch.inference_mode()
def evaluate(loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**inputs).logits
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total

@torch.inference_mode()
def compute_loss(loader):
    model.eval()
    total_loss = 0.0
    total = 0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**inputs, labels=labels)
        total_loss += out.loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total

def avg_state_dict(list_of_sd):
    avg = {}
    for k in list_of_sd[0].keys():
        avg[k] = sum(sd[k] for sd in list_of_sd) / len(list_of_sd)
    return avg

# ---------------- HELPERS ----------------
def mb(bytes_): 
    return f"{bytes_ / 1024**2:.2f} MB"

def count_params(state_dict):
    """Count total elements in a state_dict."""
    return sum(v.numel() for v in state_dict.values())

# ---------------- MAIN ----------------
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

    # save initial global weights
    global_path = run_dir / "global_r0.pt"
    torch.save(model.state_dict(), global_path)

    for r in range(1, ROUNDS + 1):
        print(f"\nRound {r}")
        comm_t = 0.0

        # === Send GLOBAL ===
        global_sd = torch.load(global_path, map_location="cpu", weights_only=True)
        num_global = count_params(global_sd)
        for cid, conn in conns:
            print(f"[Server→C{cid}] Sending GLOBAL round {r} | params={num_global}")
            send_json(conn, {"cmd": "GLOBAL", "round": r})
            sz, t = send_file(conn, str(global_path))
            comm_t += t
            print(f"[Server→C{cid}] global {mb(sz)} in {t:.2f}s")

        # === Receive LOCAL ===
        sd_list = []
        for cid, conn in conns:
            msg = recv_json(conn)
            assert msg.get("cmd") == "LOCAL" and msg.get("round") == r
            local_path = run_dir / f"local_c{cid}_r{r}.pt"
            sz, t = recv_file(conn, str(local_path))
            comm_t += t

            # load local update
            try:
                local_sd = torch.load(local_path, map_location="cpu", weights_only=True)
            except TypeError:
                local_sd = torch.load(local_path, map_location="cpu")

            num_local = count_params(local_sd)
            print(f"[Server←C{cid}] Received LOCAL round {r} | params={num_local}")
            print(f"[Server←C{cid}] local {mb(sz)} in {t:.2f}s")

            sd_list.append(local_sd)

        # === Average + Eval ===
        t0 = time.perf_counter()
        avg_sd = avg_state_dict(sd_list)
        model.load_state_dict(avg_sd)
        global_path = run_dir / f"global_r{r}.pt"
        torch.save(avg_sd, global_path)

        train_acc = evaluate(train_loader)
        val_acc   = evaluate(val_loader)
        test_acc  = evaluate(test_loader)
        val_loss  = compute_loss(val_loader)

        comp_t = time.perf_counter() - t0
        mem_mb = proc.memory_info().rss / (1024**2)

        print(f"Eval r{r} | train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}")
        print(f"comp={comp_t:.2f}s  comm={comm_t:.2f}s  mem={mem_mb:.1f}MB")

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "step", "loss", "lr", "train_acc", "val_acc", "test_acc", "timestamp",
                "computation_time", "communication_time", "memory_mb"
            ])
            writer.writerow({
                "step": r,
                "loss": val_loss,
                "lr": LR,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "computation_time": f"{comp_t:.4f}",
                "communication_time": f"{comm_t:.4f}",
                "memory_mb": f"{mem_mb:.2f}",
            })

    # signal DONE
    for cid, conn in conns:
        send_json(conn, {"cmd": "DONE"})
        conn.close()
    server.close()
    print(f"Training complete. Logs saved to: {log_path}")

if __name__ == "__main__":
    main()
