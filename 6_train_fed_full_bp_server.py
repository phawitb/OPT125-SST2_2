#!/usr/bin/env python3
# FedAvg Server with SHA256 validation

import socket, json, torch, time, csv, shutil, psutil, os, sys, threading, hashlib
from datetime import datetime
from pathlib import Path
from bp_comm import send_json, recv_json, send_file, recv_file
import config
from transformers import AutoModelForSequenceClassification, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_from_disk

# -------------------- CONFIG --------------------
USE_CPU = "--use_cpu" in sys.argv
HOST        = "0.0.0.0"
PORT        = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS
ROUNDS      = config.FED_FULL_BP.get("ROUNDS", 10)
LR_GLOBAL   = config.FED_FULL_BP.get("LEARNING_RATE", 5e-5)
SAVE_MODEL  = config.FED_FULL_BP.get("SAVE_MODEL", False)
SEED        = config.SEED
DEVICE      = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

ART_DIR  = Path("artifacts_custom")
META     = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR= Path(META["model_dir"])
DATA_DIR = Path(META["dataset_dir"])

base_output_dir = Path("output/fed_full_bp/server")
base_output_dir.mkdir(parents=True, exist_ok=True)
existing = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in existing], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)
shutil.copy("config.py", run_dir / "config.py")

log_csv = run_dir / "log_fed_bp.csv"
fieldnames = [
    "step", "loss", "lr", "train_acc", "val_acc", "test_acc",
    "timestamp", "computation_time", "communication_time", "memory_mb",
    "send_time_sec", "recv_time_sec", "avg_eval_time_sec"
]
for cid in range(NUM_CLIENTS):
    fieldnames += [f"sent_MB_C{cid}", f"send_time_C{cid}", f"recv_MB_C{cid}", f"recv_time_C{cid}"]

with open(log_csv, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()

# -------------------- HELPERS --------------------
def mb(b): return f"{b / 1024 / 1024:.2f}"
def now(): return datetime.now().strftime("%H:%M:%S")
def sha256sum(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

@torch.inference_mode()
def eval_split(loader):
    server_model.eval()
    correct = total = loss_sum = 0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = server_model(**batch, labels=labels)
        loss_sum += out.loss.item() * labels.size(0)
        correct += (out.logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total, loss_sum / total

def avg_state_dict(states):
    keys = states[0].keys()
    return {k: torch.stack([s[k] for s in states], dim=0).mean(dim=0) for k in keys}

# -------------------- MAIN --------------------
ds_all = load_from_disk(str(DATA_DIR))
train_dl = DataLoader(ds_all["train"].shuffle(seed=SEED).select(range(600)), batch_size=64, collate_fn=default_data_collator)
val_dl   = DataLoader(ds_all["validation"], batch_size=64, collate_fn=default_data_collator)
test_dl  = DataLoader(ds_all["test"], batch_size=64, collate_fn=default_data_collator)

server_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)

def main():
    print(f"[{now()}] [Server] Starting on {DEVICE}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((HOST, PORT))
    srv.listen(NUM_CLIENTS)

    conns = []
    for cid in range(NUM_CLIENTS):
        conn, addr = srv.accept()
        print(f"[{now()}] [Server] Client {cid} connected from {addr}")
        conns.append((cid, conn))

    init_path = run_dir / "global_r0.pt"
    if not init_path.exists():
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        torch.save(model.state_dict(), init_path)

    current_global = init_path
    proc = psutil.Process(os.getpid())

    for r in range(1, ROUNDS + 1):
        print(f"\n[{now()}] [Server] === ROUND {r} START ===")
        client_stats = {}

        def send_task(cid, conn):
            print(f"[{now()}] [Server→C{cid}] Sending {current_global.name}")
            send_json(conn, {"cmd": "GLOBAL", "round": r})
            size, t = send_file(conn, str(current_global))
            hash_val = sha256sum(current_global)
            print(f"[{now()}] [Server→C{cid}] SHA256 = {hash_val}")
            print(f"[{now()}] [Server→C{cid}] Sent {mb(size)} in {t:.3f}s")
            client_stats[cid] = {
                f"sent_MB_C{cid}": mb(size),
                f"send_time_C{cid}": f"{t:.4f}"
            }

        send_threads = [threading.Thread(target=send_task, args=(cid, conn)) for cid, conn in conns]
        [t.start() for t in send_threads]
        [t.join() for t in send_threads]

        local_paths = [None] * NUM_CLIENTS
        def recv_task(cid, conn):
            print(f"[{now()}] [Server←C{cid}] Waiting for LOCAL model...")
            msg = recv_json(conn)
            assert msg["cmd"] == "LOCAL" and msg["round"] == r
            path = run_dir / f"local_c{cid}_r{r}.pt"
            size, t = recv_file(conn, str(path))
            hash_val = sha256sum(path)
            print(f"[{now()}] [Server←C{cid}] SHA256 = {hash_val}")
            print(f"[{now()}] [Server←C{cid}] Received {mb(size)} in {t:.3f}s")
            client_stats[cid][f"recv_MB_C{cid}"] = mb(size)
            client_stats[cid][f"recv_time_C{cid}"] = f"{t:.4f}"
            local_paths[cid] = path

        recv_threads = [threading.Thread(target=recv_task, args=(cid, conn)) for cid, conn in conns]
        [t.start() for t in recv_threads]
        [t.join() for t in recv_threads]

        print(f"[{now()}] [Server] Averaging and evaluating...")
        t0 = time.perf_counter()
        states = [torch.load(p, map_location="cpu") for p in local_paths]
        global_sd = avg_state_dict(states)
        current_global = run_dir / f"global_r{r}.pt"
        torch.save(global_sd, current_global)

        if SAVE_MODEL:
            print(f"[{now()}] [Server] Saved global model to {current_global.name}")
        else:
            print(f"[{now()}] [Server] Saved temp global model for communication only (SAVE_MODEL=False)")
            for p in local_paths:
                if p.exists():
                    os.remove(p)

        server_model.load_state_dict(global_sd)
        train_acc, _ = eval_split(train_dl)
        val_acc, val_loss = eval_split(val_dl)
        test_acc, _ = eval_split(test_dl)
        eval_time = time.perf_counter() - t0
        mem_mb = proc.memory_info().rss / (1024 ** 2)

        print(f"[{now()}] [Server] ROUND {r} EVAL → train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f} test_acc={test_acc:.4f} | "
              f"comp={eval_time:.2f}s comm=N/A mem={mem_mb:.1f}MB")

        row = {
            "step": r, "loss": val_loss, "lr": LR_GLOBAL,
            "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "computation_time": f"{eval_time:.4f}", "communication_time": "0.0000",
            "memory_mb": f"{mem_mb:.2f}", "send_time_sec": "0.0000",
            "recv_time_sec": "0.0000", "avg_eval_time_sec": f"{eval_time:.4f}"
        }
        for cid in range(NUM_CLIENTS):
            row.update(client_stats[cid])

        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

        print(f"[{now()}] [Server] === ROUND {r} END ===")

    for _, conn in conns:
        send_json(conn, {"cmd": "DONE"})
        conn.close()
    srv.close()
    print(f"[{now()}] [Server] Training completed. Logs saved at {log_csv}")

if __name__ == "__main__":
    main()
