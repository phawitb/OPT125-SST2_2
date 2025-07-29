#!/usr/bin/env python3
# FedAvg Server (synchronous): GLOBAL -> wait all LOCAL -> avg/eval -> GLOBAL ...
# Logs to output/fed_full_bp/server/train_{id}/log_fed_bp.csv

import socket, json, torch, time, csv, shutil, psutil, os, sys
from datetime import datetime
from pathlib import Path
from bp_comm import send_json, recv_json, send_file, recv_file
import config
from transformers import AutoModelForSequenceClassification, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_from_disk

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

client_fields = []
for cid in range(NUM_CLIENTS):
    client_fields += [f"sent_MB_C{cid}", f"send_time_C{cid}", f"recv_MB_C{cid}", f"recv_time_C{cid}"]

log_csv = run_dir / "log_fed_bp.csv"
with open(log_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "step", "loss", "lr", "train_acc", "val_acc", "test_acc",
        "timestamp", "computation_time", "communication_time", "memory_mb",
        "send_time_sec", "recv_time_sec", "avg_eval_time_sec"
    ] + client_fields)
    writer.writeheader()

def mb(b): return f"{b / 1024 / 1024:.2f}"
def now(): return datetime.now().strftime("%H:%M:%S")

ds_all = load_from_disk(str(DATA_DIR))
train_for_test_dl = DataLoader(ds_all["train"].shuffle(seed=SEED).select(range(600)), batch_size=64, shuffle=False, collate_fn=default_data_collator)
val_dl  = DataLoader(ds_all["validation"], batch_size=64, shuffle=False, collate_fn=default_data_collator)
test_dl = DataLoader(ds_all["test"], batch_size=64, shuffle=False, collate_fn=default_data_collator)

server_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)

@torch.inference_mode()
def eval_split(loader):
    server_model.eval()
    correct = total = loss_sum = 0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = server_model(**batch, labels=labels)
        logits = out.logits
        loss_sum += out.loss.item() * labels.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total, loss_sum / total

def avg_state_dict(states):
    keys = states[0].keys()
    return {k: sum(sd[k] for sd in states) / len(states) for k in keys}

def main():
    print(f"[{now()}] [Server] Starting on device: {DEVICE}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((HOST, PORT))
    srv.listen(NUM_CLIENTS)
    print(f"[{now()}] [Server] Listening on {HOST}:{PORT}")

    conns = []
    for cid in range(NUM_CLIENTS):
        conn, addr = srv.accept()
        print(f"[{now()}] [Server] Client {cid} connected from {addr}")
        conns.append((cid, conn))

    init_path = run_dir / "global_r0.pt"
    if not init_path.exists():
        base = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        torch.save(base.state_dict(), init_path)
    current_global = init_path
    prev_global = None
    proc = psutil.Process(os.getpid())

    for r in range(1, ROUNDS + 1):
        print(f"\n[{now()}] [Server] === ROUND {r} START ===")
        comm_t = comp_t = 0.0
        client_stats = {cid: {} for cid, _ in conns}

        # 1) Send GLOBAL
        send_t0 = time.perf_counter()
        for cid, conn in conns:
            send_json(conn, {"cmd": "GLOBAL", "round": r})
            size, t = send_file(conn, str(current_global))
            comm_t += t
            client_stats[cid]["sent_MB"] = mb(size)
            client_stats[cid]["send_time"] = f"{t:.4f}"
            print(f"[{now()}] [Server→C{cid}] Sent {mb(size)} in {t:.3f}s")
        send_t1 = time.perf_counter()
        send_time = send_t1 - send_t0

        if not SAVE_MODEL and prev_global and prev_global.exists():
            os.remove(prev_global)
            print(f"[{now()}] [Server] Deleted temp global model {prev_global.name}")

        # 2) Receive LOCAL
        recv_t0 = time.perf_counter()
        local_paths = []
        for cid, conn in conns:
            msg = recv_json(conn)
            print(f"[{now()}] [Server←C{cid}] Waiting for LOCAL model...")
            assert msg.get("cmd") == "LOCAL" and msg.get("round") == r
            lp = run_dir / f"local_c{cid}_r{r}.pt"
            size, t = recv_file(conn, str(lp))
            comm_t += t
            print(f"[{now()}] [Server←C{cid}] Received {mb(size)} in {t:.3f}s")
            client_stats[cid]["recv_MB"] = mb(size)
            client_stats[cid]["recv_time"] = f"{t:.4f}"
            local_paths.append(lp)
        recv_t1 = time.perf_counter()
        recv_time = recv_t1 - recv_t0

        # 3) Average and Evaluate
        print(f"[{now()}] [Server] Averaging and evaluating...")
        t0 = time.perf_counter()
        states = [torch.load(lp, map_location="cpu") for lp in local_paths]
        global_sd = avg_state_dict(states)
        avg_eval_t0 = time.perf_counter()
        prev_global = current_global
        current_global = run_dir / f"global_r{r}.pt"
        torch.save(global_sd, current_global)

        if SAVE_MODEL:
            print(f"[{now()}] [Server] Saved global model to {current_global.name}")
        else:
            print(f"[{now()}] [Server] Saved temp global model for communication only (SAVE_MODEL=False)")
            for f in local_paths:
                os.remove(f)

        server_model.load_state_dict(global_sd)
        train_acc, _ = eval_split(train_for_test_dl)
        val_acc, val_loss = eval_split(val_dl)
        test_acc, _ = eval_split(test_dl)
        avg_eval_t1 = time.perf_counter()
        comp_t = avg_eval_t1 - t0
        avg_eval_time = avg_eval_t1 - avg_eval_t0
        mem_mb = proc.memory_info().rss / (1024**2)

        print(f"[{now()}] [Server] ROUND {r} EVAL → train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f} test_acc={test_acc:.4f} | "
              f"comp={comp_t:.2f}s comm={comm_t:.2f}s mem={mem_mb:.1f}MB")

        row = {
            "step": r, "loss": val_loss, "lr": LR_GLOBAL,
            "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "computation_time": f"{comp_t:.4f}", "communication_time": f"{comm_t:.4f}",
            "memory_mb": f"{mem_mb:.2f}", "send_time_sec": f"{send_time:.4f}",
            "recv_time_sec": f"{recv_time:.4f}", "avg_eval_time_sec": f"{avg_eval_time:.4f}"
        }

        for cid in range(NUM_CLIENTS):
            row[f"sent_MB_C{cid}"] = client_stats[cid].get("sent_MB", "0.00")
            row[f"send_time_C{cid}"] = client_stats[cid].get("send_time", "0.0000")
            row[f"recv_MB_C{cid}"] = client_stats[cid].get("recv_MB", "0.00")
            row[f"recv_time_C{cid}"] = client_stats[cid].get("recv_time", "0.0000")

        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        print(f"[{now()}] [Server] === ROUND {r} END ===")

    for cid, conn in conns:
        send_json(conn, {"cmd": "DONE"})
        conn.close()
    srv.close()
    print(f"[{now()}] [Server] Training completed. Logs saved at {log_csv}")

if __name__ == "__main__":
    main()
