#!/usr/bin/env python3
# FedAvg Server for FULL ZO-SGD (sends/receives full parameter vector and averages)

import os, sys, json, csv, time, socket, shutil, hashlib, threading, psutil
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, default_data_collator
from datasets import load_from_disk

from bp_comm import send_json, recv_json, send_file, recv_file
import config

# -------------------- CONFIG --------------------
USE_CPU     = "--use_cpu" in sys.argv
HOST        = "0.0.0.0"
PORT        = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS

FL = getattr(config, "FED_FULL_ZO", getattr(config, "FULL_ZO", {}))
ROUNDS       = FL.get("ROUNDS", 20)            # federated rounds
BATCH_SIZE   = FL.get("EVAL_BATCH_SIZE", 64)   # server eval only
SAVE_MODEL   = FL.get("SAVE_MODEL", False)
SEED         = getattr(config, "SEED", 42)

DEVICE = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

# ---- artifacts ----
ART_DIR   = Path("artifacts_custom")
META      = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR = Path(META["model_dir"])
DATA_DIR  = Path(META["dataset_dir"])

# ---- output ----
base_output_dir = Path("output/fed_full_zo/server")
base_output_dir.mkdir(parents=True, exist_ok=True)
runs = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in runs], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)
if Path("config.py").exists():
    shutil.copy("config.py", run_dir / "config.py")

log_csv = run_dir / "log_fed_full_zo.csv"
fieldnames = [
    "round","train_acc","val_acc","test_acc","timestamp",
    "avg_vec_mb_sent","avg_vec_mb_recv","mem_mb","avg_eval_time_sec"
] + [f"sent_MB_C{i}" for i in range(NUM_CLIENTS)] \
  + [f"send_time_C{i}" for i in range(NUM_CLIENTS)] \
  + [f"recv_MB_C{i}" for i in range(NUM_CLIENTS)] \
  + [f"recv_time_C{i}" for i in range(NUM_CLIENTS)]
with open(log_csv, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()

# -------------------- HELPERS --------------------
def now(): return datetime.now().strftime("%H:%M:%S")
def mb(nbytes): return nbytes / 1024 / 1024
def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

@torch.inference_mode()
def eval_acc(model, loader, device):
    model.eval()
    total = correct = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        correct += (logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total

# ---- flatten/unflatten based on model parameter order ----
def params_to_vector(model) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def vector_to_params_(model, vec: torch.Tensor):
    # in-place copy from vec to model.parameters()
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            num = p.numel()
            p.copy_(vec[offset:offset+num].view_as(p))
            offset += num

# -------------------- LOAD DATA/MODEL --------------------
torch.manual_seed(SEED)
ds = load_from_disk(str(DATA_DIR))
eval_train = DataLoader(ds["train"].shuffle(seed=SEED).select(range(600)), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=default_data_collator)
eval_val   = DataLoader(ds["validation"], batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=default_data_collator)
eval_test  = DataLoader(ds["test"], batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=default_data_collator)

server_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)

# initial global vector
global_vec = params_to_vector(server_model).cpu()
gpath = run_dir / "global_r0.pt"
torch.save(global_vec, gpath)

proc = psutil.Process(os.getpid())

# -------------------- MAIN --------------------
def main():
    print(f"[{now()}] [Server] Starting on {DEVICE}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((HOST, PORT)); srv.listen(NUM_CLIENTS)

    conns = []
    for cid in range(NUM_CLIENTS):
        conn, addr = srv.accept()
        print(f"[{now()}] [Server] Client {cid} connected from {addr}")
        conns.append((cid, conn))

    current_global_path = str(gpath)

    for r in range(1, ROUNDS+1):
        print(f"\n[{now()}] [Server] === ROUND {r} START ===")
        round_stats = {}

        # ---- SEND GLOBAL VEC ----
        path_to_send = current_global_path
        def send_task(cid, conn, path=path_to_send):
            send_json(conn, {"cmd":"GLOBAL","round":r})
            size, t = send_file(conn, path)
            print(f"[{now()}] [Server→C{cid}] Sent vec {mb(size):.2f} MB in {t:.3f}s | sha={sha256sum(path)[:10]}..")
            round_stats[f"sent_MB_C{cid}"] = f"{mb(size):.2f}"
            round_stats[f"send_time_C{cid}"] = f"{t:.4f}"
        threads = [threading.Thread(target=send_task, args=(cid,conn)) for cid,conn in conns]
        [t.start() for t in threads]; [t.join() for t in threads]

        # ---- RECV LOCAL VECS ----
        local_paths = [None]*NUM_CLIENTS
        def recv_task(cid, conn):
            try:
                msg = recv_json(conn)
                assert msg["cmd"]=="LOCAL" and msg["round"]==r
                p = run_dir / f"local_c{cid}_r{r}.pt"
                size, t = recv_file(conn, str(p))
                print(f"[{now()}] [Server←C{cid}] Recv vec {mb(size):.2f} MB in {t:.3f}s | sha={sha256sum(p)[:10]}..")
                round_stats[f"recv_MB_C{cid}"]  = f"{mb(size):.2f}"
                round_stats[f"recv_time_C{cid}"]= f"{t:.4f}"
                local_paths[cid] = p
            except Exception as e:
                print(f"[{now()}] [Server←C{cid}] RECV ERROR: {e}")
        threads = [threading.Thread(target=recv_task, args=(cid,conn)) for cid,conn in conns]
        [t.start() for t in threads]; [t.join() for t in threads]

        valid = [p for p in local_paths if p is not None]
        if not valid:
            print(f"[{now()}] [Server] No valid vectors received; stopping.")
            break

        # ---- AVERAGE ----
        t0 = time.perf_counter()
        vecs = [torch.load(p, map_location="cpu").float() for p in valid]
        avg = torch.stack(vecs, dim=0).mean(dim=0)
        # apply to server model for evaluation
        vector_to_params_(server_model, avg.to(DEVICE))
        # save new global vec
        new_gpath = run_dir / f"global_r{r}.pt"
        torch.save(avg.cpu(), new_gpath)
        if not SAVE_MODEL:
            for p in valid:
                try: os.remove(p)
                except: pass
        current_global_path = str(new_gpath)

        # ---- EVAL ----
        train_acc = eval_acc(server_model, eval_train, DEVICE)
        val_acc   = eval_acc(server_model, eval_val, DEVICE)
        test_acc  = eval_acc(server_model, eval_test, DEVICE)
        eval_time = time.perf_counter() - t0
        mem_mb    = proc.memory_info().rss/1024/1024

        # ---- LOG ----
        row = {
            "round": r,
            "train_acc": f"{train_acc:.4f}",
            "val_acc":   f"{val_acc:.4f}",
            "test_acc":  f"{test_acc:.4f}",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "avg_vec_mb_sent": f"{sum(float(round_stats.get(f'sent_MB_C{i}','0') or 0) for i in range(NUM_CLIENTS))/max(1,NUM_CLIENTS):.2f}",
            "avg_vec_mb_recv": f"{sum(float(round_stats.get(f'recv_MB_C{i}','0') or 0) for i in range(NUM_CLIENTS))/max(1,NUM_CLIENTS):.2f}",
            "mem_mb": f"{mem_mb:.2f}",
            "avg_eval_time_sec": f"{eval_time:.4f}",
        }
        for i in range(NUM_CLIENTS):
            row[f"sent_MB_C{i}"]   = round_stats.get(f"sent_MB_C{i}",   "")
            row[f"send_time_C{i}"] = round_stats.get(f"send_time_C{i}", "")
            row[f"recv_MB_C{i}"]   = round_stats.get(f"recv_MB_C{i}",   "")
            row[f"recv_time_C{i}"] = round_stats.get(f"recv_time_C{i}", "")
        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(f"[{now()}] [Server] ROUND {r} | train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}")

    # close
    for _,conn in conns:
        try: send_json(conn, {"cmd":"DONE"})
        except: pass
        conn.close()
    srv.close()
    print(f"[{now()}] [Server] Finished. Logs -> {log_csv}")
    # optional final model save
    if SAVE_MODEL:
        final_dir = run_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        server_model.save_pretrained(final_dir, safe_serialization=True)
        print(f"[{now()}] [Server] Final model saved to {final_dir}")

if __name__ == "__main__":
    main()
