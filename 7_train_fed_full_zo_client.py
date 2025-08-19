#!/usr/bin/env python3
# FedAvg Client for FULL ZO-SGD (receives full parameter vector, local ZO steps, sends updated vector)

import os, sys, json, csv, time, socket, shutil, hashlib, random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator, set_seed
from datasets import load_from_disk

from bp_comm import send_json, recv_json, send_file, recv_file
import config

# -------------------- ARGS --------------------
if len(sys.argv) < 2:
    print("Usage: python 9_train_fed_full_zo_client.py <cid> [--use_cpu]")
    sys.exit(1)
CID = int(sys.argv[1])
USE_CPU = "--use_cpu" in sys.argv

# -------------------- CONFIG --------------------
SERVER_IP   = config.SERVER_IP
SERVER_PORT = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS

ZO = getattr(config, "FED_FULL_ZO", getattr(config, "FULL_ZO", {}))
SEED          = ZO.get("SEED", getattr(config, "SEED", 42))
BATCH_SIZE    = ZO.get("BATCH_SIZE", 32)
LR            = ZO.get("LEARNING_RATE", 1e-4)
WEIGHT_DECAY  = ZO.get("WEIGHT_DECAY", 0.0)
LOCAL_STEPS   = ZO.get("LOCAL_STEPS", 50)      # per-round local steps
LOG_EVERY     = ZO.get("LOG_EVERY", 10)
MU            = ZO.get("MU", 1e-3)
P             = ZO.get("P", 5)                 # number of random directions per step

DEVICE = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- SEED --------------------
random.seed(SEED + CID)
np.random.seed(SEED + CID)
torch.manual_seed(SEED + CID)
set_seed(SEED + CID)

# -------------------- ARTIFACTS ----------------
ART_DIR   = Path("artifacts_custom")
META      = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR = Path(META["model_dir"])
TOK_DIR   = Path(META["tokenizer_dir"])
DATA_DIR  = Path(META["dataset_dir"])

# -------------------- OUTPUT -------------------
base_output_dir = Path(f"output/fed_full_zo/client_{CID}")
base_output_dir.mkdir(parents=True, exist_ok=True)
runs = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in runs], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)
if Path("config.py").exists():
    shutil.copy("config.py", run_dir / "config_client_copy.py")

log_csv = run_dir / f"client_{CID}_fed_full_zo.csv"
if not Path(log_csv).exists():
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "round","last_loss","lr","timestamp",
            "recv_time_sec","train_time_sec","send_time_sec",
            "comm_mb_recv","comm_mb_send","mem_mb"
        ])
        writer.writeheader()

# -------------------- HELPERS -------------------
def now(): return datetime.now().strftime("%H:%M:%S")
def mb(nbytes): return nbytes / 1024 / 1024
def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

# ---- flatten/unflatten ----
def params_to_vector(model) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def vector_to_params_(model, vec: torch.Tensor):
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            num = p.numel()
            p.copy_(vec[offset:offset+num].view_as(p))
            offset += num

@torch.no_grad()
def compute_loss(model, batch, device):
    model.eval()
    ids  = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    lbls = batch["labels"].to(device)
    out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
    return out.loss.item()

# -------------------- DATA/MODEL --------------
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)

ds = load_from_disk(str(DATA_DIR))
train_shard = ds["train"].shard(num_shards=NUM_CLIENTS, index=CID % NUM_CLIENTS)
train_loader = DataLoader(train_shard, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

# -------------------- LOCAL ZO LOOP ----------
def local_zo_sgd_round(w_init: torch.Tensor) -> (torch.Tensor, float):
    """
    รับเวกเตอร์พารามิเตอร์ global แล้วทำ ZO steps LOCAL_STEPS ครั้งบน shard ของ client
    คืนค่าพารามิเตอร์ที่อัปเดตแล้ว และ last_loss สำหรับล็อก
    """
    # apply global to model & keep working vector on DEVICE
    w = w_init.to(DEVICE).clone()
    vector_to_params_(model, w)
    loss_last = 0.0

    data_iter = iter(train_loader)
    step = 0
    while step < LOCAL_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        step += 1
        base_w = w.clone()
        grad_est = torch.zeros_like(w)
        losses = []

        for _ in range(P):
            u = torch.randn_like(w)           # random direction
            w_pos = base_w + MU * u
            w_neg = base_w - MU * u

            vector_to_params_(model, w_pos)
            L_pos = compute_loss(model, batch, DEVICE)

            vector_to_params_(model, w_neg)
            L_neg = compute_loss(model, batch, DEVICE)

            grad_est += ((L_pos - L_neg) / (2 * MU)) * u
            losses.append(0.5 * (L_pos + L_neg))

        grad_est /= P
        if WEIGHT_DECAY > 0.0:
            grad_est += WEIGHT_DECAY * base_w

        # SGD update on the vector
        w = base_w - LR * grad_est
        vector_to_params_(model, w)  # keep model in sync for next step
        loss_last = float(np.mean(losses))

        if step % LOG_EVERY == 0 or step == LOCAL_STEPS:
            print(f"[{now()}] [Client {CID}] step {step}/{LOCAL_STEPS} | loss={loss_last:.4f}")

    return w.detach().cpu(), loss_last

# -------------------- MAIN --------------------
def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((SERVER_IP, SERVER_PORT))
        print(f"[{now()}] [Client {CID}] Connected to {SERVER_IP}:{SERVER_PORT} | device={DEVICE}")

        while True:
            msg = recv_json(sock)
            cmd = msg.get("cmd")
            if cmd == "GLOBAL":
                r = msg["round"]
                gpath = run_dir / f"global_r{r}.pt"
                print(f"[{now()}] [Client {CID}] Receiving global vector r{r}...")
                t0 = time.perf_counter()
                size_recv, t_recv = recv_file(sock, str(gpath))
                print(f"[{now()}] [Client {CID}] Recv vec {mb(size_recv):.2f} MB in {t_recv:.3f}s | sha={sha256sum(gpath)[:10]}..")

                # load vector (tensor or dict{'w':tensor})
                obj = torch.load(gpath, map_location="cpu")
                w_global = obj["w"] if isinstance(obj, dict) and "w" in obj else obj
                os.remove(gpath)

                # local train
                t_train0 = time.perf_counter()
                w_local, last_loss = local_zo_sgd_round(w_global)
                t_train = time.perf_counter() - t_train0

                # send local vector
                lpath = run_dir / f"local_c{CID}_r{r}.pt"
                torch.save(w_local, lpath)
                send_json(sock, {"cmd":"LOCAL","round":r})
                size_send, t_send = send_file(sock, str(lpath))
                print(f"[{now()}] [Client {CID}] Sent vec {mb(size_send):.2f} MB in {t_send:.3f}s | sha={sha256sum(lpath)[:10]}..")
                try: os.remove(lpath)
                except: pass

                # log
                mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
                with open(log_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "round","last_loss","lr","timestamp",
                        "recv_time_sec","train_time_sec","send_time_sec",
                        "comm_mb_recv","comm_mb_send","mem_mb"
                    ])
                    writer.writerow({
                        "round": r,
                        "last_loss": f"{last_loss:.6f}",
                        "lr": LR,
                        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        "recv_time_sec": f"{t_recv:.4f}",
                        "train_time_sec": f"{t_train:.4f}",
                        "send_time_sec": f"{t_send:.4f}",
                        "comm_mb_recv": f"{mb(size_recv):.2f}",
                        "comm_mb_send": f"{mb(size_send):.2f}",
                        "mem_mb": f"{mem_mb:.2f}",
                    })

            elif cmd == "DONE":
                print(f"[{now()}] [Client {CID}] DONE from server. Bye.")
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")

if __name__ == "__main__":
    import psutil  # used in logging above
    main()
