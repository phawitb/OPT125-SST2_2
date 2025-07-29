#!/usr/bin/env python3
# bp_client.py <cid> [--use_cpu] -- client trains after GLOBAL and sends LOCAL back
# Logs to output/train_{id}/client_{cid}_fed_bp.csv

import sys, socket, json, torch, random, numpy as np, time, csv, psutil, os, shutil
from pathlib import Path
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
from datetime import datetime

# --------- ARGS ---------
if len(sys.argv) < 2:
    print("Usage: python bp_client.py <cid> [--use_cpu]")
    sys.exit(1)
CID = int(sys.argv[1])
USE_CPU = "--use_cpu" in sys.argv

# --------- CONFIG ---------
SERVER_IP   = config.SERVER_IP
SERVER_PORT = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS

BP = config.FED_FULL_BP
SEED         = BP.get("SEED", getattr(config, "SEED", 42))
BATCH_SIZE   = BP.get("BATCH_SIZE", 32)
LR           = BP.get("LEARNING_RATE", 5e-5)
LOCAL_STEPS  = BP.get("LOCAL_STEPS", 50)
PRINT_EVERY  = BP.get("LOG_EVERY", 10)
SAVE_MODEL   = BP.get("SAVE_MODEL", False)
DEVICE       = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED + CID); np.random.seed(SEED + CID); torch.manual_seed(SEED + CID); set_seed(SEED + CID)

# --------- PATHS ---------
ART_DIR  = Path("artifacts_custom")
META     = json.loads((ART_DIR / "meta.json").read_text())
TOK_DIR  = Path(META["tokenizer_dir"])
MODEL_DIR= Path(META["model_dir"])
DATA_DIR = Path(META["dataset_dir"])

base_output_dir = Path(f"output/fed_full_bp/client_{CID}")
base_output_dir.mkdir(parents=True, exist_ok=True)
existing = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in existing], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)

client_log = run_dir / f"client_{CID}_fed_bp.csv"
if not client_log.exists():
    with open(client_log, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step","loss","lr","timestamp",
                "computation_time","communication_time","memory_mb",
                "recv_time_sec","train_time_sec","send_time_sec"
            ]
        )
        writer.writeheader()

cfg_dst = run_dir / "config_client_copy.py"
if not cfg_dst.exists() and Path("config.py").exists():
    shutil.copy("config.py", cfg_dst)

# --------- LOADER SETUP ---------
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
ds = load_from_disk(str(DATA_DIR))
train_shard = ds["train"].shard(num_shards=NUM_CLIENTS, index=CID % NUM_CLIENTS)
loader = DataLoader(train_shard, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=default_data_collator)

# --------- HELPERS ---------
def now(): return datetime.now().strftime("%H:%M:%S")
def mb(b): return f"{b / 1024 / 1024:.2f} MB"

def local_train(model) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step = 0
    it = iter(loader)
    running = 0.0
    while step < LOCAL_STEPS:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)

        labels = batch.pop("labels").to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch, labels=labels)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad()
        step += 1
        running += loss.item()

        if step % PRINT_EVERY == 0 or step == LOCAL_STEPS:
            denom = PRINT_EVERY if step % PRINT_EVERY == 0 else (step % PRINT_EVERY)
            print(f"[{now()}] [Client {CID}] step {step}/{LOCAL_STEPS} | loss={running/denom:.4f}")

    return running / LOCAL_STEPS

# --------- MAIN ---------
def main():
    proc = psutil.Process(os.getpid())

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((SERVER_IP, SERVER_PORT))
        print(f"[{now()}] [Client {CID}] Connected to {SERVER_IP}:{SERVER_PORT} | device={DEVICE}")

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)

        while True:
            msg = recv_json(sock)
            cmd = msg.get("cmd")

            if cmd == "GLOBAL":
                round_id = msg.get("round")
                global_path = run_dir / f"global_r{round_id}.pt"

                print(f"[{now()}] [Client {CID}] Receiving global model r{round_id}...")
                t0_recv = time.perf_counter()
                size_recv, t_recv = recv_file(sock, str(global_path))
                t1_recv = time.perf_counter()
                print(f"[{now()}] [Client {CID}] Received global ({mb(size_recv)}) in {t_recv:.3f}s")

                try:
                    sd = torch.load(global_path, map_location=DEVICE, weights_only=True)
                except TypeError:
                    sd = torch.load(global_path, map_location=DEVICE)
                model.load_state_dict(sd)
                os.remove(global_path)

                print(f"[{now()}] [Client {CID}] Starting local training...")
                t0_train = time.perf_counter()
                avg_loss = local_train(model)
                t1_train = time.perf_counter()
                train_time = t1_train - t0_train
                print(f"[{now()}] [Client {CID}] Training done in {train_time:.2f}s | avg_loss={avg_loss:.4f}")

                out_path = run_dir / f"local_c{CID}_r{round_id}.pt"
                torch.save(model.state_dict(), out_path)
                if SAVE_MODEL:
                    print(f"[{now()}] [Client {CID}] Saved local model: {out_path.name}")
                else:
                    print(f"[{now()}] [Client {CID}] Saved temp model for upload only (SAVE_MODEL=False)")

                send_json(sock, {"cmd": "LOCAL", "round": round_id})
                t0_send = time.perf_counter()
                size_send, t_send = send_file(sock, out_path)
                t1_send = time.perf_counter()
                send_time = t1_send - t0_send
                print(f"[{now()}] [Client {CID}] Sent local ({mb(size_send)}) in {t_send:.3f}s")

                if not SAVE_MODEL and out_path.exists():
                    os.remove(out_path)
                    print(f"[{now()}] [Client {CID}] Deleted temp model after sending")

                mem_mb = proc.memory_info().rss / (1024**2)

                with open(client_log, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "step", "loss", "lr", "timestamp",
                        "computation_time", "communication_time", "memory_mb",
                        "recv_time_sec", "train_time_sec", "send_time_sec"
                    ])
                    writer.writerow({
                        "step": round_id,
                        "loss": avg_loss,
                        "lr": LR,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "computation_time": f"{train_time:.4f}",
                        "communication_time": f"{t_recv + t_send:.4f}",
                        "memory_mb": f"{mem_mb:.2f}",
                        "recv_time_sec": f"{t_recv:.4f}",
                        "train_time_sec": f"{train_time:.4f}",
                        "send_time_sec": f"{t_send:.4f}",
                    })

            elif cmd == "DONE":
                print(f"[{now()}] [Client {CID}] Received DONE from server. Exiting...")
                break

            else:
                raise RuntimeError(f"Unknown command received: {cmd}")

if __name__ == "__main__":
    main()
