#!/usr/bin/env python3
# bp_client.py <cid>  -- client trains immediately after GLOBAL and sends LOCAL back
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

if len(sys.argv) != 2:
    print("Usage: python bp_client.py <cid>")
    sys.exit(1)
CID = int(sys.argv[1])

SERVER_IP   = config.SERVER_IP
SERVER_PORT = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS

BP = config.FED_FULL_BP
SEED         = BP.get("SEED", getattr(config, "SEED", 42))
BATCH_SIZE   = BP.get("BATCH_SIZE", 32)
LR           = BP.get("LEARNING_RATE", 5e-5)
LOCAL_STEPS  = BP.get("LOCAL_STEPS", 50)
PRINT_EVERY  = BP.get("LOG_EVERY", 10)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); set_seed(SEED)

ART_DIR  = Path("artifacts_custom")
META     = json.loads((ART_DIR / "meta.json").read_text())
TOK_DIR  = Path(META["tokenizer_dir"])
MODEL_DIR= Path(META["model_dir"])
DATA_DIR = Path(META["dataset_dir"])

# --- run dir / logging ---
# base_output_dir = Path("output/fed_full_bp")
# base_output_dir.mkdir(parents=True, exist_ok=True)
# existing = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
# next_id = max([int(d.name.split("_")[1]) for d in existing], default=0) + 1
# run_dir = base_output_dir / f"train_{next_id}" / "server"
# run_dir.mkdir(parents=True, exist_ok=True)


base_output_dir = Path(f"output/fed_full_bp/client_{CID}")
base_output_dir.mkdir(parents=True, exist_ok=True)
existing = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in existing], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"  # no server subdir
run_dir.mkdir(parents=True, exist_ok=True)

# # pick the latest train_* (server and clients should agree; simplest)
# if existing:
#     run_dir = sorted(existing, key=lambda p: int(p.name.split("_")[1]))[-1]
# else:
#     # fallback create
#     run_dir = base_output_dir / "train_client_tmp"
#     run_dir.mkdir(exist_ok=True)

client_log = run_dir / f"client_{CID}_fed_bp.csv"
if not client_log.exists():
    with open(client_log, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step","loss","lr","timestamp","computation_time","communication_time","memory_mb"]
        )
        writer.writeheader()

# copy config once (if not already copied by server or another client)
cfg_dst = run_dir / "config_client_copy.py"
if not cfg_dst.exists() and Path("config.py").exists():
    shutil.copy("config.py", cfg_dst)

tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
ds = load_from_disk(str(DATA_DIR))

train_shard = ds["train"].shard(num_shards=NUM_CLIENTS, index=CID % NUM_CLIENTS)
loader = DataLoader(train_shard, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=default_data_collator)

def mb(b): return f"{b/1024/1024:.2f} MB"

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
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch, labels=labels)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad()
        step += 1
        running += loss.item()

        if step % PRINT_EVERY == 0 or step == LOCAL_STEPS:
            denom = PRINT_EVERY if step % PRINT_EVERY == 0 else (step % PRINT_EVERY)
            print(f"[Client {CID}] step {step}/{LOCAL_STEPS} | loss={running/denom:.4f}")

    return running / LOCAL_STEPS  # avg loss

def main():
    proc = psutil.Process(os.getpid())

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((SERVER_IP, SERVER_PORT))
        print(f"[Client {CID}] connected -> {SERVER_IP}:{SERVER_PORT}")

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)

        while True:
            msg = recv_json(sock)
            cmd = msg.get("cmd")

            if cmd == "GLOBAL":

                global_path = run_dir / "global.pt"
                size, t_recv = recv_file(sock, str(global_path))

                print("global_path:", global_path)
                # # recv global
                # size, t_recv = recv_file(sock, f"{run_dir}/global.pt")
                try:
                    sd = torch.load(global_path, map_location=DEVICE, weights_only=True)
                except TypeError:
                    sd = torch.load(global_path, map_location=DEVICE)
                model.load_state_dict(sd)
                print(f"[Client {CID}] recv global (r{msg['round']}) {mb(size)} {t_recv:.3f}s")

                # compute
                t0 = time.perf_counter()
                avg_loss = local_train(model)
                comp_t = time.perf_counter() - t0

                # send local
                out_path = f"{run_dir}/local_c{CID}_r{msg['round']}.pt"
                torch.save(model.state_dict(), out_path)
                send_json(sock, {"cmd": "LOCAL", "round": msg["round"]})
                size2, t_send = send_file(sock, out_path)
                print(f"[Client {CID}] sent local  (r{msg['round']}) {mb(size2)} {t_send:.3f}s")

                mem_mb = proc.memory_info().rss / (1024**2)

                with open(client_log, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["step","loss","lr","timestamp","computation_time","communication_time","memory_mb"]
                    )
                    writer.writerow({
                        "step": msg["round"],
                        "loss": avg_loss,
                        "lr": LR,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "computation_time": f"{comp_t:.4f}",
                        "communication_time": f"{t_recv + t_send:.4f}",
                        "memory_mb": f"{mem_mb:.2f}",
                    })

            elif cmd == "DONE":
                print(f"[Client {CID}] DONE")
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")

if __name__ == "__main__":
    main()
