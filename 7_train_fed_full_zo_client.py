# zo_client.py <cid> [--use_cpu] -- optimized CPU version with batched u and timing logs

import sys, socket, json, torch, random, numpy as np, time, os, csv, psutil, shutil
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
    set_seed,
)
from zo_comm import send_json, recv_json, send_file, recv_file
import config

if len(sys.argv) < 2:
    print("Usage: python zo_client.py <client_id> [--use_cpu]")
    sys.exit(1)

CID = int(sys.argv[1])
USE_CPU = "--use_cpu" in sys.argv

SEED         = config.ZO.get("SEED", config.SEED + CID)
LR           = config.ZO.get("LEARNING_RATE", 1e-4)
BATCH_SIZE   = config.ZO.get("BATCH_SIZE", 4)
MU           = config.ZO.get("MU", 1e-3)
P            = config.ZO.get("P", 5)
LOCAL_STEPS  = config.ZO.get("LOCAL_STEPS", 10)
PRINT_EVERY  = config.ZO.get("LOG_EVERY", 2)

SERVER_IP    = config.SERVER_IP
SERVER_PORT  = config.SERVER_PORT
NUM_CLIENTS  = config.NUM_CLIENTS
DEVICE       = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

ART_DIR   = Path("artifacts_custom")
META      = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR = Path(META["model_dir"])
TOK_DIR   = Path(META["tokenizer_dir"])
DATA_DIR  = Path(META["dataset_dir"])

tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
ds = load_from_disk(str(DATA_DIR))
train_ds = ds["train"].shard(num_shards=NUM_CLIENTS, index=CID)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=default_data_collator)

base_output_dir = Path("output/fed_full_zo/client")
run_dirs = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
run_dir = sorted(run_dirs, key=lambda d: int(d.name.split("_")[1]))[-1] if run_dirs else Path("output/train_client_tmp")
run_dir.mkdir(parents=True, exist_ok=True)

log_path = run_dir / f"client_{CID}_fed_zo.csv"
if not log_path.exists():
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "loss", "lr", "timestamp", "computation_time", "communication_time", "memory_mb"
        ])
        writer.writeheader()

cfg_dst = run_dir / "config_client_copy.py"
if not cfg_dst.exists() and Path("config.py").exists():
    shutil.copy("config.py", cfg_dst)

@torch.no_grad()
def compute_loss(model, batch):
    model.eval()
    ids = batch["input_ids"].to(DEVICE)
    mask = batch["attention_mask"].to(DEVICE)
    lbls = batch["labels"].to(DEVICE)
    out = model(input_ids=ids, attention_mask=mask, labels=lbls)
    return out.loss.item()

def mb(b): return f"{b/1024/1024:.2f} MB"

def main():
    print(f"[Client {CID}] using device: {DEVICE}")
    print(f"[Client {CID}] connecting to {SERVER_IP}:{SERVER_PORT} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))
    print(f"[Client {CID}] connected.")

    proc = psutil.Process(os.getpid())
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    iter_loader = iter(train_loader)

    while True:
        msg = recv_json(sock)
        if msg["cmd"] == "DONE":
            print(f"[Client {CID}] received DONE.")
            break

        if msg["cmd"] == "GLOBAL":
            rnd = msg["round"]
            print(f"\n[Client {CID}] GLOBAL r{rnd}")

            size_in, t_recv = recv_file(sock, "global.pt")
            global_sd = torch.load("global.pt", map_location=DEVICE)
            model.load_state_dict(global_sd)

            w = torch.cat([p.detach().view(-1) for p in model.parameters()]).to(DEVICE)
            total_loss = 0.0
            start_comp = time.perf_counter()

            for step in range(1, LOCAL_STEPS + 1):
                try:
                    batch = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(train_loader)
                    batch = next(iter_loader)

                print(f"[Client {CID}] Step {step}/{LOCAL_STEPS}")
                grad_est = torch.zeros_like(w)
                losses = []

                t_u = time.perf_counter()
                u_batch = torch.randn(P, w.numel(), device=DEVICE)
                print(f"  randn batch time: {time.perf_counter() - t_u:.4f}s")

                for i in range(P):
                    u = u_batch[i]
                    w_pos = w + MU * u
                    w_neg = w - MU * u
                    apply_flat_weights(model, w_pos)
                    L_pos = compute_loss(model, batch)
                    apply_flat_weights(model, w_neg)
                    L_neg = compute_loss(model, batch)
                    grad_est += ((L_pos - L_neg) / (2 * MU)) * u
                    losses.append((L_pos + L_neg) / 2.0)

                grad_est /= P
                avg_loss = np.mean(losses)
                w -= LR * grad_est
                apply_flat_weights(model, w)
                total_loss += avg_loss

                if step % PRINT_EVERY == 0 or step == LOCAL_STEPS:
                    print(f"  avg_loss={avg_loss:.4f}, grad_norm={grad_est.norm().item():.2f}")

            comp_t = time.perf_counter() - start_comp

            local_path = f"local_c{CID}_r{rnd}.pt"
            torch.save(model.state_dict(), local_path)
            send_json(sock, {"cmd": "LOCAL", "round": rnd})
            size_out, t_send = send_file(sock, local_path)
            mem_mb = proc.memory_info().rss / 1024 / 1024

            print(f"[Client {CID}] sent LOCAL r{rnd} ({mb(size_out)} in {t_send:.2f}s)")

            with open(log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "step", "loss", "lr", "timestamp",
                    "computation_time", "communication_time", "memory_mb"
                ])
                writer.writerow({
                    "step": rnd,
                    "loss": total_loss / LOCAL_STEPS,
                    "lr": LR,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "computation_time": f"{comp_t:.4f}",
                    "communication_time": f"{t_recv + t_send:.4f}",
                    "memory_mb": f"{mem_mb:.2f}",
                })

    sock.close()
    print(f"[Client {CID}] disconnected.")

if __name__ == "__main__":
    main()
