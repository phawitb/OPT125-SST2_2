#!/usr/bin/env python3
# FedAvg Client for ZO-LoRA: รับ LoRA → ทำ ZO steps เฉพาะ LoRA → ส่ง LoRA กลับ

import os, sys, json, csv, time, socket, shutil, hashlib, random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator, set_seed
from datasets import load_from_disk

from peft import (
    LoraConfig, get_peft_model, TaskType,
    get_peft_model_state_dict, set_peft_model_state_dict,
)

from bp_comm import send_json, recv_json, send_file, recv_file
import config

# -------------------- ARGS --------------------
if len(sys.argv) < 2:
    print("Usage: python 10_train_fed_zo_lora_client.py <cid> [--use_cpu]")
    sys.exit(1)
CID = int(sys.argv[1])
USE_CPU = "--use_cpu" in sys.argv

# -------------------- CONFIG --------------------
SERVER_IP   = config.SERVER_IP
SERVER_PORT = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS

FL = getattr(config, "FED_ZO_LORA", getattr(config, "FULL_ZO", {}))
SEED          = FL.get("SEED", getattr(config, "SEED", 42))
BATCH_SIZE    = FL.get("BATCH_SIZE", 32)
LR            = FL.get("LEARNING_RATE", 1e-4)
WEIGHT_DECAY  = FL.get("WEIGHT_DECAY", 0.0)
LOCAL_STEPS   = FL.get("LOCAL_STEPS", 50)
LOG_EVERY     = FL.get("LOG_EVERY", 10)
MU            = FL.get("MU", 1e-3)
P             = FL.get("P", 5)  # random directions per step

LORA_R       = FL.get("LORA_R", 16)
LORA_ALPHA   = FL.get("LORA_ALPHA", 32)
LORA_DROPOUT = FL.get("LORA_DROPOUT", 0.1)

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
base_output_dir = Path(f"output/fed_zo_lora/client_{CID}")
base_output_dir.mkdir(parents=True, exist_ok=True)
runs = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in runs], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)
if Path("config.py").exists():
    shutil.copy("config.py", run_dir / "config_client_copy.py")

log_csv = run_dir / f"client_{CID}_fed_zo_lora.csv"
if not Path(log_csv).exists():
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round","last_loss","lr","timestamp",
                "recv_time_sec","train_time_sec","send_time_sec",
                "comm_mb_recv","comm_mb_send","mem_mb"
            ]
        ); writer.writeheader()

# -------------------- HELPERS -------------------
def now(): return datetime.now().strftime("%H:%M:%S")
def mb(n): return f"{n/1024/1024:.2f} MB"
def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for ch in iter(lambda: fp.read(1<<20), b""):
            h.update(ch)
    return h.hexdigest()

# Backward-compat apply (handle old peft without 'strict')
def safe_set_peft_model_state_dict(model, peft_sd):
    try:
        set_peft_model_state_dict(model, peft_sd, strict=False)
        return
    except TypeError:
        pass
    current = get_peft_model_state_dict(model)
    filtered = {k: v for k, v in peft_sd.items() if k in current}
    set_peft_model_state_dict(model, filtered)

# ---- LoRA flatten/unflatten (use sorted keys for deterministic order) ----
def lora_template(model) -> Tuple[List[str], List[torch.Size], List[torch.dtype]]:
    sd = get_peft_model_state_dict(model)
    keys = sorted(sd.keys())
    shapes = [sd[k].shape for k in keys]
    dtypes = [sd[k].dtype for k in keys]
    return keys, shapes, dtypes

def lora_sd_to_vec(sd: Dict[str, torch.Tensor], keys: List[str], device: str) -> torch.Tensor:
    vecs = [sd[k].reshape(-1).to(device=device, dtype=torch.float32) for k in keys]
    return torch.cat(vecs, dim=0)

def vec_to_lora_sd(vec: torch.Tensor, keys: List[str], shapes: List[torch.Size], dtypes: List[torch.dtype]) -> Dict[str, torch.Tensor]:
    sd = {}
    off = 0
    for k, sh, dt in zip(keys, shapes, dtypes):
        n = int(torch.tensor(sh).prod().item()) if isinstance(sh, torch.Size) else int(np.prod(sh))
        chunk = vec[off:off+n].view(sh).to(dtype=dt)
        sd[k] = chunk
        off += n
    return sd

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
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False,
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj","k_proj","v_proj","out_proj"],  # OPT family
)
model = get_peft_model(base_model, lora_cfg).to(DEVICE)

ds = load_from_disk(str(DATA_DIR))
train_shard = ds["train"].shard(num_shards=NUM_CLIENTS, index=CID % NUM_CLIENTS)
train_loader = DataLoader(train_shard, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

# -------------------- LOCAL ZO ON LoRA --------
def local_zo_on_lora(global_sd: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], float):
    # apply received global LoRA
    safe_set_peft_model_state_dict(model, global_sd)

    # template (keys/shapes/dtypes) after applying global
    keys, shapes, dtypes = lora_template(model)

    # working vector w on DEVICE
    w = lora_sd_to_vec(get_peft_model_state_dict(model), keys, device=DEVICE)
    last_loss = 0.0

    it = iter(train_loader)
    step = 0
    while step < LOCAL_STEPS:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)

        step += 1
        base_w = w.clone()
        grad_est = torch.zeros_like(w)
        losses = []

        for _ in range(P):
            u = torch.randn_like(w)

            w_pos = base_w + MU * u
            sd_pos = vec_to_lora_sd(w_pos, keys, shapes, dtypes)
            safe_set_peft_model_state_dict(model, sd_pos)
            L_pos = compute_loss(model, batch, DEVICE)

            w_neg = base_w - MU * u
            sd_neg = vec_to_lora_sd(w_neg, keys, shapes, dtypes)
            safe_set_peft_model_state_dict(model, sd_neg)
            L_neg = compute_loss(model, batch, DEVICE)

            grad_est += ((L_pos - L_neg) / (2 * MU)) * u
            losses.append(0.5 * (L_pos + L_neg))

        grad_est /= P
        if WEIGHT_DECAY > 0.0:
            grad_est += WEIGHT_DECAY * base_w

        # update & apply
        w = base_w - LR * grad_est
        sd_new = vec_to_lora_sd(w, keys, shapes, dtypes)
        safe_set_peft_model_state_dict(model, sd_new)
        last_loss = float(np.mean(losses))

        if step % LOG_EVERY == 0 or step == LOCAL_STEPS:
            print(f"[{now()}] [Client {CID}] step {step}/{LOCAL_STEPS} | loss={last_loss:.4f}")

    # return final LoRA sd (on CPU for transmission)
    final_sd = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
    return final_sd, last_loss

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
                gpath = run_dir / f"lora_global_r{r}.pt"
                print(f"[{now()}] [Client {CID}] Receiving global LoRA r{r}...")
                size_recv, t_recv = recv_file(sock, str(gpath))
                print(f"[{now()}] [Client {CID}] Recv {mb(size_recv)} in {t_recv:.3f}s | sha={sha256sum(gpath)[:10]}..")

                global_sd = torch.load(gpath, map_location="cpu")
                try: os.remove(gpath)
                except: pass

                t0 = time.perf_counter()
                local_sd, last_loss = local_zo_on_lora(global_sd)
                t_train = time.perf_counter() - t0

                lpath = run_dir / f"lora_local_c{CID}_r{r}.pt"
                torch.save(local_sd, lpath)
                send_json(sock, {"cmd":"LOCAL", "round":r})
                size_send, t_send = send_file(sock, str(lpath))
                print(f"[{now()}] [Client {CID}] Sent {mb(size_send)} in {t_send:.3f}s | sha={sha256sum(lpath)[:10]}..")
                try: os.remove(lpath)
                except: pass

                import psutil
                mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
                with open(log_csv, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=[
                        "round","last_loss","lr","timestamp",
                        "recv_time_sec","train_time_sec","send_time_sec",
                        "comm_mb_recv","comm_mb_send","mem_mb"
                    ]).writerow({
                        "round": r,
                        "last_loss": f"{last_loss:.6f}",
                        "lr": LR,
                        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        "recv_time_sec": f"{t_recv:.4f}",
                        "train_time_sec": f"{t_train:.4f}",
                        "send_time_sec": f"{t_send:.4f}",
                        "comm_mb_recv": f"{size_recv/1024/1024:.2f}",
                        "comm_mb_send": f"{size_send/1024/1024:.2f}",
                        "mem_mb": f"{mem_mb:.2f}",
                    })

            elif cmd == "DONE":
                print(f"[{now()}] [Client {CID}] DONE from server. Bye.")
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
