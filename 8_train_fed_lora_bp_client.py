#!/usr/bin/env python3
# 8_train_fed_lora_bp_client.py — Client (LoRA + BP)
# รับ global LoRA → เทรน LoRA เฉพาะฝั่ง client → ส่ง LoRA กลับ

import sys, socket, json, torch, random, numpy as np, time, csv, psutil, os, shutil, hashlib
from pathlib import Path
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader
from datasets import load_from_disk

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from bp_comm import send_json, recv_json, send_file, recv_file
import config

# --------- ARGS ---------
if len(sys.argv) < 2:
    print("Usage: python 8_train_fed_lora_bp_client.py <cid> [--use_cpu]")
    sys.exit(1)
CID = int(sys.argv[1])
USE_CPU = "--use_cpu" in sys.argv

# --------- CONFIG ---------
SERVER_IP   = config.SERVER_IP
SERVER_PORT = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS

FL = getattr(config, "FED_LORA_BP", getattr(config, "BACKPROP_LORA", {}))
SEED         = FL.get("SEED", getattr(config, "SEED", 42))
BATCH_SIZE   = FL.get("BATCH_SIZE", 4)
LR           = FL.get("LEARNING_RATE", 5e-5)
LOCAL_STEPS  = FL.get("LOCAL_STEPS", 50)
PRINT_EVERY  = FL.get("LOG_EVERY", 2)
SAVE_MODEL   = FL.get("SAVE_MODEL", False)

# LoRA hyperparams (ต้องตรงกับ server)
LORA_R       = FL.get("LORA_R", 16)
LORA_ALPHA   = FL.get("LORA_ALPHA", 32)
LORA_DROPOUT = FL.get("LORA_DROPOUT", 0.1)

DEVICE = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED + CID); np.random.seed(SEED + CID); torch.manual_seed(SEED + CID); set_seed(SEED + CID)

# --------- PATHS ---------
ART_DIR  = Path("artifacts_custom")
META     = json.loads((ART_DIR / "meta.json").read_text())
TOK_DIR  = Path(META["tokenizer_dir"])
MODEL_DIR= Path(META["model_dir"])
DATA_DIR = Path(META["dataset_dir"])

base_output_dir = Path(f"output/fed_lora_bp/client_{CID}")
base_output_dir.mkdir(parents=True, exist_ok=True)
existing = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in existing], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)

client_log = run_dir / f"client_{CID}_fed_lora_bp.csv"
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

# --------- HELPERS ---------
def now(): return datetime.now().strftime("%H:%M:%S")
def mb(b): return f"{b / 1024 / 1024:.2f} MB"
def sha256sum(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# --- Backward-compatible setter for PEFT state dict ---
def safe_set_peft_model_state_dict(model, peft_sd):
    """
    ทำให้รองรับทั้ง peft เวอร์ชันใหม่/เก่า:
    - ถ้าเวอร์ชันใหม่: พยายามเรียกด้วย strict=False
    - ถ้าเก่า: จะ TypeError -> กรองคีย์ให้เหลือเฉพาะที่โมเดลนี้มี แล้วค่อยใส่
    """
    try:
        # บางเวอร์ชันใหม่รองรับอาร์กิวเมนต์ strict
        set_peft_model_state_dict(model, peft_sd, strict=False)  # อาจจะ TypeError ถ้าเวอร์ชันเก่า
        return
    except TypeError:
        pass  # ตกมาทางเก่า

    current = get_peft_model_state_dict(model)  # คีย์ที่โมเดลนี้คาดหวัง
    filtered = {k: v for k, v in peft_sd.items() if k in current}
    set_peft_model_state_dict(model, filtered)

# --------- LOADERS ---------
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
ds = load_from_disk(str(DATA_DIR))
train_shard = ds["train"].shard(num_shards=NUM_CLIENTS, index=CID % NUM_CLIENTS)
loader = DataLoader(train_shard, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=default_data_collator)

# --------- MODEL (LoRA) ---------
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    # สำคัญ: ต้องตรงกับ server (OPT)
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
)
model = get_peft_model(base_model, lora_cfg).to(DEVICE)

def local_train_lora(model) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step = 0
    it = iter(loader)
    running = 0.0
    last_loss = 0.0
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
        last_loss = float(loss.item())

        if step % PRINT_EVERY == 0 or step == LOCAL_STEPS:
            denom = PRINT_EVERY if step % PRINT_EVERY == 0 else (step % PRINT_EVERY)
            print(f"[{now()}] [Client {CID}] step {step}/{LOCAL_STEPS} | loss={running/denom:.4f}")
            running = 0.0 if step % PRINT_EVERY == 0 else running
    return last_loss

# --------- MAIN ---------
def main():
    proc = psutil.Process(os.getpid())

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((SERVER_IP, SERVER_PORT))
        print(f"[{now()}] [Client {CID}] Connected to {SERVER_IP}:{SERVER_PORT} | device={DEVICE}")

        while True:
            msg = recv_json(sock)
            cmd = msg.get("cmd")

            if cmd == "GLOBAL":
                round_id = msg.get("round")
                global_path = run_dir / f"lora_global_r{round_id}.pt"

                print(f"[{now()}] [Client {CID}] Receiving global LoRA r{round_id}...")
                t0_recv = time.perf_counter()
                size_recv, t_recv = recv_file(sock, str(global_path))
                t1_recv = time.perf_counter()
                print(f"[{now()}] [Client {CID}] Received global ({mb(size_recv)}) in {t_recv:.3f}s")

                try:
                    lora_sd = torch.load(global_path, map_location="cpu")
                except Exception as e:
                    print(f"[Client {CID}] ERROR loading global LoRA: {e}")
                    raise

                # ใช้ helper ที่รองรับเวอร์ชัน peft เก่า/ใหม่
                safe_set_peft_model_state_dict(model, lora_sd)
                hash_global = sha256sum(global_path)
                print(f"[{now()}] [Client {CID}] SHA256 of received global LoRA: {hash_global}")
                os.remove(global_path)

                print(f"[{now()}] [Client {CID}] Starting local training (LoRA only)...")
                t0_train = time.perf_counter()
                last_loss = local_train_lora(model)
                t1_train = time.perf_counter()
                train_time = t1_train - t0_train
                print(f"[{now()}] [Client {CID}] Training done in {train_time:.2f}s | last_loss={last_loss:.4f}")

                out_path = run_dir / f"lora_local_c{CID}_r{round_id}.pt"
                torch.save(get_peft_model_state_dict(model), out_path)
                hash_local = sha256sum(out_path)
                print(f"[{now()}] [Client {CID}] SHA256 of local LoRA to send: {hash_local}")

                if SAVE_MODEL:
                    print(f"[{now()}] [Client {CID}] Saved local LoRA: {out_path.name}")
                else:
                    print(f"[{now()}] [Client {CID}] Saved temp LoRA for upload only (SAVE_MODEL=False)")

                send_json(sock, {"cmd": "LOCAL", "round": round_id})
                t0_send = time.perf_counter()
                size_send, t_send = send_file(sock, out_path)
                t1_send = time.perf_counter()
                send_time = t1_send - t0_send
                print(f"[{now()}] [Client {CID}] Sent local LoRA ({mb(size_send)}) in {t_send:.3f}s")

                if not SAVE_MODEL and out_path.exists():
                    os.remove(out_path)
                    print(f"[{now()}] [Client {CID}] Deleted temp LoRA after sending")

                mem_mb = proc.memory_info().rss / (1024**2)
                with open(client_log, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "step", "loss", "lr", "timestamp",
                        "computation_time", "communication_time", "memory_mb",
                        "recv_time_sec", "train_time_sec", "send_time_sec"
                    ])
                    writer.writerow({
                        "step": round_id,
                        "loss": f"{last_loss:.6f}",
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
