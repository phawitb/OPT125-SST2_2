#!/usr/bin/env python3
# 9_train_fed_lora_zo_server.py — FedAvg Server (ZO-LoRA)
# ส่ง LoRA adapters → รับ LoRA adapters (ที่เทรนแบบ Zero-Order ที่ client) → เฉลี่ย → ประเมิน

import socket, json, torch, time, csv, shutil, psutil, os, sys, threading, hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from bp_comm import send_json, recv_json, send_file, recv_file
import config

from transformers import AutoModelForSequenceClassification, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_from_disk

from peft import (
    LoraConfig, get_peft_model, TaskType,
    get_peft_model_state_dict, set_peft_model_state_dict,
)

# -------------------- CONFIG --------------------
USE_CPU     = "--use_cpu" in sys.argv
HOST        = "0.0.0.0"
PORT        = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS

FL = getattr(config, "FED_ZO_LORA", getattr(config, "BACKPROP_LORA", {}))
ROUNDS      = FL.get("ROUNDS", 20)
EVAL_BS     = FL.get("EVAL_BATCH_SIZE", 64)
SAVE_MODEL  = FL.get("SAVE_MODEL", False)
SEED        = getattr(config, "SEED", 42)

LORA_R       = FL.get("LORA_R", 16)
LORA_ALPHA   = FL.get("LORA_ALPHA", 32)
LORA_DROPOUT = FL.get("LORA_DROPOUT", 0.1)

DEVICE = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- ARTIFACTS --------------------
ART_DIR  = Path("artifacts_custom")
META     = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR= Path(META["model_dir"])
DATA_DIR = Path(META["dataset_dir"])

# -------------------- OUTPUT --------------------
base_output_dir = Path("output/fed_lora_zo/server")
base_output_dir.mkdir(parents=True, exist_ok=True)
runs = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in runs], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)
if Path("config.py").exists():
    shutil.copy("config.py", run_dir / "config.py")

log_csv = run_dir / "log_fed_lora_zo.csv"
fieldnames = [
    "round","train_acc","val_acc","test_acc","timestamp","mem_mb","avg_eval_time_sec"
]
for cid in range(NUM_CLIENTS):
    fieldnames += [f"sent_MB_C{cid}", f"send_time_C{cid}", f"recv_MB_C{cid}", f"recv_time_C{cid}"]
with open(log_csv, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()

# -------------------- HELPERS --------------------
def mb(b): return f"{b/1024/1024:.2f}"
def now(): return datetime.now().strftime("%H:%M:%S")
def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for ch in iter(lambda: fp.read(1<<20), b""):
            h.update(ch)
    return h.hexdigest()

@torch.inference_mode()
def eval_acc(loader, model):
    model.eval()
    correct = total = 0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        correct += (logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
    return correct/total

# เฉลี่ย state dict ของ LoRA (เอาเฉพาะ key ที่มีร่วมกัน)
def avg_lora_states(states: List[Dict[str, torch.Tensor]]):
    keyset = set(states[0].keys())
    for s in states[1:]:
        keyset &= set(s.keys())
    if not keyset:
        keyset = set(states[0].keys())
    avg_sd = {}
    for k in sorted(keyset):
        avg_sd[k] = torch.stack([s[k].float() for s in states if k in s], dim=0).mean(dim=0)
    return avg_sd

# ตัวช่วยรองรับ peft เก่า/ใหม่ (บางเวอร์ชันไม่มี arg strict)
def safe_set_peft_model_state_dict(model, peft_sd):
    try:
        set_peft_model_state_dict(model, peft_sd, strict=False)
        return
    except TypeError:
        pass
    current = get_peft_model_state_dict(model)
    filtered = {k: v for k, v in peft_sd.items() if k in current}
    set_peft_model_state_dict(model, filtered)

# -------------------- DATA & MODEL --------------------
torch.manual_seed(SEED)
ds_all = load_from_disk(str(DATA_DIR))
eval_train = DataLoader(ds_all["train"].shuffle(seed=SEED).select(range(600)), batch_size=EVAL_BS,
                        shuffle=False, collate_fn=default_data_collator)
eval_val   = DataLoader(ds_all["validation"], batch_size=EVAL_BS, shuffle=False,
                        collate_fn=default_data_collator)
eval_test  = DataLoader(ds_all["test"], batch_size=EVAL_BS, shuffle=False,
                        collate_fn=default_data_collator)

base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False,
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj","k_proj","v_proj","out_proj"],  # OPT family
)
server_model = get_peft_model(base_model, lora_cfg).to(DEVICE)

# เตรียม global LoRA (r0)
gpath = run_dir / "lora_global_r0.pt"
if not gpath.exists():
    torch.save(get_peft_model_state_dict(server_model), gpath)

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

    # ✅ กำหนด path เริ่มต้นภายใน main ก่อนใช้งาน
    current_global_path = str(gpath)

    for r in range(1, ROUNDS+1):
        print(f"\n[{now()}] [Server] === ROUND {r} START ===")
        round_stats = {}

        # ---- SEND GLOBAL ----
        path_to_send = current_global_path
        def send_task(cid, conn, path=path_to_send):
            try:
                send_json(conn, {"cmd":"GLOBAL","round":r})
                size, t = send_file(conn, path)
                print(f"[{now()}] [Server→C{cid}] Sent {Path(path).name} ({mb(size)} MB) in {t:.3f}s | sha={sha256sum(path)[:10]}..")
                round_stats[f"sent_MB_C{cid}"]   = mb(size)
                round_stats[f"send_time_C{cid}"] = f"{t:.4f}"
            except Exception as e:
                print(f"[{now()}] [Server→C{cid}] SEND ERROR: {e}")
        ths = [threading.Thread(target=send_task, args=(cid,conn)) for cid,conn in conns]
        [t.start() for t in ths]; [t.join() for t in ths]

        # ---- RECV LOCAL ----
        local_paths = [None]*NUM_CLIENTS
        def recv_task(cid, conn):
            try:
                msg = recv_json(conn)
                assert msg["cmd"]=="LOCAL" and msg["round"]==r
                path = run_dir / f"lora_local_c{cid}_r{r}.pt"
                size, t = recv_file(conn, str(path))
                print(f"[{now()}] [Server←C{cid}] Recv {Path(path).name} ({mb(size)} MB) in {t:.3f}s | sha={sha256sum(path)[:10]}..")
                round_stats[f"recv_MB_C{cid}"]   = mb(size)
                round_stats[f"recv_time_C{cid}"] = f"{t:.4f}"
                local_paths[cid] = path
            except Exception as e:
                print(f"[{now()}] [Server←C{cid}] RECV ERROR: {e}")
        ths = [threading.Thread(target=recv_task, args=(cid,conn)) for cid,conn in conns]
        [t.start() for t in ths]; [t.join() for t in ths]

        valid = [p for p in local_paths if p is not None]
        if not valid:
            print(f"[{now()}] [Server] No valid local adapters; stop.")
            break

        # ---- AVERAGE & APPLY ----
        states = [torch.load(p, map_location="cpu") for p in valid]
        global_lora = avg_lora_states(states)
        safe_set_peft_model_state_dict(server_model, global_lora)

        # ---- SAVE NEW GLOBAL & UPDATE PATH ----
        new_gpath = run_dir / f"lora_global_r{r}.pt"
        torch.save(global_lora, new_gpath)
        if not SAVE_MODEL:
            for p in valid:
                try: os.remove(p)
                except: pass
        current_global_path = str(new_gpath)  # ✅ อัปเดตสำหรับรอบถัดไป

        # ---- EVAL ----
        t0 = time.perf_counter()
        tr = eval_acc(eval_train, server_model)
        va = eval_acc(eval_val,   server_model)
        te = eval_acc(eval_test,  server_model)
        et = time.perf_counter() - t0
        mem = proc.memory_info().rss/1024/1024

        row = {
            "round": r, "train_acc": f"{tr:.4f}", "val_acc": f"{va:.4f}", "test_acc": f"{te:.4f}",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "mem_mb": f"{mem:.2f}", "avg_eval_time_sec": f"{et:.4f}",
        }
        for cid in range(NUM_CLIENTS):
            row[f"sent_MB_C{cid}"]   = round_stats.get(f"sent_MB_C{cid}","")
            row[f"send_time_C{cid}"] = round_stats.get(f"send_time_C{cid}","")
            row[f"recv_MB_C{cid}"]   = round_stats.get(f"recv_MB_C{cid}","")
            row[f"recv_time_C{cid}"] = round_stats.get(f"recv_time_C{cid}","")

        with open(log_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(f"[{now()}] [Server] ROUND {r} | train={tr:.4f} val={va:.4f} test={te:.4f}")

    # ---- CLOSE ----
    for _,conn in conns:
        try: send_json(conn, {"cmd":"DONE"})
        except: pass
        conn.close()
    srv.close()

    print(f"[{now()}] [Server] Finished. Logs -> {log_csv}")
    if SAVE_MODEL:
        final_dir = run_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        server_model.save_pretrained(final_dir, safe_serialization=True)
        print(f"[{now()}] [Server] Saved final model to {final_dir}")

if __name__ == "__main__":
    main()
