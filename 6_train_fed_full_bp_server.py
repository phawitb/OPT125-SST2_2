#!/usr/bin/env python3
# FedAvg server (sync) : GLOBAL -> wait all LOCAL -> avg/eval -> GLOBAL ...
# Logs to output/train_{id}/log_fed_bp.csv

import socket, json, torch, time, csv, shutil, psutil, os
from pathlib import Path
from bp_comm import send_json, recv_json, send_file, recv_file
import config
from transformers import AutoModelForSequenceClassification, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_from_disk

HOST        = "0.0.0.0"
SEED        = config.SEED
PORT        = config.SERVER_PORT
NUM_CLIENTS = config.NUM_CLIENTS
ROUNDS      = config.FED_FULL_BP.get("ROUNDS", 10) # config.ROUNDS
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LR_GLOBAL   = config.FED_FULL_BP.get("LEARNING_RATE", 5e-5)

ART_DIR  = Path("artifacts_custom")
META     = json.loads((ART_DIR / "meta.json").read_text())
MODEL_DIR= Path(META["model_dir"])
DATA_DIR = Path(META["dataset_dir"])

# --------- run dir / logging ----------
# base_output_dir = Path("output/fed_full_bp/server")
# base_output_dir.mkdir(parents=True, exist_ok=True)
# existing = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
# next_id = max([int(d.name.split("_")[1]) for d in existing], default=0) + 1
# run_dir = base_output_dir / f"train_{next_id}"
# run_dir.mkdir(parents=True, exist_ok=True)

base_output_dir = Path("output/fed_full_bp/server")
base_output_dir.mkdir(parents=True, exist_ok=True)
existing = [d for d in base_output_dir.iterdir() if d.name.startswith("train_")]
next_id = max([int(d.name.split("_")[1]) for d in existing], default=0) + 1
run_dir = base_output_dir / f"train_{next_id}"
run_dir.mkdir(parents=True, exist_ok=True)

# copy config.py
shutil.copy("config.py", run_dir / "config.py")

log_csv = run_dir / "log_fed_bp.csv"
with open(log_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["step","loss","lr","train_acc","val_acc","test_acc",
                    "timestamp","computation_time","communication_time","memory_mb"]
    )
    writer.writeheader()

def mb(b): return f"{b/1024/1024:.2f} MB"

# -------- Eval setup --------
ds_all  = load_from_disk(str(DATA_DIR))

train_dl= DataLoader(ds_all["train"], batch_size=64, shuffle=False,
                     collate_fn=default_data_collator)
val_dl  = DataLoader(ds_all["validation"], batch_size=64, shuffle=False,
                     collate_fn=default_data_collator)
test_dl = DataLoader(ds_all["test"], batch_size=64, shuffle=False,
                     collate_fn=default_data_collator)
train_for_test_dl = DataLoader(ds_all["train"].shuffle(seed=SEED).select(range(600)), batch_size=64, shuffle=False,
                     collate_fn=default_data_collator)

server_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)

@torch.inference_mode()
def eval_split(loader):
    server_model.eval()
    correct = total = 0
    loss_sum = 0.0
    for batch in loader:
        labels = batch.pop("labels").to(DEVICE)
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        out = server_model(**batch, labels=labels)
        logits = out.logits
        loss_sum += out.loss.item() * labels.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
    return correct/total, loss_sum/total

def avg_state_dict(states):
    keys = states[0].keys()
    return {k: sum(sd[k] for sd in states) / len(states) for k in keys}

def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((HOST, PORT)); srv.listen(NUM_CLIENTS)
    print(f"[Server] listening on {HOST}:{PORT}")

    conns = []
    for cid in range(NUM_CLIENTS):
        conn, addr = srv.accept()
        print(f"[Server] client {cid} connected from {addr}")
        conns.append((cid, conn))

    init_path = run_dir / "global_r0.pt"
    if not init_path.exists():
        base = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        torch.save(base.state_dict(), init_path)

    current_global = init_path

    proc = psutil.Process(os.getpid())

    for r in range(1, ROUNDS + 1):
        comm_t = 0.0
        comp_t = 0.0
        # 1) broadcast GLOBAL
        for cid, conn in conns:
            send_json(conn, {"cmd": "GLOBAL", "round": r})
            size, t = send_file(conn, str(current_global))
            comm_t += t
            print(f"[Server→C{cid}] global r{r} ({mb(size)}) {t:.3f}s")

        # 2) recv LOCAL
        local_paths = []
        for cid, conn in conns:
            msg = recv_json(conn)
            assert msg.get("cmd") == "LOCAL" and msg.get("round") == r
            lp = run_dir / f"local_c{cid}_r{r}.pt"
            size, t = recv_file(conn, str(lp))
            comm_t += t
            print(f"[Server←C{cid}] local r{r} ({mb(size)}) {t:.3f}s")
            local_paths.append(lp)

        # 3) avg + eval (compute time)
        t0 = time.perf_counter()
        states = []
        for lp in local_paths:
            try:
                sd = torch.load(lp, map_location="cpu", weights_only=True)
            except TypeError:
                sd = torch.load(lp, map_location="cpu")
            states.append(sd)
        global_sd = avg_state_dict(states)
        current_global = run_dir / f"global_r{r}.pt"
        torch.save(global_sd, current_global)

        server_model.load_state_dict(global_sd)
        train_acc, train_loss = eval_split(train_for_test_dl)
        val_acc,   val_loss   = eval_split(val_dl)
        test_acc,  test_loss  = eval_split(test_dl)
        comp_t = time.perf_counter() - t0
        mem_mb = proc.memory_info().rss / (1024**2)

        print(f"[Server] r{r} avg -> {current_global.name} | "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
              f"| comp={comp_t:.2f}s comm={comm_t:.2f}s mem={mem_mb:.1f}MB")

        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f,
                fieldnames=["step","loss","lr","train_acc","val_acc","test_acc",
                            "timestamp","computation_time","communication_time","memory_mb"])
            writer.writerow({
                "step": r,
                "loss": val_loss,      # global loss เลือกใช้ val_loss
                "lr": LR_GLOBAL,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "computation_time": f"{comp_t:.4f}",
                "communication_time": f"{comm_t:.4f}",
                "memory_mb": f"{mem_mb:.2f}",
            })

    for cid, conn in conns:
        send_json(conn, {"cmd": "DONE"})
        conn.close()
    srv.close()
    print("[Server] finished. Logs at", log_csv)

if __name__ == "__main__":
    main()
