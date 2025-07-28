SEED = 42
# BATCH_SIZE = 32        # ค่า default สำหรับวิธีอื่น

# --- Networking / Fed settings ---
SERVER_IP   = "192.168.1.45"   # เปลี่ยนให้ตรงกับเครื่อง Server
SERVER_PORT = 55552
NUM_CLIENTS = 2
# ROUNDS      = 5                 # จำนวนรอบ FedAvg

# 2_train_full_bp.py
FULL_BP = {
    "MAX_STEPS":    100,
    "EVAL_EVERY":   10,
    "LOG_EVERY":    2,
    "SAVE_EVERY":   50,
    "GRAD_ACCUM":   1,
    "WARMUP_STEPS": 10,
    "LEARNING_RATE": 5e-5,
    "WEIGHT_DECAY": 0.0,
    "BATCH_SIZE":   4,
    "SEED":         42,
}

# 3_train_full_zo.py
FULL_ZO = {
    "MAX_STEPS":     10000,
    # "LOCAL_STEPS":   1000,       # ZO steps per round
    "EVAL_EVERY":    100,
    "LOG_EVERY":     10,
    "SAVE_EVERY":    10000,
    "GRAD_ACCUM":    1,
    "MU":            1e-4,
    "P":             10,
    "LEARNING_RATE": 5e-5,
    "WEIGHT_DECAY":  0.0,
    "BATCH_SIZE":    8,
    "SEED":          42,
}

# 4_train_lora_bp.py
LORA_BP = {
    "MAX_STEPS":    1000,
    "EVAL_EVERY":   100,
    "LOG_EVERY":    4,
    "SAVE_EVERY":   1000,
    "GRAD_ACCUM":   1,
    "WARMUP_STEPS": 10,
    "LEARNING_RATE": 4e-5, #5e-5,
    "WEIGHT_DECAY": 0.0,
    "BATCH_SIZE":   4,
}

# 5_train_lora_zo.py
LORA_ZO = {
    "BATCH_SIZE":    8,
    "LEARNING_RATE": 3e-5, #5e-5,
    "WEIGHT_DECAY":  1e-2, #0.0,
    "MAX_STEPS":     10000,
    "EVAL_EVERY":    100,
    "LOG_EVERY":     10,
    "SAVE_EVERY":    10000,
    "GRAD_ACCUM":    1,
    "MU":            1e-4,
    "P":             10,

    "LORA_R":        16,
    "LORA_ALPHA":    32,
    "LORA_DROPOUT":  0.1, 
}

# ---------------------------------------------------------------
# 6_train_fed_full_bp_xxxxx.py
# FED_FULL_BP 
FED_FULL_BP = {

    "ROUNDS":       10,        # จำนวนรอบ FedAvg
    "LOCAL_STEPS":  10,       # จำนวน step ที่ client train ต่อรอบ
    "LOG_EVERY":    2,        # log ทุกกี่ step
    "BATCH_SIZE":   4,        # batch size ต่อรอบ
    "LEARNING_RATE": 5e-5,    # learning rate ต่อรอบ
      
}

# FED_FULL_BP = {

#     "ROUNDS":       10,        # จำนวนรอบ FedAvg
#     "LOCAL_STEPS":  50,       # จำนวน step ที่ client train ต่อรอบ
#     "LOG_EVERY":    5,        # log ทุกกี่ step
#     "BATCH_SIZE":   4,        # batch size ต่อรอบ
#     "LEARNING_RATE": 5e-5,    # learning rate ต่อรอบ
      
# }

# 7_train_fed_full_zo_xxxxxx.py
# FED_FULL_ZO = {


# 8_train_fed_lora_bp_xxxxxx.py
# FED_LORA_BP = {


# 9_train_fed_lora_zo_xxxxxx.py
# FED_LORA_ZO = {















BACKPROP_LORA = {

    "ROUNDS": 20,
    "LEARNING_RATE": 5e-5,
    
    "LOCAL_STEPS":  50,   # จำนวน step ที่ client train ต่อรอบ
    "LOG_EVERY":    2,
    "BATCH_SIZE":   4,
    "SEED":         42,

    # "MAX_STEPS":    400,
    # "EVAL_EVERY":   10,
    # "SAVE_EVERY":   5,
    # "GRAD_ACCUM":   1,
    # "WARMUP_STEPS": 10,
    # "WEIGHT_DECAY": 0.0,
    
}

# ------- Zeroth-Order (ZO-SGD) -------

# ZO = {
#     "MAX_STEPS":     10000,
#     "EVAL_EVERY":    100,
#     "LOG_EVERY":     20,
#     "SAVE_EVERY":    100,
#     "GRAD_ACCUM":    4,        # รวม 4 batches ก่อนอัปเดต
#     "MU":            1e-3,
#     "P":             5,       # หรือ 5 แต่ใช้ orthogonal
#     "LEARNING_RATE": 5e-6,
#     "WEIGHT_DECAY":  0.0,
#     "BATCH_SIZE":    8,
#     "SEED":          42,
# }

ZO_FULL = {
    "MAX_STEPS":     10000,
    # "LOCAL_STEPS":   1000,       # ZO steps per round
    "EVAL_EVERY":    100,
    "LOG_EVERY":     10,
    "SAVE_EVERY":    10000,
    "GRAD_ACCUM":    1,
    "MU":            1e-4,
    "P":             10,
    "LEARNING_RATE": 5e-5,
    "WEIGHT_DECAY":  0.0,
    "BATCH_SIZE":    8,
    "SEED":          42,
}

ZO_LORA = {
    "ROUNDS":       20,
    "LOCAL_STEPS":  1000,        # doubled
    "LOG_EVERY":    2,
    "MU":            1e-3,     # halved
    "P":            5,        # x4 more directions
    "LEARNING_RATE": 1e-4,     # halve to start
    "WEIGHT_DECAY":  0.0,     # small regularization
    "BATCH_SIZE":    4,        # twice as big
    "SEED":          42,
}


# ZO_LORA = {
#     "ROUNDS":      20,
#     # "MAX_STEPS":     100,
#     "LOCAL_STEPS":   10,       # ZO steps per round
#     "EVAL_EVERY":    5,
#     "LOG_EVERY":     1,
#     "SAVE_EVERY":    10,
#     "GRAD_ACCUM":    1,
#     "MU":            1e-3,
#     "P":             5,
#     "LEARNING_RATE": 1e-4,
#     "WEIGHT_DECAY":  0.0,
#     "BATCH_SIZE":    4,
#     "SEED":          42,
# }


