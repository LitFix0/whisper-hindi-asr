# config.py — Shared configuration & constants

# --- Audio ---
SAMPLE_RATE = 16000

# --- Model ---
MODEL_NAME = "openai/whisper-small"
LANGUAGE   = "hi"
TASK       = "transcribe"

# --- Training output ---
OUTPUT_DIR = "./whisper-hi"
FINAL_DIR  = "./whisper-hi-final"

# --- Training hyper-params (tuned for RTX 3050 4GB) ---
PER_DEVICE_TRAIN_BATCH_SIZE = 2       # RTX 3050 can handle 4
GRADIENT_ACCUMULATION_STEPS = 2       # effective batch = 16
LEARNING_RATE               = 1e-5
EVAL_STRATEGY               = "steps"
EVAL_STEPS                  = 500
SAVE_STEPS                  = 500
LOGGING_STEPS               = 25
GENERATION_MAX_LENGTH       = 225
PREDICT_WITH_GENERATE       = True
METRIC_FOR_BEST_MODEL       = "wer"
GREATER_IS_BETTER           = False
FP16                        = True    # safe on RTX 3050 Ampere
REPORT_TO                   = "none"
GRADIENT_CHECKPOINTING      = False

# --- DataLoader ---
DATALOADER_NUM_WORKERS     = 2
DATALOADER_PIN_MEMORY      = True
DATALOADER_PREFETCH_FACTOR = 2

# --- Evaluation ---
EVAL_BATCH_SIZE = 2

# --- IndicVoices dataset ---
HF_DATASET_NAME   = "ai4bharat/IndicVoices"
HF_DATASET_CONFIG = "hindi"
HF_TRAIN_SPLIT    = "train"
HF_VAL_SPLIT      = "valid"
HF_SUBSET_SIZE    = 2000    # samples from train split
HF_VAL_SUBSET     = 200     # samples from val split

