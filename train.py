# train.py — Master entry-point
#
# Usage:
#   python train.py            # fresh start
#   python train.py --resume   # resume from last checkpoint

import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

RESUME = "--resume" in sys.argv


def main():
    # ------------------------------------------------------------------
    # GPU info
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU — training on CPU (will be slow)")

    # ------------------------------------------------------------------
    # Step 1 — Load processor
    # ------------------------------------------------------------------
    print("\n[1/5] Loading processor...")
    from model_setup import load_processor, load_model, get_collator_fn, get_metrics_fn
    processor = load_processor()

    # ------------------------------------------------------------------
    # Step 2 — Load IndicVoices Hindi subset
    # ------------------------------------------------------------------
    print("\n[2/5] Loading IndicVoices Hindi dataset...")
    from data_hf import load_hf_splits
    train_data, val_data = load_hf_splits(processor)
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ------------------------------------------------------------------
    # Step 3 — Load model
    # ------------------------------------------------------------------
    print("\n[3/5] Loading model...")
    model       = load_model(processor)
    collator_fn = get_collator_fn(processor)
    metrics_fn  = get_metrics_fn(processor)

    # Reduce VRAM usage
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # Step 4 — Train
    # ------------------------------------------------------------------
    print("\n[4/5] Training...")
    from trainer_setup import get_training_args, build_trainer, train_and_save
    from config import OUTPUT_DIR

    training_args = get_training_args()
    trainer       = build_trainer(
        model, training_args, train_data, val_data,
        processor, collator_fn, metrics_fn,
    )

    if RESUME:
        checkpoints = [
            os.path.join(OUTPUT_DIR, d)
            for d in os.listdir(OUTPUT_DIR)
            if d.startswith("checkpoint-")
        ] if os.path.exists(OUTPUT_DIR) else []

        if checkpoints:
            latest = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from: {latest}")
            trainer.train(resume_from_checkpoint=latest)
        else:
            print("No checkpoint found — starting fresh")
            trainer.train()
    else:
        trainer.train()

    train_and_save(trainer, processor)

    # ------------------------------------------------------------------
    # Step 5 — Evaluate
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating on FLEURS Hindi test set...")
    from evaluate import run_evaluation
    run_evaluation()


if __name__ == "__main__":
    main()