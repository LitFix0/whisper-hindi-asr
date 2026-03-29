# evaluate.py — Evaluation on FLEURS Hindi Test Set

import torch
import numpy as np
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer as compute_wer

from config import LANGUAGE, TASK, FINAL_DIR, EVAL_BATCH_SIZE


def load_fleurs_test():
    fleurs_test = load_dataset(
        "google/fleurs", "hi_in",
        split="test",
        trust_remote_code=True,
    )
    print(f"FLEURS Hindi test samples: {len(fleurs_test)}")
    return fleurs_test


def evaluate_model(model_name_or_path: str, fleurs_dataset, desc: str = "") -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*50}")
    print(f"Evaluating: {desc or model_name_or_path}")
    print(f"{'='*50}")

    proc = WhisperProcessor.from_pretrained(
        model_name_or_path, language=LANGUAGE, task=TASK
    )
    mdl = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path
    ).to(device)
    mdl.eval()

    # Use generation config instead of forced_decoder_ids to avoid warnings
    mdl.generation_config.forced_decoder_ids = None
    forced_ids = proc.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    all_preds  = []
    all_labels = []

    for i in range(0, len(fleurs_dataset), EVAL_BATCH_SIZE):
        batch        = fleurs_dataset[i : i + EVAL_BATCH_SIZE]
        audio_arrays = [np.array(a["array"]) for a in batch["audio"]]
        references   = batch["transcription"]

        inputs = proc.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            generated = mdl.generate(
                **inputs,
                forced_decoder_ids=forced_ids,
                max_new_tokens=224,   # reduced to avoid length errors
            )

        preds = proc.tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_preds.extend(preds)
        all_labels.extend(references)

        if (i // EVAL_BATCH_SIZE) % 10 == 0:
            done = min(i + EVAL_BATCH_SIZE, len(fleurs_dataset))
            print(f"  Processed {done}/{len(fleurs_dataset)} samples...")

    wer_score = compute_wer(all_labels, all_preds)
    print(f"\n  WER: {wer_score * 100:.2f}%")

    print("\n  Sample predictions:")
    for ref, pred in zip(all_labels[:3], all_preds[:3]):
        print(f"    REF : {ref}")
        print(f"    PRED: {pred}")
        print()

    # Free GPU memory immediately after evaluation
    del mdl
    torch.cuda.empty_cache()

    return wer_score, all_preds, all_labels


def run_evaluation():
    fleurs_test = load_fleurs_test()

    # Evaluate fine-tuned model first
    finetuned_wer, _, _ = evaluate_model(
        FINAL_DIR,
        fleurs_test,
        desc="Fine-tuned: whisper-small on Hindi IndicVoices",
    )

    # Then baseline — GPU memory freed from previous run
    baseline_wer, _, _ = evaluate_model(
        "openai/whisper-small",
        fleurs_test,
        desc="Baseline: openai/whisper-small (pretrained)",
    )

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY — FLEURS Hindi Test Set")
    print("=" * 50)
    print(f"{'Model':<40} {'WER':>8}")
    print("-" * 50)
    print(f"{'Baseline (whisper-small)':<40} {baseline_wer  * 100:>7.2f}%")
    print(f"{'Fine-tuned (whisper-small + Hindi data)':<40} {finetuned_wer * 100:>7.2f}%")
    print("-" * 50)

    improvement = baseline_wer - finetuned_wer
    if improvement > 0:
        print(f"Improvement: {improvement * 100:.2f} pp (↓ WER)")
    else:
        print(f"No improvement: {abs(improvement) * 100:.2f} pp (↑ WER)")


if __name__ == "__main__":
    run_evaluation()