# data_hf.py — Load IndicVoices Hindi, pre-compute features once before training

import numpy as np
import librosa
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset

from config import (
    HF_DATASET_NAME, HF_DATASET_CONFIG,
    HF_TRAIN_SPLIT, HF_VAL_SPLIT,
    HF_SUBSET_SIZE, HF_VAL_SUBSET,
    SAMPLE_RATE,
)

AUDIO_COL    = "audio_filepath"
TEXT_COL     = "text"
MIN_DURATION = 1.0


class WhisperHFDataset(TorchDataset):
    """
    Stores pre-computed (input_features, labels) — no per-step CPU work.
    Feature extraction happens once during data loading, not every training step.
    """
    def __init__(self, features_list: list):
        self.data = features_list
        print(f"Dataset ready: {len(features_list)} samples (features pre-computed)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _collect_and_process(split_name: str, max_samples: int, processor) -> list:
    """Stream, decode audio, AND pre-compute log-mel features in one pass."""
    print(f"Streaming + processing {max_samples} samples from '{split_name}'...")

    ds = load_dataset(
        HF_DATASET_NAME,
        HF_DATASET_CONFIG,
        split=split_name,
        streaming=True,
    )

    processed = []
    errors    = 0

    for i, item in enumerate(ds):
        if len(processed) >= max_samples:
            break

        text     = str(item.get(TEXT_COL, "")).strip()
        duration = item.get("duration", 0)

        if not text or duration < MIN_DURATION:
            continue

        try:
            audio_field = item[AUDIO_COL]
            array = np.array(audio_field["array"], dtype=np.float32)
            sr    = audio_field.get("sampling_rate", 48000)

            if sr != SAMPLE_RATE:
                array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)

            if array.ndim == 2:
                array = array.mean(axis=1)

            if len(array) == 0:
                continue

            # Pre-compute log-mel spectrogram NOW (not per training step)
            input_features = processor.feature_extractor(
                array, sampling_rate=SAMPLE_RATE
            )["input_features"][0]

            # Pre-compute token labels NOW
            labels = processor.tokenizer(
                text, truncation=True, max_length=448
            ).input_ids

            processed.append({
                "input_features": input_features,
                "labels":         labels,
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Skipping sample {i}: {e}")
            continue

        if len(processed) % 500 == 0 and len(processed) > 0:
            print(f"  Processed {len(processed)}/{max_samples}...")

    print(f"  Done — {len(processed)} samples ({errors} skipped)")
    return processed


def load_hf_splits(processor):
    print(f"Loading {HF_DATASET_NAME} ({HF_DATASET_CONFIG}) via streaming...")
    print(f"Train: {HF_SUBSET_SIZE} samples | Val: {HF_VAL_SUBSET} samples")

    train_processed = _collect_and_process(HF_TRAIN_SPLIT, HF_SUBSET_SIZE, processor)
    val_processed   = _collect_and_process(HF_VAL_SPLIT,   HF_VAL_SUBSET,  processor)

    train_data = WhisperHFDataset(train_processed)
    val_data   = WhisperHFDataset(val_processed)

    return train_data, val_data


if __name__ == "__main__":
    from model_setup import load_processor
    processor  = load_processor()
    train_data, val_data = load_hf_splits(processor)
    print(f"Input shape: {train_data[0]['input_features'].shape}")
    print(f"Sample text tokens: {len(train_data[0]['labels'])}")