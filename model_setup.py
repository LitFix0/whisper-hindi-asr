# model_setup.py — Processor, model loading, data collator & WER metric

import torch
from jiwer import wer as _jiwer_wer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from config import MODEL_NAME, LANGUAGE, TASK


# ---------------------------------------------------------------------------
# 5. Load Processor
# ---------------------------------------------------------------------------

def load_processor() -> WhisperProcessor:
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    return processor


# ---------------------------------------------------------------------------
# 6. Load Model & configure for Hindi
# ---------------------------------------------------------------------------

def load_model(processor: WhisperProcessor) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.generation_config.suppress_tokens = []

    print(
        "Model loaded. Parameters:",
        sum(p.numel() for p in model.parameters()) // 1_000_000, "M"
    )
    return model


# ---------------------------------------------------------------------------
# 7. Data Collator
# ---------------------------------------------------------------------------

def data_collator(features: list, processor: WhisperProcessor) -> dict:
    input_features = [f["input_features"] for f in features]
    labels         = [f["labels"]         for f in features]

    # Pad log-mel features to the same length
    batch = processor.feature_extractor.pad(
        {"input_features": input_features},
        return_tensors="pt",
    )

    # Pad token labels; replace pad id with -100 so loss ignores padding
    labels_batch  = processor.tokenizer.pad(
        {"input_ids": labels}, return_tensors="pt"
    )
    labels_padded = labels_batch["input_ids"].masked_fill(
        labels_batch["attention_mask"] != 1, -100
    )

    batch["labels"] = labels_padded
    return batch


def get_collator_fn(processor: WhisperProcessor):
    def _collate(features):
        return data_collator(features, processor)
    return _collate


# ---------------------------------------------------------------------------
# 8. WER Metric
# ---------------------------------------------------------------------------

# Load once at module level (avoids repeated HF Hub requests)



def compute_metrics(pred, processor: WhisperProcessor) -> dict:
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id before decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * _jiwer_wer(label_str, pred_str)
    return {"wer": wer}


def get_metrics_fn(processor: WhisperProcessor):
    def _metrics(pred):
        return compute_metrics(pred, processor)
    return _metrics


# ---------------------------------------------------------------------------
# Entry-point (smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    processor = load_processor()
    model     = load_model(processor)
    print("Processor and model loaded successfully.")