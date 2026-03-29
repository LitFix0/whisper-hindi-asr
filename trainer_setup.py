# trainer_setup.py — Training arguments, Seq2SeqTrainer, train & save

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from config import (
    OUTPUT_DIR, FINAL_DIR,
    PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, EVAL_STRATEGY, EVAL_STEPS, SAVE_STEPS, LOGGING_STEPS,
    GENERATION_MAX_LENGTH, PREDICT_WITH_GENERATE,
    METRIC_FOR_BEST_MODEL, GREATER_IS_BETTER, FP16, REPORT_TO,
)


def get_training_args() -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        learning_rate               = LEARNING_RATE,
        eval_strategy               = EVAL_STRATEGY,
        eval_steps                  = EVAL_STEPS,
        save_steps                  = SAVE_STEPS,
        logging_steps               = LOGGING_STEPS,
        generation_max_length       = GENERATION_MAX_LENGTH,
        predict_with_generate       = PREDICT_WITH_GENERATE,
        metric_for_best_model       = METRIC_FOR_BEST_MODEL,
        greater_is_better           = GREATER_IS_BETTER,
        fp16                        = FP16,
        report_to                   = REPORT_TO,
        dataloader_num_workers      = 0,     # 0 = main process only, no overhead on Windows
        dataloader_pin_memory       = True,  # faster CPU→GPU transfer
    )


def build_trainer(
    model,
    training_args,
    train_data,
    val_data,
    processor,
    collator_fn,
    metrics_fn,
) -> Seq2SeqTrainer:
    return Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_data,
        eval_dataset     = val_data,
        processing_class = processor,
        data_collator    = collator_fn,
        compute_metrics  = metrics_fn,
    )


def train_and_save(trainer: Seq2SeqTrainer, processor) -> None:
    trainer.train()
    trainer.save_model(FINAL_DIR)
    processor.save_pretrained(FINAL_DIR)
    print(f"Model saved to {FINAL_DIR}")