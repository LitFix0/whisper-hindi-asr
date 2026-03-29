# whisper-hindi-asr

Fine-tuning OpenAI's Whisper-small model for Hindi Automatic Speech Recognition (ASR) using the IndicVoices dataset.

## Results

| Model | WER (FLEURS Hindi Test Set) |
|---|---|
| Baseline `openai/whisper-small` | 67.36% |
| Fine-tuned (this repo) | 40.47% |
| **Improvement** | **↓ 26.89 pp** |

## Dataset

[IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) — a large-scale Indian language speech dataset by AI4Bharat. We use a 1000-sample Hindi subset for training.

## Model

Base model: `openai/whisper-small` (241M parameters)  
Language: Hindi (`hi`)  
Task: Transcribe

## Project Structure

```
├── config.py          # All hyperparameters and settings
├── data_hf.py         # Load & stream IndicVoices dataset
├── model_setup.py     # Processor, model, collator, WER metric
├── trainer_setup.py   # Seq2SeqTrainer configuration
├── train.py           # Main entry point
├── evaluate.py        # WER evaluation on FLEURS Hindi test set
├── inference.py       # Transcribe any audio file
└── requirements.txt   # Dependencies
```

## Setup

**Requirements:** Python 3.12, CUDA GPU recommended

```bash
git clone https://github.com/LitFix0/whisper-hindi-asr.git
cd whisper-hindi-asr

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Training

**1. Login to HuggingFace** (required for IndicVoices dataset access):
```bash
hf auth login
```

**2. Run training:**
```bash
python train.py
```

**3. Resume from checkpoint:**
```bash
python train.py --resume
```

Training will:
- Stream 1000 Hindi samples from IndicVoices
- Fine-tune Whisper-small for ~1500 steps
- Save model to `./whisper-hi-final`

## Evaluation

Evaluate against FLEURS Hindi test set:
```bash
python evaluate.py
```

## Inference

Transcribe a Hindi audio file:
```bash
python inference.py path/to/audio.wav
```

## Hardware

Trained on:
- GPU: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
- RAM: 16GB
- Training time: ~4-5 hours

## Configuration

Key settings in `config.py`:

```python
MODEL_NAME                  = "openai/whisper-small"
HF_SUBSET_SIZE              = 1000   # training samples
PER_DEVICE_TRAIN_BATCH_SIZE = 2
LEARNING_RATE               = 1e-5
FP16                        = True
```

## License

MIT
