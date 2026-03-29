# inference.py — Transcribe a single WAV file using the fine-tuned model
#
# Usage:
#   python inference.py                        # transcribes test.wav
#   python inference.py path/to/audio.wav      # transcribes any wav file

import sys
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from config import FINAL_DIR, LANGUAGE, TASK

CHUNK_LENGTH  = 30        # seconds — Whisper's native context window
CHUNK_SAMPLES = CHUNK_LENGTH * 16000


def transcribe_file(audio_path: str, model_path: str = FINAL_DIR) -> str:
    """Transcribe a WAV file (any length) using the fine-tuned Whisper model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(model_path)
    model     = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    # Load full audio at 16 kHz
    audio, sr = librosa.load(audio_path, sr=16000)

    # Split into 30-second chunks
    chunks = [
        audio[i : i + CHUNK_SAMPLES]
        for i in range(0, len(audio), CHUNK_SAMPLES)
    ]

    full_transcription = []

    for chunk in chunks:
        inputs          = processor(chunk, sampling_rate=16000, return_tensors="pt")
        input_features  = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        full_transcription.append(text)

    return " ".join(full_transcription)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    print(f"Transcribing: {audio_file}")
    result = transcribe_file(audio_file)
    print("\nTranscription:")
    print(result)