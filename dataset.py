# dataset.py — PyTorch Dataset class for Whisper fine-tuning
#
# Uses soundfile.read(start=, stop=) to read ONLY the exact byte range
# needed for each segment — no full file decode, no large RAM usage.

import soundfile as sf
import librosa
from torch.utils.data import Dataset as TorchDataset


class WhisperSegmentDataset(TorchDataset):
    """
    A PyTorch Dataset that loads audio segments on-the-fly from disk.

    Each sample is a tuple: (audio_path, start_sample, end_sample, text)
    Audio is sliced by sample index using soundfile — only the required
    bytes are read, keeping RAM usage minimal even for large datasets.
    """

    def __init__(self, samples: list[tuple], processor):
        """
        Args:
            samples:   list of (audio_path, start_sample, end_sample, text)
            processor: WhisperProcessor instance
        """
        self.samples   = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        audio_path, start_sample, end_sample, text = self.samples[idx]

        # Read only the exact byte range — no full file decode
        audio, sr = sf.read(
            audio_path,
            start=start_sample,
            stop=end_sample,
            dtype="float32",
            always_2d=False,
        )

        # Resample to 16 kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Stereo → mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Audio → log-mel spectrogram
        features = self.processor.feature_extractor(
            audio, sampling_rate=16000
        )["input_features"][0]

        # Text → token ids (truncated to Whisper's 448-token limit)
        labels = self.processor.tokenizer(
            text, truncation=True, max_length=448
        ).input_ids

        return {"input_features": features, "labels": labels}