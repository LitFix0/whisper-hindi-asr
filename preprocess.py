# preprocess.py — Segment loading, speaker-aware merging & dataset splitting
#
# Key design choices (new version):
#   - Uses soundfile.info() to read sample rate WITHOUT decoding the whole file
#   - Uses soundfile.read(start=, stop=) to read ONLY the required byte range
#   - Merges consecutive same-speaker segments that:
#       * have a gap < MAX_GAP_SECONDS
#       * stay under MAX_CHUNK_DURATION when combined
#   - Uses sklearn train_test_split (not HuggingFace datasets) to avoid
#     Windows multiprocessing/serialization issues
#   - Returns plain Python lists of (audio_path, start_sample, end_sample, text)
#     — no numpy arrays in RAM

import os
import json
import soundfile as sf
from sklearn.model_selection import train_test_split

from config import (
    AUDIO_DIR, TEXT_DIR,
    MAX_CHUNK_DURATION, MAX_GAP_SECONDS, MIN_CHUNK_DURATION,
)


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def _load_segments(text_path: str, audio_sr: int) -> list[dict]:
    """Parse a transcription JSON and return a list of segment dicts."""
    with open(text_path, "r", encoding="utf-8") as f:
        segs = json.load(f)
    if not isinstance(segs, list):
        return []

    out = []
    for seg in segs:
        text  = seg.get("text", "").strip()
        start = seg.get("start")
        end   = seg.get("end")
        spk   = seg.get("speaker_id")
        if not text or start is None or end is None:
            continue
        if (end - start) <= 0:
            continue
        out.append({
            "start":        start,
            "end":          end,
            "speaker_id":   spk,
            "text":         text,
            "start_sample": int(start * audio_sr),
            "end_sample":   int(end   * audio_sr),
        })
    return out


def _merge_segments(segs: list[dict], max_duration: float, max_gap: float) -> list[dict]:
    """
    Merge consecutive same-speaker segments that:
      - have a gap smaller than max_gap seconds
      - stay under max_duration seconds when combined
    """
    if not segs:
        return []

    merged = []
    buf = dict(segs[0])
    buf["text"] = [buf["text"]]

    for seg in segs[1:]:
        gap      = seg["start"] - buf["end"]
        duration = seg["end"]   - buf["start"]
        same_spk = seg["speaker_id"] == buf["speaker_id"]

        if same_spk and gap <= max_gap and duration <= max_duration:
            buf["end"]        = seg["end"]
            buf["end_sample"] = seg["end_sample"]
            buf["text"].append(seg["text"])
        else:
            merged.append({**buf, "text": " ".join(buf["text"])})
            buf = dict(seg)
            buf["text"] = [buf["text"]]

    merged.append({**buf, "text": " ".join(buf["text"])})
    return merged


# ---------------------------------------------------------------------------
# 3. Build sample list (paths + sample indices only — no audio in RAM)
# ---------------------------------------------------------------------------

def build_samples() -> list[tuple]:
    """
    Returns a list of (audio_path, start_sample, end_sample, text) tuples.
    Audio is NOT loaded here — soundfile.info() only reads the file header.
    """
    all_samples = []

    for file in sorted(os.listdir(AUDIO_DIR)):
        rec_id     = os.path.splitext(file)[0]
        audio_path = os.path.join(AUDIO_DIR, file)
        text_path  = os.path.join(TEXT_DIR,  f"{rec_id}.json")

        if not os.path.exists(text_path):
            continue

        try:
            # Read sample rate from file header only (no audio decoded)
            info = sf.info(audio_path)
            sr   = info.samplerate

            segs   = _load_segments(text_path, sr)
            chunks = _merge_segments(segs, MAX_CHUNK_DURATION, MAX_GAP_SECONDS)

            for chunk in chunks:
                duration = chunk["end"] - chunk["start"]
                if duration < MIN_CHUNK_DURATION:
                    continue
                all_samples.append((
                    audio_path,
                    chunk["start_sample"],
                    chunk["end_sample"],
                    chunk["text"],
                ))

        except Exception as e:
            print(f"Error processing {rec_id}: {e}")
            continue

    print(f"Total chunks: {len(all_samples)}")
    if all_samples:
        print(f"Sample: {all_samples[0]}")

    return all_samples


# ---------------------------------------------------------------------------
# 4. Train / Val / Test split (80 / 10 / 10)
# ---------------------------------------------------------------------------

def split_samples(all_samples: list[tuple]) -> tuple[list, list, list]:
    """Split sample list into train / val / test using sklearn."""
    train_s, test_s = train_test_split(all_samples, test_size=0.2, random_state=42)
    val_s,   test_s = train_test_split(test_s,       test_size=0.5, random_state=42)

    print(f"Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")
    return train_s, val_s, test_s


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = build_samples()
    train_s, val_s, test_s = split_samples(samples)