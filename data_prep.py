# data_prep.py — Load metadata sheet & download audio + transcription files

import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import METADATA_URL, AUDIO_DIR, TEXT_DIR, MAX_WORKERS


# ---------------------------------------------------------------------------
# 1. Load metadata sheet
# ---------------------------------------------------------------------------

def load_metadata(url: str = METADATA_URL) -> pd.DataFrame:
    df = pd.read_excel(url, engine="openpyxl")

    # Rebuild GCS public URLs from the last two path segments
    url_cols = ["rec_url_gcp", "transcription_url_gcp", "metadata_url_gcp"]
    for col in url_cols:
        df[col] = (
            "https://storage.googleapis.com/upload_goai/"
            + df[col].str.extract(r"([^/]+/[^/]+)$")[0]
        )

    print(f"Total recordings in sheet: {len(df)}")
    print("Sample rec URL:", df["rec_url_gcp"].iloc[0])
    print("Sample transcription URL:", df["transcription_url_gcp"].iloc[0])
    return df


# ---------------------------------------------------------------------------
# 2. Download audio + transcription files
# ---------------------------------------------------------------------------

def _download_file(url: str, path: str) -> str:
    if os.path.exists(path):
        return "skipped"
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def _process_row(row) -> tuple:
    rec_id     = str(row["recording_id"])
    audio_path = os.path.join(AUDIO_DIR, f"{rec_id}.wav")
    text_path  = os.path.join(TEXT_DIR,  f"{rec_id}.json")
    a_status   = _download_file(row["rec_url_gcp"],           audio_path)
    t_status   = _download_file(row["transcription_url_gcp"], text_path)
    return rec_id, a_status, t_status


def download_files(df: pd.DataFrame) -> None:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR,  exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_process_row, row) for _, row in df.iterrows()]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(result)

    ok      = sum(1 for _, a, t in results if a == "ok"      and t == "ok")
    skipped = sum(1 for _, a, t in results if a == "skipped" or  t == "skipped")
    errors  = sum(1 for _, a, t in results if "error" in a   or  "error" in t)
    print(f"Downloaded: {ok} | Skipped (cached): {skipped} | Errors: {errors}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_metadata()
    download_files(df)