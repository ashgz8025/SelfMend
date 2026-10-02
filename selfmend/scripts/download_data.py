"""
Download the Kaggle "Dirty Cafe Sales" dataset.

Usage:
    python scripts/download_data.py

Requires KAGGLE_USERNAME and KAGGLE_KEY in .env (or ~/.kaggle/kaggle.json).

Manual fallback
---------------
If Kaggle CLI fails:
1. Go to https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training
2. Click Download → download the zip
3. Extract the CSV into  selfmend/data/raw/
4. Rename the file to  dirty_cafe_sales.csv  if it has a different name
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATASET_SLUG = "ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training"
EXPECTED_CSV = RAW_DIR / "dirty_cafe_sales.csv"


def _set_kaggle_env() -> bool:
    """Push KAGGLE_* vars from .env into the environment so the kaggle CLI picks them up."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        log.warning("KAGGLE_USERNAME / KAGGLE_KEY not found in .env")
        return False
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    return True


def download() -> Path:
    """Download dataset and return path to the CSV file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if EXPECTED_CSV.exists():
        log.info("Dataset already present at %s — skipping download.", EXPECTED_CSV)
        return EXPECTED_CSV

    if not _set_kaggle_env():
        raise EnvironmentError(
            "Kaggle credentials missing. See the manual fallback instructions at the top of this file."
        )

    try:
        import kaggle  # noqa: PLC0415  (lazy import — may not be installed)

        log.info("Downloading dataset '%s' …", DATASET_SLUG)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            DATASET_SLUG,
            path=str(RAW_DIR),
            unzip=False,
            quiet=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Kaggle download failed: {exc}") from exc

    # Find and extract the zip
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip found in {RAW_DIR} after download.")

    zip_path = zips[0]
    log.info("Extracting %s …", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()

    # Normalise filename to expected name
    csvs = [f for f in RAW_DIR.glob("*.csv") if f != EXPECTED_CSV]
    if csvs and not EXPECTED_CSV.exists():
        shutil.move(str(csvs[0]), str(EXPECTED_CSV))
        log.info("Renamed %s → %s", csvs[0].name, EXPECTED_CSV.name)

    if not EXPECTED_CSV.exists():
        raise FileNotFoundError(
            f"Download succeeded but expected CSV not found at {EXPECTED_CSV}. "
            f"Files in raw dir: {list(RAW_DIR.iterdir())}"
        )

    log.info("Dataset ready: %s  (%.1f KB)", EXPECTED_CSV, EXPECTED_CSV.stat().st_size / 1024)
    return EXPECTED_CSV


if __name__ == "__main__":
    path = download()
    print(f"\nDataset path: {path}")
