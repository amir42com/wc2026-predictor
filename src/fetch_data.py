"""
Download historical international football match data from Kaggle.

Dataset: martj42/international-football-results-from-1872-to-2017
Files saved to: data/raw/
  - results.csv      (all match results)
  - goalscorers.csv  (individual goalscorer records)
  - shootouts.csv    (penalty shootout outcomes)

Usage:
    python src/fetch_data.py

Requires kaggle.json credentials in %USERPROFILE%\.kaggle\ (Windows)
or ~/.kaggle/ (Linux/Mac). Get yours at https://www.kaggle.com/settings.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

DATASET_SLUG = "martj42/international-football-results-from-1872-to-2017"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

EXPECTED_FILES = ["results.csv", "goalscorers.csv", "shootouts.csv"]


def _kaggle_api():
    """Return an authenticated KaggleApi instance, with a clear error if creds are missing."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit(
            "kaggle package not installed. Run: pip install kaggle"
        )

    cred_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json",
    ]
    if not any(p.exists() for p in cred_paths):
        sys.exit(
            "Kaggle credentials not found.\n"
            "1. Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "2. Save kaggle.json to: " + str(Path.home() / ".kaggle" / "kaggle.json")
        )

    api = KaggleApi()
    api.authenticate()
    return api


def download_via_kaggle_api(dest: Path) -> None:
    """Download and unzip the dataset using the official kaggle Python package."""
    api = _kaggle_api()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{DATASET_SLUG}' ...")
    api.dataset_download_files(DATASET_SLUG, path=str(dest), unzip=False)

    zip_path = dest / (DATASET_SLUG.split("/")[1] + ".zip")
    if not zip_path.exists():
        # Some kaggle versions name the zip differently
        zips = list(dest.glob("*.zip"))
        if not zips:
            sys.exit("Download succeeded but no zip file was found in data/raw/")
        zip_path = zips[0]

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    zip_path.unlink()
    print("Done. Files extracted:")
    for f in dest.iterdir():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:30s}  {size_kb:,.0f} KB")


def download_via_kaggle_cli(dest: Path) -> None:
    """Fallback: invoke the kaggle CLI directly via subprocess."""
    import subprocess

    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(dest),
        "--unzip",
    ]
    print("Falling back to kaggle CLI:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit("kaggle CLI download failed. See error above.")
    print(result.stdout)


def verify_files(dest: Path) -> bool:
    missing = [f for f in EXPECTED_FILES if not (dest / f).exists()]
    if missing:
        print(f"Warning: expected files not found: {missing}")
        return False
    print("\nAll expected files present:")
    for fname in EXPECTED_FILES:
        path = dest / fname
        print(f"  {fname:30s}  {path.stat().st_size / 1024:,.0f} KB")
    return True


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    if all((RAW_DIR / f).exists() for f in EXPECTED_FILES):
        print("Data already present in data/raw/. Delete files to re-download.")
        verify_files(RAW_DIR)
        return

    try:
        download_via_kaggle_api(RAW_DIR)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"kaggle API download failed ({exc}), trying CLI ...")
        download_via_kaggle_cli(RAW_DIR)

    if not verify_files(RAW_DIR):
        sys.exit(1)

    print("\nRaw data saved to:", RAW_DIR.resolve())


if __name__ == "__main__":
    main()
