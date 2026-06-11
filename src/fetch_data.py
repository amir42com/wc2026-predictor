
"""
Download historical international football match data from GitHub.

Source: https://github.com/martj42/international_results
Files saved to: data/raw/
  - results.csv      (all match results)
  - goalscorers.csv  (individual goalscorer records)
  - shootouts.csv    (penalty shootout outcomes)

Usage:
    python src/fetch_data.py
"""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master/"
FILES = ["results.csv", "goalscorers.csv", "shootouts.csv"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_file(url: str, dest: Path) -> None:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        dest = RAW_DIR / fname
        if dest.exists():
            print(f"  {fname} already present, skipping.")
            continue
        url = BASE_URL + fname
        print(f"Downloading {url}")
        try:
            download_file(url, dest)
        except requests.HTTPError as e:
            sys.exit(f"Failed to download {fname}: {e}")

    print("\nData summary:")
    try:
        import pandas as pd
        for fname in FILES:
            df = pd.read_csv(RAW_DIR / fname)
            print(f"  {fname:20s}  {len(df):>7,} rows  x  {df.shape[1]} cols")
    except ImportError:
        for fname in FILES:
            size_kb = (RAW_DIR / fname).stat().st_size / 1024
            print(f"  {fname:20s}  {size_kb:,.0f} KB")

    print(f"\nRaw data saved to: {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()
