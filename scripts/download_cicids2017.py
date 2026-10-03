"""Download the CICIDS-2017 MachineLearningCSV archive (research use).

Official source: Canadian Institute for Cybersecurity, UNB
https://www.unb.ca/cic/datasets/ids-2017.html

The UNB HTTP host often requires a click-through. This script uses the public
Hugging Face mirror of the same zip and extracts CSVs to data/raw/cicids2017/.
Cite Sharafaldin, Lashkari & Ghorbani (2018) if you use the data.
"""
from __future__ import annotations

import os
import sys
import zipfile
from urllib.request import urlretrieve

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = (
    "https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/"
    "csvs/MachineLearningCSV.zip"
)


def main() -> int:
    dest_dir = os.path.join(_PROJECT_ROOT, "data", "raw", "cicids2017")
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "MachineLearningCSV.zip")
    print("[info] downloading", MIRROR)
    urlretrieve(MIRROR, zip_path)
    print("[info] extracting to", dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    # Flatten MachineLearningCVE/ if present.
    nested = os.path.join(dest_dir, "MachineLearningCVE")
    if os.path.isdir(nested):
        for name in os.listdir(nested):
            os.replace(os.path.join(nested, name), os.path.join(dest_dir, name))
        os.rmdir(nested)
    os.remove(zip_path)
    csvs = [n for n in os.listdir(dest_dir) if n.endswith(".csv")]
    print(f"[info] {len(csvs)} CSV files ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
