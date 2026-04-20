"""
Download pykurucz runtime data from HuggingFace Hub.

Usage:
    python scripts/download_data.py

Downloads data/lines/ (~0.5 GB) and data/molecules/ (~2.8 GB) from the
public HuggingFace dataset at huggingface.co/datasets/elliotk19/pykurucz-data.

No authentication required — the dataset is publicly accessible.

NOTE on gfpred29dec2014.bin (~3.9 GB, required for atlas_py atmosphere
iteration): this file is too large to distribute via HuggingFace and must
be obtained separately. If you have access to a local Kurucz data tree, run:
    bash scripts/setup_data.sh --source /path/to/kurucz
For synthe_py synthesis from an existing .atm file this file is NOT needed.
"""

import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub is not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "elliotk19/pykurucz-data"
DATA_DIR = Path(__file__).parent.parent / "data"

# Files needed for synthe_py (spectrum synthesis from .atm)
SYNTHE_FILES = [
    "lines/continua.dat",
    "lines/he1tables.dat",
    "lines/hilines.bin",
    "lines/lowobsat12.bin",
    "lines/diatomicspacksrt.bin",
    "lines/nltelinobsat12.bin",
    "lines/molecules.dat",
    "lines/molecules.new",
    "molecules/",   # all molecule catalogs + TiO + H2O
]


def main():
    print(f"Downloading pykurucz data from {REPO_ID} ...")
    print(f"  Destination: {DATA_DIR}")
    print(f"  Includes: all molecule tables (TiO, H2O, ~2.8 GB) + atomic line lists (~0.5 GB)")
    print()

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DATA_DIR,
        ignore_patterns=[
            "*.gitattributes",
            ".gitattributes",
            "README.md",
            # Partial gfpred parts (incomplete — not yet fully available)
            "*.part_aa",
            "*.part_ab",
            "*.part_ac",
            "*.partaa",
            "*.partab",
            "*.partac",
            "*.partad",
            "*.partae",
            "*.partaf",
        ],
    )

    print()
    print("Done. Data is ready:")
    print("  data/lines/     — atomic line lists for synthe_py")
    print("  data/molecules/ — TiO, H2O, and molecular catalogs")
    print()
    print("NOTE: gfpred29dec2014.bin (~3.9 GB, for atlas_py) is not included here.")
    print("      For synthe_py synthesis from an existing .atm file it is NOT needed.")
    print("      To get it, run: bash scripts/setup_data.sh --source /path/to/kurucz")


if __name__ == "__main__":
    main()
