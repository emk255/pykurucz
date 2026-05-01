"""Download pykurucz runtime data from this repo's GitHub release.

Usage:
    python scripts/download_data.py                 # full data (~5.2 GB)
    python scripts/download_data.py --synthe-only   # ~1.3 GB, enough for synthesis
    python scripts/download_data.py --tag v1.0      # pin to a specific release

What this script does in one step:
  1. Fetches release assets from this repo's GitHub release (no auth needed).
  2. Verifies SHA256 of every asset.
  3. Automatically EXTRACTS the synthe tarball into data/lines/ and
     data/molecules/, then deletes the tarball.
  4. Automatically REASSEMBLES the gfpred split parts into a single
     data/lines/gfpred29dec2014.bin (verified by SHA256), then deletes the parts.

Assets fetched:
  * pykurucz-data-synthe-v1.0.tar.gz  (~1.3 GB compressed, ~3.3 GB extracted)
      Atomic line lists + molecular catalogs. Required by synthe_py.
  * gfpred29dec2014.bin               (~3.9 GB; uploaded as 3 split parts
      because GitHub caps individual release assets at 2 GB).
      Required by atlas_py for full atmosphere iteration.
      Not needed for synthe_py alone (use --synthe-only).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "tingyuansen/pykurucz"
DEFAULT_TAG = "latest"

SYNTHE_FILENAME = "pykurucz-data-synthe-v1.0.tar.gz"
GFPRED_FILENAME = "gfpred29dec2014.bin"
GFPRED_PARTS = [
    "gfpred29dec2014.bin.part_aa",
    "gfpred29dec2014.bin.part_ab",
    "gfpred29dec2014.bin.part_ac",
]

# SHA256 of each release asset and the reassembled gfpred file.
ASSET_SHA256 = {
    SYNTHE_FILENAME: "fd1ac48cfe2b4a146b2b121c467a8a422b5cef85076912cac99c372fe5a1246b",
    "gfpred29dec2014.bin.part_aa": "d81dd1703e318bc6e38ead280c26ef81855647c64e45f20428457c07a002cf26",
    "gfpred29dec2014.bin.part_ab": "b84c115cc80edce178422a961b7e0b9e47da91402077dd48a34b8af25e1fb079",
    "gfpred29dec2014.bin.part_ac": "60beef0cb030cf71056a84984f640fde488d69472bdb1ba8d8e2c78de91e85f7",
}
GFPRED_FULL_SHA256 = (
    "2e9afd8ad1765d79768469216ac8f5751dcefb9b9ab49b23ea6642e289035593"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _release_asset_url(tag: str, filename: str) -> str:
    if tag == "latest":
        return f"https://github.com/{REPO}/releases/latest/download/{filename}"
    return f"https://github.com/{REPO}/releases/download/{tag}/{filename}"


def _http_download(url: str, dest: Path) -> None:
    """Stream a URL to *dest* with a textual progress meter."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 1 << 20  # 1 MiB
        downloaded = 0
        last_pct = -1
        with dest.open("wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct:
                        print(
                            f"\r  {dest.name}: {downloaded/(1<<20):,.0f} MiB / "
                            f"{total/(1<<20):,.0f} MiB ({pct}%)",
                            end="",
                            flush=True,
                        )
                        last_pct = pct
        print()


def _ensure_asset(tag: str, filename: str, dest: Path) -> Path:
    expected = ASSET_SHA256.get(filename)
    if dest.exists() and expected and _sha256(dest) == expected:
        print(f"  {filename}: already present, SHA256 OK, skipping download.")
        return dest
    _http_download(_release_asset_url(tag, filename), dest)
    if expected:
        actual = _sha256(dest)
        if actual != expected:
            raise RuntimeError(
                f"SHA256 mismatch for {dest}\n"
                f"  expected {expected}\n"
                f"  got      {actual}\n"
                f"Delete {dest} and re-run."
            )
        print(f"  {filename}: SHA256 OK.")
    return dest


def _extract_synthe(tarball: Path) -> None:
    print(f"  Extracting {tarball.name} into {REPO_ROOT} ...")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(REPO_ROOT)
    print(f"  Extract complete.  Removing tarball.")
    tarball.unlink()


def _synthe_extracted() -> bool:
    markers = [
        DATA_DIR / "lines" / "hilines.bin",
        DATA_DIR / "lines" / "lowobsat12.bin",
        DATA_DIR / "molecules" / "tio" / "schwenke.bin",
        DATA_DIR / "molecules" / "h2o" / "h2ofastfix.bin",
    ]
    return all(p.exists() for p in markers)


def _reassemble_gfpred(parts: list[Path], dest: Path) -> None:
    print(f"  Reassembling {dest.name} from {len(parts)} parts ...")
    with dest.open("wb") as out:
        for p in parts:
            with p.open("rb") as f:
                while True:
                    buf = f.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
    if GFPRED_FULL_SHA256 and not GFPRED_FULL_SHA256.startswith("__"):
        actual = _sha256(dest)
        if actual != GFPRED_FULL_SHA256:
            raise RuntimeError(
                f"SHA256 mismatch for reassembled {dest.name}\n"
                f"  expected {GFPRED_FULL_SHA256}\n"
                f"  got      {actual}\n"
                "Delete the file and re-run."
            )
        print(f"  {dest.name}: reassembled OK (SHA256 verified).")
    for p in parts:
        p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--synthe-only",
        action="store_true",
        help="Download only the synthe bundle (skip the 3.9 GB gfpred file). "
        "Enough for synthe_py synthesis from an existing .atm file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files appear to be present.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"Release tag to pull from (default: {DEFAULT_TAG}). "
        "Use a specific tag like 'v1.0' for reproducible runs.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "lines").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "molecules").mkdir(parents=True, exist_ok=True)

    print(f"Downloading pykurucz runtime data from {REPO} release {args.tag} ...")
    print(f"  Destination: {DATA_DIR}")
    print()

    # synthe bundle
    if args.force or not _synthe_extracted():
        synthe_tar = DATA_DIR / SYNTHE_FILENAME
        print(f"[1/2] {SYNTHE_FILENAME}  (~1.3 GB compressed)")
        _ensure_asset(args.tag, SYNTHE_FILENAME, synthe_tar)
        _extract_synthe(synthe_tar)
    else:
        print(f"[1/2] {SYNTHE_FILENAME}: already extracted, skipping.")
    print()

    # gfpred (optional, split into 2 parts)
    if args.synthe_only:
        print("[2/2] gfpred29dec2014.bin: skipped (--synthe-only).")
        print("      atlas_py atmosphere iteration will not be available.")
    else:
        gfpred_path = DATA_DIR / "lines" / GFPRED_FILENAME
        if gfpred_path.exists() and not args.force:
            print(f"[2/2] {GFPRED_FILENAME}: already present, skipping.")
        else:
            print(f"[2/2] {GFPRED_FILENAME}  (~3.9 GB; downloading 2 parts)")
            part_paths = [DATA_DIR / "lines" / name for name in GFPRED_PARTS]
            for name, p in zip(GFPRED_PARTS, part_paths):
                _ensure_asset(args.tag, name, p)
            _reassemble_gfpred(part_paths, gfpred_path)
    print()

    print("Done.  Data is ready:")
    print("  data/lines/                              -- atomic line lists")
    print("  data/molecules/                          -- molecular catalogs (TiO, H2O, ...)")
    if not args.synthe_only:
        print("  data/lines/gfpred29dec2014.bin           -- predicted atomic line list")


if __name__ == "__main__":
    main()
