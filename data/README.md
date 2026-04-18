# `data/` — runtime binaries and Fortran assets

This directory is **not** fully tracked in git. Large line-list binaries, molecule tables, optional Fortran SYNTHE line files (`synthe/linelists_full/tfort.*`, ~13 GB), and Fortran executables are excluded via `.gitignore`. Git LFS is intentionally not used — the files are too large for LFS bandwidth quotas to be practical at public scale.

## How to populate `data/`

### Option A — Download the release bundle (external users, recommended)

A versioned `pykurucz-data-*.tar.xz` archive is distributed with each GitHub Release. Download it, verify the SHA-256 printed on the release page, then extract **at the repository root**:

```bash
# Replace URL and filename with the one from the release page
curl -L -O https://github.com/emk255/pykurucz/releases/download/vX.Y.Z/pykurucz-data-X.Y.Z.tar.xz
shasum -a 256 pykurucz-data-X.Y.Z.tar.xz   # compare against release notes
tar -xJf pykurucz-data-X.Y.Z.tar.xz        # extracts ./data/... at repo root
```

> **Note:** The first public data bundle has not been released yet. Check the
> [Releases page](https://github.com/emk255/pykurucz/releases) for availability.
> If no bundle is posted, use Option B or contact the authors.

### Option B — Copy from a local Kurucz data tree (developers / lab machines)

If you already have a Kurucz data directory on disk (e.g. from
[tingyuansen/kurucz](https://github.com/tingyuansen/kurucz)):

```bash
bash scripts/setup_data.sh --source /path/to/kurucz
# omit --source if ../kurucz sits next to this repo (default)
```

Use `--no-synthe` to skip the ~13 GB Fortran SYNTHE line lists if you only need the Python synthesis path.

After either option, the code expects the same layout:
`data/lines/`, `data/molecules/`, `data/bin_macos/` or `data/bin_linux/`, `data/synthe/` (optional).
