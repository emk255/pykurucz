# Downloading Data

pykurucz ships with small physics tables and source code in the repository, but the large runtime binaries — atomic line lists, molecular catalogs, and opacity tables — are distributed via GitHub release assets. You only need to download them once.

## Why Data Is Needed

The synthesis engine relies on several multi-gigabyte binary files that are too large for Git source control:

- **Atomic line lists** (~1.3 million transitions) for computing line opacity
- **Predicted-line binaries** used by `atlas_py` for line selection during atmosphere iteration
- **Molecular line catalogs** (TiO, H₂O, CN, CO, C₂, etc.) for cool-star spectra
- **Helium broadening tables** and bound-free cross-section data

Without these files, `synthe_py` cannot load lines and `atlas_py` cannot perform line selection. The CLI will raise a clear `FileNotFoundError` pointing back to this step.

## Using the Downloader

The simplest way to obtain the data is to run the provided downloader script:

```bash
# Full data (~5.2 GB extracted) — required for Stellar Parameters, recommended for Existing Atmosphere
python scripts/download_data.py

# Synthesis-only data (~1.3 GB extracted) — sufficient if you only use Existing Atmosphere
python scripts/download_data.py --synthe-only
```

The downloader performs the following steps automatically:

1. **Fetches release assets** from the project's GitHub releases (no authentication required)
2. **Extracts tarballs** into `data/lines/` and `data/molecules/`
3. **Reassembles split binaries** — the ~3.9 GB `gfpred29dec2014.bin` file is uploaded as **3 split parts** (`part_aa`, `part_ab`, `part_ac`) because GitHub caps individual release assets at 2 GB; the downloader fetches all three and concatenates them
4. **Verifies SHA256 checksums** at every step
5. **Cleans up** intermediate archives and part files

!!! tip "Pin a specific release"
    Use `--tag v1.0` to download a specific release instead of the default `latest`. This is useful for reproducible research.

```bash
python scripts/download_data.py --tag v1.0
```

!!! tip "Force a re-download"
    The downloader skips assets whose SHA256 already matches. Pass `--force`
    to ignore the cache and re-fetch everything — useful if you suspect a
    corrupted local file or want to refresh from a freshly-published tag.

```bash
python scripts/download_data.py --force
```

## What Gets Downloaded and Where

After running the downloader, your working directory will contain:

```
data/
├── lines/
│   ├── gfallvac.latest              # Kurucz GFALL atomic line list (~1.3M lines)
│   ├── gfpred29dec2014.bin          # Predicted-line binary for atlas_py (~3.9 GB)
│   ├── continua.dat                 # Bound-free edge / cross-section table
│   ├── molecules.dat                # Dissociation energies / equilibrium constants
│   ├── he1tables.dat                # Helium broadening profiles
│   └── ...                          # Additional line-list binaries
└── molecules/
    ├── tio/schwenke.bin             # Schwenke TiO line list
    ├── h2o/h2ofastfix.bin           # Partridge–Schwenke H₂O line list
    └── ...                          # Kurucz ASCII molecular catalogs (CN, CO, C₂, etc.)
```

!!! note "Total disk usage"
    The full download extracts to approximately **5.2 GB** of data. The `--synthe-only` option reduces this to ~1.3 GB by omitting the `gfpred29dec2014.bin` predicted-line binary, which is only needed for `atlas_py` atmosphere iteration.

## Manual Download Option

If you prefer to download the files manually (e.g., for offline installation or institutional mirrors), the release assets are hosted on the project's GitHub Releases page:

1. Navigate to `https://github.com/tingyuansen/pykurucz/releases`
2. Download the latest `pykurucz-data-synthe-v*.tar.gz` asset
3. Extract it into `data/lines/` and `data/molecules/`
4. If using `atlas_py`, also download the three `gfpred29dec2014.bin.part_*` parts and concatenate them:

```bash
cat gfpred29dec2014.bin.part_aa \
    gfpred29dec2014.bin.part_ab \
    gfpred29dec2014.bin.part_ac \
    > data/lines/gfpred29dec2014.bin
```

!!! warning "Verify checksums after manual assembly"
    The SHA256 checksum for `gfpred29dec2014.bin` is verified automatically by the downloader script. If assembling manually, compare your file against the checksum listed in the release notes to ensure integrity.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: Missing line list: lines/gfallvac.latest` | Downloader not run | `python scripts/download_data.py` |
| `FileNotFoundError: Required atlas_py binary not found` | `gfpred29dec2014.bin` missing | Run downloader without `--synthe-only` |
| Slow download | Large binary size | Use `--synthe-only` if only synthesizing from existing `.atm` files |
| `urllib.error.HTTPError: 404` | Release tag misspelled | Check available tags on GitHub Releases |

## Next Steps

Once the data is downloaded, proceed to the [Quickstart](quickstart.md) or the [First Spectrum](first-spectrum.md) walkthrough.
