# `data/` — runtime binaries and Fortran assets

Large binary files (line lists, molecule tables) are tracked with
[DVC](https://dvc.org) and stored on Google Drive. They are **not** stored in
git. Git LFS is not used — the files are too large for LFS bandwidth quotas
to be practical at public scale.

## Get the data (all users)

```bash
pip install dvc dvc-gdrive
dvc pull
```

This downloads `data/lines/` (~4.5 GB) and `data/molecules/` (~2.8 GB) from
the shared Google Drive folder. No authentication is required — the folder is
publicly readable.

After `dvc pull`, the layout under `data/` matches what the code expects:

```
data/
├── lines/
│   ├── gfpred29dec2014.bin          # Kurucz GFALL predicted lines (fort.11)
│   ├── hilines.bin                  # High-excitation lines (fort.21)
│   ├── diatomicspacksrt.bin         # Diatomic molecular lines
│   ├── lowobsat12.bin               # Observed low lines (fort.111)
│   └── nltelinobsat12.bin           # NLTE lines
└── molecules/
    ├── tio/
    │   ├── schwenke.bin             # Schwenke TiO line list
    │   └── eschwenke.bin
    ├── h2o/
    │   └── h2ofastfix.bin           # Partridge-Schwenke H2O line list
    └── [*.dat / *.asc molecule catalogs]
```

## For developers with a local Kurucz tree

If you already have a Kurucz data directory on disk, you can populate `data/`
without DVC:

```bash
bash scripts/setup_data.sh --source /path/to/kurucz
```

Fortran-only files (executables, SYNTHE `tfort.*` line lists, `atlas12.for`)
are not distributed via DVC and are not needed for the Python pipeline.
