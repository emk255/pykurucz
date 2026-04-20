# `data/` — runtime binaries and line-list assets

Large binary files (line lists, molecule tables) are hosted on
[HuggingFace Hub](https://huggingface.co/datasets/elliotk19/pykurucz-data).
They are **not** stored in git. No authentication is required — the dataset is
publicly accessible.

## Get the data (all users)

```bash
pip install huggingface_hub
python scripts/download_data.py
```

This downloads `data/lines/` (~4.5 GB) and `data/molecules/` (~2.8 GB).
No login, no OAuth, no Google account needed.

After the download, the layout under `data/` matches what the code expects:

```
data/
├── lines/
│   ├── gfpred29dec2014.bin          # Kurucz GFALL predicted lines (fort.11) ~3.9 GB
│   ├── hilines.bin                  # High-excitation lines (fort.21)
│   ├── diatomicspacksrt.bin         # Diatomic molecular lines
│   ├── lowobsat12.bin               # Observed low lines (fort.111)
│   ├── nltelinobsat12.bin           # NLTE lines
│   ├── continua.dat                 # Bound-free absorption edges
│   ├── molecules.dat                # Dissociation energies / equilibrium constants
│   ├── molecules.new                # Extended molecule list for ATLAS12
│   └── he1tables.dat                # Helium broadening tables
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
without downloading from HuggingFace:

```bash
bash scripts/setup_data.sh --source /path/to/kurucz
```

Fortran-only files (executables, SYNTHE `tfort.*` line lists, `atlas12.for`)
are not needed for the Python pipeline.
