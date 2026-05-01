# `data/` — runtime binaries and line-list assets

Large binary files (line lists, molecule tables) are distributed via this
repository's **GitHub release assets**.  They are *not* stored in git.  No
authentication is required — release assets on a public repo are served
free and unmetered by GitHub.

## Get the data (all users)

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # full data (~5.2 GB)
# or, if you don't need atlas_py atmosphere iteration:
python scripts/download_data.py --synthe-only   # ~1.3 GB, enough for synthesis
# pin a specific release for reproducibility (default: latest):
python scripts/download_data.py --tag v1.0
```

Three assets are fetched (`gfpred` is split into two parts because GitHub
caps individual release assets at 2 GB; the downloader reassembles them):

| File | Size | Used by |
|---|---|---|
| `pykurucz-data-synthe-v1.0.tar.gz` | ~1.3 GB (→ ~3.3 GB extracted) | `synthe_py` |
| `gfpred29dec2014.bin.part_aa` + `.part_ab` | ~3.9 GB combined | `atlas_py` (optional) |

SHA256 is verified for every asset and for the reassembled `gfpred` file.
The synthe tarball is extracted into `data/lines/` and `data/molecules/`
and then deleted; the gfpred parts are concatenated into
`data/lines/gfpred29dec2014.bin` and the parts are removed.

### Resulting layout

```
data/
├── lines/
│   ├── gfpred29dec2014.bin          # Kurucz GFPRED predicted lines (fort.11) ~3.9 GB
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

## Updating the data

Each new release tag (`v1.0`, `v1.1`, …) carries its own asset bundle, so
old data versions remain available indefinitely for reproducibility.  By
default the downloader pulls from the `latest` release; pass `--tag <tag>`
to pin to a specific version.
