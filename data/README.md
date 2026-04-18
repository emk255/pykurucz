# `data/` — runtime binaries and Fortran assets

This directory is **not** fully tracked in git. Large line-list binaries, molecule tables, optional Fortran SYNTHE line files (`synthe/linelists_full/tfort.*`, ~13 GB), and executables are listed in `.gitignore`.

## Populate `data/` (two equivalent ways)

1. **From a local Kurucz tree** (development / lab machines):

   ```bash
   bash scripts/setup_data.sh
   # or: bash scripts/setup_data.sh --source /path/to/kurucz
   ```

2. **From a published tarball** (reproducibility / external users): download the release asset built with `scripts/package_data_for_release.sh`, verify the published SHA-256, then extract **at the repository root** so you have `./data/lines/`, `./data/molecules/`, etc.

After either path, the code expects the same layout under `data/` that `setup_data.sh` would create.
