#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build a distributable archive of pykurucz/data/ for publishing (GitHub
# Release, Zenodo, lab server, etc.). Large binaries are not stored in git;
# this tarball is the reproducible byte bundle users unpack into the repo.
#
# Usage (from repo root):
#   bash scripts/package_data_for_release.sh [VERSION_LABEL]
#
# Example:
#   bash scripts/package_data_for_release.sh 1.0.0
#
# Output:
#   pykurucz-data-<VERSION_LABEL>.tar.xz in the repo root, plus SHA-256 on stdout.
#
# Requires: a populated data/ tree (see scripts/setup_data.sh).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [[ ! -d data ]]; then
  echo "error: data/ not found. Run: bash scripts/setup_data.sh" >&2
  exit 1
fi

VERSION="${1:-$(date +%Y%m%d)}"
OUT_NAME="pykurucz-data-${VERSION}.tar.xz"

echo "Packaging data/ -> ${OUT_NAME} (this can take a long time and use tens of GB)..." >&2
tar -cJf "${OUT_NAME}" data

echo
echo "SHA-256 (record this next to your code release tag):"
shasum -a 256 "${OUT_NAME}"
echo
echo "Archive: ${REPO_ROOT}/${OUT_NAME}"
echo "Users should extract from the repo root so ./data/... matches setup_data.sh output."
