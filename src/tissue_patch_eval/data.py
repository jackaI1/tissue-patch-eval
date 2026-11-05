"""Official PCam validation release, with verified row and source identity."""

import csv
import gzip
import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

PREFIX = "camelyonpatch_level_2_split_valid"
RECORD = "https://zenodo.org/records/2546921"
# Pinned to Zenodo's own record, whose valid/test checksums differ from the GitHub table.
FILES = {
    f"{PREFIX}_x.h5.gz": "d5b63470df7cfa627aeec8b9dc0c066e",
    f"{PREFIX}_y.h5.gz": "2b85f58b927af9964a4c15b8f7e8f179",
    f"{PREFIX}_meta.csv": "67589e00a4a37ec317f2d1932c7502ca",
}


def digest(path: Path, algorithm="sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, expected: str):
    if digest(path, "md5") != expected:
        raise ValueError(f"checksum mismatch: {path.name}; remove it and fetch again")


def fetch_data(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        path = root / name
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            try:
                url = f"{RECORD}/files/{name}?download=1"
                print(f"Downloading {name}", flush=True)
                with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as f:
                    shutil.copyfileobj(response, f, length=1024 * 1024)
                verify_file(partial, md5)
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        verify_file(path, md5)
        if name.endswith(".gz"):
            # Always refresh from the checked archive rather than trust a stale HDF5.
            dest = path.with_suffix("")
            partial = dest.with_suffix(".h5.part")
            try:
                with gzip.open(path, "rb") as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                partial.replace(dest)
            finally:
                partial.unlink(missing_ok=True)


@dataclass
class PatchData:
    images: np.ndarray
    labels: np.ndarray
    slides: np.ndarray
    rows: list[dict]


def load_data(root: Path) -> PatchData:
    with (root / f"{PREFIX}_meta.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with h5py.File(root / f"{PREFIX}_y.h5", "r") as f:
        labels = np.asarray(f["y"]).reshape(-1)
    with h5py.File(root / f"{PREFIX}_x.h5", "r") as f:
        images = np.asarray(f["x"])
    if images.dtype != np.uint8 or images.shape != (len(labels), 96, 96, 3):
        raise ValueError("images must be N x 96 x 96 x 3 uint8")
    if len(rows) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("image/metadata/label alignment failure")
    for i, row in enumerate(rows):
        if row.get("") != str(i):
            raise ValueError("metadata row index is missing or out of order")
        if row.get("center_tumor_patch") not in ("True", "False"):
            raise ValueError("missing center label for alignment check")
        if int(row["center_tumor_patch"] == "True") != int(labels[i]):
            raise ValueError("metadata/label alignment failure")
        if not row.get("wsi", "").strip():
            raise ValueError("missing source slide")
    return PatchData(images, labels.astype(int), np.array([r["wsi"] for r in rows]), rows)
