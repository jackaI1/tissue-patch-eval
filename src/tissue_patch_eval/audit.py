"""Reconstruct published source/fold/example claims from row-level evidence."""

import hashlib
import json
from collections import Counter

import numpy as np
from PIL import Image

from .evaluation import make_folds
from .qc import FLAGS, THRESHOLDS
from .stress import damage


def verify_evidence(result, rows, out):
    n = len(rows)
    if [int(r["row"]) for r in rows] != list(range(n)):
        raise ValueError("prediction rows must be unique and ordered")
    y = np.array([int(r["label"]) for r in rows])
    slides = np.array([r["slide"] for r in rows])
    hashes = np.array([r["patch_sha256"] for r in rows])
    coords = [(r["slide"], r["coord_x"], r["coord_y"]) for r in rows]
    counts = {
        "patches": n,
        "slides": len(set(slides)),
        "positives": int(y.sum()),
        "duplicate_coordinate_rows": sum(c - 1 for c in Counter(coords).values()),
        "duplicate_image_rows": sum(c - 1 for c in Counter(hashes).values()),
    }
    if any(result["dataset"][k] != v for k, v in counts.items()):
        raise ValueError("dataset counts differ from prediction evidence")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("non-binary prediction labels")
    for row in rows:
        for flag, measurement, threshold, direction in (
            ("blur", "laplacian_variance", "laplacian_variance_below", "below"),
            ("pen", "pen_fraction", "pen_fraction_above", "above"),
            ("background", "background_fraction", "background_fraction_above", "above"),
        ):
            v = float(row[measurement])
            intended = (
                v < THRESHOLDS[threshold] if direction == "below" else v > THRESHOLDS[threshold]
            )
            if not np.isfinite(v) or row[flag] != str(intended):
                raise ValueError("QC flag differs from saved measurement")
    config = result["configuration"]
    if config["qc_thresholds"] != THRESHOLDS:
        raise ValueError("saved QC thresholds differ from implementation")
    if set(result["fold_audit"]) != {"random", "slide"}:
        raise ValueError("missing protocol audit")
    for mode in ("random", "slide"):
        probabilities = np.array([float(r[f"{mode}_probability"]) for r in rows])
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("invalid saved probability")
        folds = make_folds(y, slides, mode, config["folds"], config["seed"])
        actual_folds = np.array([int(r[f"{mode}_fold"]) for r in rows])
        expected_folds = np.full(n, -1)
        expected_audit = []
        for fold, (train, test) in enumerate(folds):
            expected_folds[test] = fold
            expected_audit.append(
                {
                    "fold": fold,
                    "train_n": len(train),
                    "test_n": len(test),
                    "train_slides": len(set(slides[train])),
                    "test_slides": len(set(slides[test])),
                    "test_positive_fraction": float(y[test].mean()),
                    "train_positive_fraction": float(y[train].mean()),
                    "overlap_slides": len(set(slides[train]) & set(slides[test])),
                    "overlap_image_hashes": len(set(hashes[train]) & set(hashes[test])),
                }
            )
        if not np.array_equal(actual_folds, expected_folds):
            raise ValueError("fold membership differs from configured split")
        if expected_audit != result["fold_audit"][mode]:
            raise ValueError("fold audit differs from prediction evidence")
    for example in result["examples"]:
        i = example["row"]
        if not 0 <= i < n or example["path"] != f"examples/patch_{i}.png":
            raise ValueError("invalid example row or path")
        row = rows[i]
        expected = {
            "path": example["path"],
            "row": i,
            "slide": row["slide"],
            "label": int(row["label"]),
            "probability": float(row["slide_probability"]),
            "flags": [k for k in FLAGS if row[k] == "True"],
            "error": bool((float(row["slide_probability"]) >= 0.5) != int(row["label"])),
        }
        if example != expected:
            raise ValueError("example caption differs from prediction evidence")
        image = np.asarray(Image.open(out / example["path"]).convert("RGB"))
        if hashlib.sha256(image.tobytes()).hexdigest() != row["patch_sha256"]:
            raise ValueError("example image differs from patch fingerprint")
    stress_rows = json.loads((out / "stress_predictions.json").read_text())
    idx = sorted(
        np.random.default_rng(config["seed"]).choice(n, min(256, n), replace=False).tolist()
    )
    if len(stress_rows) != len(idx) * len(FLAGS) or set(result["stress"]) != set(FLAGS):
        raise ValueError("incomplete controlled-damage evidence")
    for kind in FLAGS:
        subset = [r for r in stress_rows if r["kind"] == kind]
        if [r["row"] for r in subset] != idx:
            raise ValueError("stress selection differs from seeded sample")
        for record in subset:
            row = rows[record["row"]]
            if (
                record["label"] != int(row["label"])
                or record["clean_probability"] != float(row["slide_probability"])
                or not 0 <= record["damaged_probability"] <= 1
            ):
                raise ValueError("stress result is not aligned with clean prediction")
    clean = np.asarray(Image.open(out / "examples/stress_clean.png").convert("RGB"))
    if hashlib.sha256(clean.tobytes()).hexdigest() != rows[idx[0]]["patch_sha256"]:
        raise ValueError("stress clean example differs from patch fingerprint")
    for kind in FLAGS:
        saved = np.asarray(Image.open(out / f"examples/stress_{kind}.png").convert("RGB"))
        if not np.array_equal(saved, damage(clean, kind)):
            raise ValueError("stress example differs from specified transformation")
