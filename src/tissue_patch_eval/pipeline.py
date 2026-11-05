"""End-to-end experiment; raw labels/source metadata never enter feature extraction."""

import csv
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .audit import verify_evidence
from .data import digest
from .evaluation import MODEL, cross_validate, summarize
from .qc import FLAGS, THRESHOLDS, extract_features, quality_flags
from .report import write_report
from .stress import damage


def run_pipeline(data, out: Path, provenance: dict, folds=5, trees=96, bootstrap=1000, seed=2026):
    out.mkdir(parents=True, exist_ok=True)
    print(f"Extracting image features and QC for {len(data.labels)} patches", flush=True)
    x = np.stack([extract_features(im) for im in data.images])
    qc_rows = [quality_flags(im) for im in data.images]
    qc = {k: np.array([r[k] for r in qc_rows], bool) for k in FLAGS}
    # Fixed random subset, chosen before fitting or looking at errors.
    idx = np.sort(np.random.default_rng(seed).choice(len(x), min(256, len(x)), replace=False))
    damaged = {k: np.stack([damage(data.images[i], k) for i in idx]) for k in FLAGS}
    stress_x = {
        k: (idx, np.stack([extract_features(im) for im in images])) for k, images in damaged.items()
    }
    print("Training random and slide-separated out-of-fold baselines", flush=True)
    cv = cross_validate(x, data.labels, data.slides, folds, seed, trees, stress_x)
    probabilities = {m: result["probability"] for m, result in cv.items()}
    print(f"Computing {bootstrap} paired slide-bootstrap draws", flush=True)
    stats = summarize(data.labels, probabilities, data.slides, qc, bootstrap, seed)
    patch_hashes = [hashlib.sha256(im.tobytes()).hexdigest() for im in data.images]
    coordinates = [(s, r["coord_x"], r["coord_y"]) for s, r in zip(data.slides, data.rows)]
    for mode in cv:
        for audit in cv[mode]["audit"]:
            test = cv[mode]["fold"] == audit["fold"]
            audit["overlap_image_hashes"] = len(
                set(np.array(patch_hashes)[test]) & set(np.array(patch_hashes)[~test])
            )
    columns = [
        "row",
        "slide",
        "coord_x",
        "coord_y",
        "label",
        "patch_sha256",
        "random_probability",
        "slide_probability",
        "random_fold",
        "slide_fold",
        *FLAGS,
        "laplacian_variance",
        "pen_fraction",
        "background_fraction",
    ]
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for i, (label, slide, row, q) in enumerate(
            zip(data.labels, data.slides, data.rows, qc_rows)
        ):
            writer.writerow(
                {
                    "row": i,
                    "slide": slide,
                    "coord_x": row["coord_x"],
                    "coord_y": row["coord_y"],
                    "label": label,
                    "patch_sha256": patch_hashes[i],
                    **{f"{m}_probability": probabilities[m][i] for m in cv},
                    **{f"{m}_fold": cv[m]["fold"][i] for m in cv},
                    **q,
                }
            )
    stress = {}
    stress_records = []
    for kind, images in damaged.items():
        p = cv["slide"]["stress"][kind]
        flags = [quality_flags(im)[kind] for im in images]
        clean_errors = (probabilities["slide"][idx] >= 0.5) != data.labels[idx]
        changed_errors = (p >= 0.5) != data.labels[idx]
        stress[kind] = {
            "n": len(idx),
            "flagged_fraction": float(np.mean(flags)),
            "clean_error_rate": float(clean_errors.mean()),
            "damaged_error_rate": float(changed_errors.mean()),
            "error_change": float(changed_errors.mean() - clean_errors.mean()),
        }
        for j, i in enumerate(idx):
            stress_records.append(
                {
                    "row": int(i),
                    "kind": kind,
                    "label": int(data.labels[i]),
                    "clean_probability": float(probabilities["slide"][i]),
                    "damaged_probability": float(p[j]),
                    "flagged": bool(flags[j]),
                }
            )
    (out / "stress_predictions.json").write_text(json.dumps(stress_records, indent=2) + "\n")
    # Show errors and correct predictions in each natural stratum without cherry-picking scores.
    examples = []
    errors = (probabilities["slide"] >= 0.5) != data.labels
    example_dir = out / "examples"
    example_dir.mkdir(exist_ok=True)
    selected = set()
    for flag, mask in [*qc.items(), ("unflagged", ~np.logical_or.reduce(list(qc.values())))]:
        for wrong in (True, False):
            candidates = np.flatnonzero(mask & (errors == wrong))
            if len(candidates):
                i = int(candidates[0])
                if i in selected:
                    continue
                selected.add(i)
                name = f"examples/patch_{i}.png"
                Image.fromarray(data.images[i]).save(out / name)
                examples.append(
                    {
                        "path": name,
                        "row": i,
                        "slide": str(data.slides[i]),
                        "label": int(data.labels[i]),
                        "probability": float(probabilities["slide"][i]),
                        "flags": [k for k in FLAGS if qc[k][i]],
                        "error": bool(errors[i]),
                    }
                )
    i = int(idx[0])
    Image.fromarray(data.images[i]).save(example_dir / "stress_clean.png")
    for kind in FLAGS:
        Image.fromarray(damaged[kind][0]).save(example_dir / f"stress_{kind}.png")
    result = {
        "dataset": {
            **provenance,
            "patches": len(x),
            "slides": len(np.unique(data.slides)),
            "positives": int(data.labels.sum()),
            "duplicate_coordinate_rows": sum(c - 1 for c in Counter(coordinates).values()),
            "duplicate_image_rows": sum(c - 1 for c in Counter(patch_hashes).values()),
        },
        "configuration": {
            "folds": folds,
            "seed": seed,
            "trees": trees,
            "model": "RandomForestClassifier",
            "model_parameters": MODEL,
            "features": x.shape[1],
            "qc_thresholds": THRESHOLDS,
        },
        "evaluation": stats,
        "fold_audit": {m: cv[m]["audit"] for m in cv},
        "stress": stress,
        "examples": examples,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "scikit-learn", "pillow", "h5py")
        },
        "artifacts": {
            "predictions.csv": digest(out / "predictions.csv"),
            "stress_predictions.json": digest(out / "stress_predictions.json"),
            **{str(p.relative_to(out)): digest(p) for p in example_dir.glob("*.png")},
        },
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, out)
    return result


def statistics_equal(a, b):
    """Allow platform-level floating-point rounding, never count/schema changes."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(statistics_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(statistics_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (float, int)):
        return bool(np.isclose(a, b, rtol=0, atol=1e-12))
    return type(a) is type(b) and a == b


def verify_readable_report(result, out):
    with tempfile.TemporaryDirectory(prefix="tissue-report-replay-") as scratch:
        root = Path(scratch)
        shutil.copytree(out / "examples", root / "examples")
        write_report(result, root)
        for name in ("report.md", "report.html"):
            if (root / name).read_bytes() != (out / name).read_bytes():
                raise ValueError(f"stale readable report: {name}; regenerate with run")


def verify_report(out: Path):
    """Replay all reported intervals/point estimates from the committed row-level evidence."""
    result = json.loads((out / "results.json").read_text())
    for name, sha in result["artifacts"].items():
        if digest(out / name) != sha:
            raise ValueError(f"artifact checksum mismatch: {name}")
    with (out / "predictions.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["label"]) for r in rows])
    slides = np.array([r["slide"] for r in rows])
    qc = {k: np.array([r[k] == "True" for r in rows]) for k in FLAGS}
    p = {k: np.array([float(r[f"{k}_probability"]) for r in rows]) for k in ("random", "slide")}
    verify_evidence(result, rows, out)
    b = result["evaluation"]["bootstrap"]
    replay = summarize(y, p, slides, qc, b["draws"], b["seed"])
    if not statistics_equal(replay, result["evaluation"]):
        raise ValueError("reported metrics or confidence intervals differ from replay")
    stress_rows = json.loads((out / "stress_predictions.json").read_text())
    for kind, expected in result["stress"].items():
        subset = [r for r in stress_rows if r["kind"] == kind]
        clean = np.mean([(r["clean_probability"] >= 0.5) != r["label"] for r in subset])
        damaged = np.mean([(r["damaged_probability"] >= 0.5) != r["label"] for r in subset])
        actual = {
            "n": len(subset),
            "flagged_fraction": float(np.mean([r["flagged"] for r in subset])),
            "clean_error_rate": float(clean),
            "damaged_error_rate": float(damaged),
            "error_change": float(damaged - clean),
        }
        if not statistics_equal(actual, expected):
            raise ValueError("stress metrics differ from replay")
    verify_readable_report(result, out)
    print(
        f"Verified {len(rows)} predictions, source isolation, metrics and all bootstrap intervals"
    )
    return result
