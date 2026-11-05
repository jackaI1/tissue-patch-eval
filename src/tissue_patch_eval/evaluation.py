"""Matched out-of-fold predictions and paired, slide-cluster bootstrap intervals."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

METRICS = ("accuracy", "balanced_accuracy", "roc_auc")
MODEL = {"max_depth": 16, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": 2}


def make_folds(y, slides, mode, n_splits=5, seed=2026):
    if len(y) != len(slides) or set(np.unique(y)) != {0, 1}:
        raise ValueError("aligned binary labels and source slides are required")
    if mode not in ("random", "slide"):
        raise ValueError("mode must be random or slide")
    if n_splits < 2 or np.bincount(y).min() < n_splits:
        raise ValueError("too few examples per class for folds")
    if mode == "slide" and len(np.unique(slides)) < n_splits:
        raise ValueError("too few source slides for folds")
    splitter = (
        StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        if mode == "slide"
        else StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    )
    folds = list(
        splitter.split(np.zeros(len(y)), y, slides)
        if mode == "slide"
        else splitter.split(np.zeros(len(y)), y)
    )
    for train, test in folds:
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("a fold lacks a class; use fewer folds or more source slides")
        if mode == "slide" and set(slides[train]) & set(slides[test]):
            raise ValueError("source slide leakage")
    return folds


def cross_validate(x, y, slides, n_splits=5, seed=2026, trees=96, stress_features=None):
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or trees < 1:
        raise ValueError("finite features must align with labels; trees must be positive")
    result = {}
    for mode in ("random", "slide"):
        probability = np.full(len(y), np.nan)
        fold_id = np.full(len(y), -1)
        audit = []
        stress = {k: np.full(len(v[0]), np.nan) for k, v in (stress_features or {}).items()}
        for fold, (train, test) in enumerate(make_folds(y, slides, mode, n_splits, seed)):
            model = RandomForestClassifier(n_estimators=trees, random_state=seed + fold, **MODEL)
            model.fit(x[train], y[train])
            probability[test] = model.predict_proba(x[test])[:, 1]
            fold_id[test] = fold
            if mode == "slide":
                for key, (indices, damaged_x) in (stress_features or {}).items():
                    selected = np.isin(indices, test)
                    if selected.any():
                        stress[key][selected] = model.predict_proba(damaged_x[selected])[:, 1]
            audit.append(
                {
                    "fold": fold,
                    "train_n": len(train),
                    "test_n": len(test),
                    "train_slides": len(np.unique(slides[train])),
                    "test_slides": len(np.unique(slides[test])),
                    "test_positive_fraction": float(y[test].mean()),
                    "train_positive_fraction": float(y[train].mean()),
                    "overlap_slides": len(set(slides[train]) & set(slides[test])),
                }
            )
        if not np.isfinite(probability).all() or (fold_id < 0).any():
            raise ValueError("incomplete out-of-fold coverage")
        result[mode] = {
            "probability": probability,
            "fold": fold_id,
            "audit": audit,
            "stress": stress,
        }
    return result


def cluster_draw(slides, rng):
    """A slide is the resampling unit: every patch in a drawn slide travels together."""
    unique = np.unique(slides)
    members = [np.flatnonzero(slides == s) for s in unique]
    return np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])


def scores(y, p):
    if len(np.unique(y)) != 2:
        return {k: float("nan") for k in METRICS}
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def rate(errors, mask):
    return float(errors[mask].mean()) if mask.any() else float("nan")


def interval(estimate, replicates):
    valid = np.asarray(replicates)[np.isfinite(replicates)]
    return {
        "estimate": float(estimate) if np.isfinite(estimate) else None,
        "ci95": np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
        "usable_draws": len(valid),
    }


def summarize(y, probabilities, slides, qc, n_bootstrap=1000, seed=2026):
    if n_bootstrap < 20 or len(np.unique(slides)) < 2:
        raise ValueError("at least 20 bootstrap draws and two slides required")
    point = {m: scores(y, p) for m, p in probabilities.items()}
    reps = {m: {k: [] for k in METRICS} for m in probabilities}
    errors = (probabilities["slide"] >= 0.5) != y
    qc_reps = {flag: {"flagged": [], "unflagged": [], "difference": []} for flag in qc}
    rng = np.random.default_rng(seed)
    # Precompute memberships so repeated draws do not repeatedly scan every slide.
    members = [np.flatnonzero(slides == s) for s in np.unique(slides)]
    for _ in range(n_bootstrap):
        idx = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        for mode, p in probabilities.items():
            for metric, value in scores(y[idx], p[idx]).items():
                reps[mode][metric].append(value)
        for flag, mask in qc.items():
            a, b = rate(errors[idx], mask[idx]), rate(errors[idx], ~mask[idx])
            qc_reps[flag]["flagged"].append(a)
            qc_reps[flag]["unflagged"].append(b)
            qc_reps[flag]["difference"].append(a - b)
    protocols = {m: {k: interval(point[m][k], reps[m][k]) for k in METRICS} for m in point}
    gap = {
        k: interval(
            point["random"][k] - point["slide"][k],
            np.array(reps["random"][k]) - np.array(reps["slide"][k]),
        )
        for k in METRICS
    }
    qc_stats = {}
    for flag, mask in qc.items():
        strata = {}
        for key, submask in (("flagged", mask), ("unflagged", ~mask)):
            strata[key] = {
                "n": int(submask.sum()),
                "errors": int(errors[submask].sum()),
                "slides": len(np.unique(slides[submask])),
                "error_rate": interval(rate(errors, submask), qc_reps[flag][key]),
            }
        strata["error_difference"] = interval(
            rate(errors, mask) - rate(errors, ~mask), qc_reps[flag]["difference"]
        )
        qc_stats[flag] = strata
    return {
        "protocols": protocols,
        "gap_random_minus_slide": gap,
        "qc": qc_stats,
        "bootstrap": {
            "unit": "source slide",
            "paired": True,
            "draws": n_bootstrap,
            "seed": seed,
            "confidence": 0.95,
            "limitation": "Conditional on fitted out-of-fold models; no retraining "
            "inside bootstrap. Not a causal estimate of leakage.",
        },
    }
