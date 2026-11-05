"""A GitHub-readable report and an offline HTML copy with embedded real examples."""

import base64
import html
import re
from pathlib import Path

import markdown


def value(x, percent=True):
    if x is None:
        return "not estimable"
    return f"{100 * x:.2f}%" if percent else f"{x:.4f}"


def estimate(stat, percent=True):
    point = value(stat["estimate"], percent)
    ci = stat["ci95"]
    return point + (
        f" [{value(ci[0], percent)}, {value(ci[1], percent)}]" if ci else " [no interval]"
    )


def metrics_table(result):
    ev = result["evaluation"]
    lines = [
        "| Split method | Accuracy (95% interval) | Balanced accuracy | ROC AUC |",
        "| --- | --- | --- | --- |",
    ]
    for mode, title in (("random", "Random patches"), ("slide", "Separate source slides")):
        metrics = ev["protocols"][mode]
        lines.append(
            f"| {title} | {estimate(metrics['accuracy'])} | "
            f"{estimate(metrics['balanced_accuracy'])} | {estimate(metrics['roc_auc'], False)} |"
        )
    return "\n".join(lines)


def readme_metrics(result):
    d = result["dataset"]
    stat = result["evaluation"]["gap_random_minus_slide"]["accuracy"]
    low, high = stat["ci95"]
    return (
        f"I evaluated {d['patches']:,} public patches from {d['slides']} source slides. "
        f"Random splitting scored {100 * stat['estimate']:.2f} percentage points higher in accuracy "
        f"(95% interval: {100 * low:.2f} to {100 * high:.2f} points). "
        "This is the measured difference for this experiment, not a guaranteed effect on other data.\n\n"
        + metrics_table(result)
    )


def write_report(result, out: Path):
    d, config = result["dataset"], result["configuration"]
    b = result["evaluation"]["bootstrap"]
    text = [
        "# Tissue patch evaluation",
        "",
        (
            "I built this comparison because I wanted to check whether tissue patches from the same "
            "slide make an image classifier look more reliable than it is on a new slide."
        ),
        "",
        (
            "**This is a public-data learning experiment, not a diagnostic tool.** "
            "It cannot decide whether a person has cancer."
        ),
        "",
        "## What I measured",
        "",
        readme_metrics(result),
        "",
        (
            "Accuracy is the share of correct patch labels. Balanced accuracy averages the success "
            "rate for each label. ROC AUC measures how well the scores rank positive patches above "
            "negative patches; higher is better. Square brackets show uncertainty, not score ranges."
        ),
        "",
        (
            f"Source: {html.escape(d['source'])}. Positive labels: {d['positives']:,}. "
            f"Repeated coordinate rows: {d['duplicate_coordinate_rows']}; "
            f"repeated image rows: {d['duplicate_image_rows']}."
        ),
        "",
        "## What makes this a source-aware comparison",
        "",
        (
            f"I train a fixed {config['trees']}-tree random forest using {config['features']} color and "
            f"texture measurements per image, with {config['folds']} folds (separate train/test rounds). "
            "Each row gets exactly one prediction from a model that did not train on that row. The random split can still put an exact copy of its image in training. "
            "The random protocol mixes source slides across training and testing. The slide protocol "
            "keeps every patch from a source slide together. Both score the same rows, use the same "
            "image features, hyperparameters, probability threshold of 0.5, and seed. "
            "Slide IDs, coordinates and metadata labels never enter the model. Nothing is tuned on these results."
        ),
        "",
        (
            "This re-partitions the official validation archive as a small development experiment. "
            "It is not a score on the official PCam train/test benchmark. The archive contains exact image repeats; the audit shows whether those copies cross folds. Repeated images can make random evaluation easier, so the gap cannot be attributed only to stain or slide appearance. Group folds can have different "
            "class mixes and training sizes, which are shown below. The observed gap combines source "
            "similarity and split composition; it does not isolate a causal effect of leakage. "
            "The published benchmark already separates slides. I do not claim to have found leakage in it. "
            "I use actual CAMELYON16 slide IDs, and do not infer hospitals from slide names."
        ),
        "",
        (
            f"Intervals use {b['draws']:,} paired bootstrap draws: sample whole source slides with "
            "replacement, keep all of their patches, then recompute both scores and their difference. "
            "The interval for the difference uses the same slides in both protocols. These intervals "
            "condition on the fitted models and this archive. They do not include retraining uncertainty "
            "or establish clinical performance."
        ),
        "",
        (
            "| Protocol / fold | Train patches | Test patches | Train slides | Test slides | Shared slides | "
            "Shared image hashes | Train positive share | Test positive share |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode, folds in result["fold_audit"].items():
        for f in folds:
            text.append(
                f"| {mode} / {f['fold']} | {f['train_n']} | {f['test_n']} | "
                f"{f['train_slides']} | {f['test_slides']} | {f['overlap_slides']} | "
                f"{f['overlap_image_hashes']} | {value(f['train_positive_fraction'])} | "
                f"{value(f['test_positive_fraction'])} |"
            )
    text += [
        "",
        "## Quality flags and real prediction errors",
        "",
        (
            "I flag low sharpness (blur), large connected areas of saturated blue/green/red (possible pen), "
            "and mostly pale, unsaturated pixels (background). The thresholds are fixed image heuristics, "
            "not expert annotations. Natural stain or tissue structure can trigger them. "
            "Flags can overlap; I keep every patch in the evaluation rather than remove hard cases."
        ),
        "",
        (
            "The table uses slide-separated predictions. Each error rate has a slide-bootstrap interval. "
            "A positive difference means flagged patches made more errors. An association does not prove "
            "that a defect caused the errors. A missing interval means there were no observations "
            "in a needed stratum, not that its error rate was zero. A zero-error stratum can have a collapsed bootstrap interval; that does not guarantee zero future errors."
        ),
        "",
        (
            "| Flag | Flagged patches / slides | Flagged error rate | Unflagged patches | "
            "Unflagged error rate | Difference (percentage points) |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for flag, q in result["evaluation"]["qc"].items():
        a, c, gap = q["flagged"], q["unflagged"], q["error_difference"]
        diff = estimate(gap).replace("%", " pp")
        text.append(
            f"| {flag} | {a['n']} / {a['slides']} | {estimate(a['error_rate'])} | "
            f"{c['n']} | {estimate(c['error_rate'])} | {diff} |"
        )
    text += [
        "",
        "### Real patch examples",
        "",
        (
            "Examples are the first available correct and incorrect predictions in each flag group, "
            "with duplicate examples removed. They are illustrative, not a random prevalence sample. "
            "A positive label means tumor pixels occur in the central region; surrounding tumor alone "
            "does not make the label positive. The score is the model's positive-class score, "
            "not a calibrated clinical probability."
        ),
        "",
    ]
    for e in result["examples"]:
        caption = (
            f"Row {e['row']}, {e['slide']}; label {e['label']}, score {e['probability']:.3f}; "
            f"flags: {', '.join(e['flags']) or 'none'}; "
            f"{'incorrect' if e['error'] else 'correct'} prediction"
        )
        text.extend([f"![{html.escape(caption)}]({e['path']})", "", html.escape(caption), ""])
    text += [
        "## Deliberately damaged patches: a separate stress check",
        "",
        (
            "I pick a fixed random subset before training and apply Gaussian blur, a blue line, or "
            "a white rectangle covering most of each real image. Each damaged patch is scored by the "
            "same slide-held-out model as its original. These are simulated defects, not verified "
            "natural pen marks or background. I retain the original label to measure sensitivity to "
            "damage, although occlusion can remove the evidence needed to infer that label. "
            "Detection rates here are not estimates of sensitivity on natural defects."
        ),
        "",
        "| Damage | Patches | Intended flag fires | Original error rate | Damaged error rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for flag, s in result["stress"].items():
        text.append(
            f"| {flag} | {s['n']} | {value(s['flagged_fraction'])} | "
            f"{value(s['clean_error_rate'])} | {value(s['damaged_error_rate'])} |"
        )
    text += [
        "",
        "| Original | Blur | Simulated pen | Simulated background |",
        "| --- | --- | --- | --- |",
        (
            "| ![Original](examples/stress_clean.png) | ![Blur](examples/stress_blur.png) | "
            "![Simulated pen](examples/stress_pen.png) | "
            "![Simulated background](examples/stress_background.png) |"
        ),
        "",
        "## Reproduce and inspect",
        "",
        "```sh",
        "uv sync --frozen --extra dev",
        "uv run tissue-patch-eval fetch --data data",
        "uv run tissue-patch-eval run --data data --out reports",
        "uv run tissue-patch-eval verify --report reports",
        "```",
        "",
        (
            "`results.json` contains configuration, file checksums, versions, fold audits and all "
            "statistics. `predictions.csv` retains every row's source, coordinates, patch hash, label, "
            "folds, scores and QC measurements. `stress_predictions.json` retains the separate damaged "
            "patch results. The verify command replays metrics and every interval without raw images. "
            "See [method details](../docs/method.md) for thresholds and scope."
        ),
        "",
        (
            "Public sources: [PCam release and labeling](https://github.com/basveeling/pcam), "
            "[download archive](https://zenodo.org/records/2546921), "
            "[PCam paper](https://arxiv.org/abs/1806.03962), "
            "[CAMELYON16 paper](https://doi.org/10.1001/jama.2017.14585). "
            "The source patches and these derived examples are CC0. No private data was used."
        ),
        "",
    ]
    md = "\n".join(text)
    (out / "report.md").write_text(md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])

    def embed(match):
        path = out / match.group(1)
        return 'src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '"'

    body = re.sub(r'src="(examples/[^"]+\.png)"', embed, body)
    style = (
        "body{font:17px/1.6 system-ui,sans-serif;color:#222;background:#faf9f5;margin:0}"
        "main{max-width:1100px;margin:auto;padding:32px}h1,h2{line-height:1.2}"
        "table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}"
        "td,th{padding:10px;border:1px solid #ccc;text-align:left}"
        "th{background:#e7ede8}img{width:144px;image-rendering:pixelated}"
        "pre{padding:16px;background:#e9e9e9;overflow:auto}a{color:#145a42}"
    )
    (out / "report.html").write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Tissue patch evaluation</title><style>{style}</style>"
        f"</head><body><main>{body}</main></body></html>\n"
    )
