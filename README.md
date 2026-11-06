I merged this into my existing [lab-assay-lims project](https://github.com/jackaI1/lab-assay-lims/tree/main/projects/tissue-patch-eval). The current code is in `projects/tissue-patch-eval/` there. I kept the code here so older links still work.

The move keeps the original layout, including `tests/test_data_qc.py`, `tests/test_published.py`, `tests/test_report.py`, `tests/test_evaluation.py`. I replaced the README with the report below and left out Git history, installed dependencies, caches and generated build output from the hub copy.

The 4 test source files moved with the code. No project-specific GitHub Actions workflow was retained; the report records the omitted checks, and the test sources can be run locally from the linked folder.

Other specific omissions: reports/ (generated output).

Omitted workflows: `tissue-patch-eval-ci.yml`: The relocated workflow failed in GitHub Actions and was removed to keep the hub checks independent. Local test sources remain available.

---

# tissue-patch-eval

I built a tissue-image evaluation to learn how the underlying idea works. This is a learning project.

## What happened

<!-- results:start -->
I evaluated 32,768 public patches from 54 source slides. Random splitting scored 16.08 percentage points higher in accuracy (95% interval: 9.42 to 22.94 points). This is the measured difference for this experiment, not a guaranteed effect on other data.

| Split method | Accuracy (95% interval) | Balanced accuracy | ROC AUC |
| --- | --- | --- | --- |
| Random patches | 91.83% [90.12%, 93.22%] | 91.83% [90.29%, 93.20%] | 0.9750 [0.9647, 0.9837] |
| Separate source slides | 75.75% [69.22%, 82.25%] | 75.74% [69.57%, 81.57%] | 0.8781 [0.8375, 0.9136] |
<!-- results:end -->

The intervals resample whole slides, rather than treat related patches as independent observations. I compare the same patches in both methods. A higher random-split score is evidence about this experiment, not a promise about another model or dataset. The report also shows balanced accuracy, which gives the two labels equal weight, and ROC AUC, which measures how well the scores rank the two labels.

## Try it

I use Python and [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the exact saved dependencies. The full run uses a CPU and downloads the official validation archive, without an account or API key. Allow about 2 GB of disk space for compressed and unpacked data. Download time depends on the archive server.

```sh
uv sync --frozen --python 3.12 --extra dev
uv run tissue-patch-eval fetch --data data
uv run tissue-patch-eval run --data data --out reports
uv run tissue-patch-eval verify --report reports
```

The last command also works on the committed report without downloading images or training. It checks file fingerprints, source-slide separation, and recalculates all score and QC confidence intervals from saved predictions. It prints the table above, so every measured figure here can be traced to a command.

To see each patch's label, source slide, coordinates, fold, score and quality measurements, open [predictions.csv](reports/predictions.csv). Settings, source checksums and the complete statistical summary are in [results.json](reports/results.json). Deliberately damaged examples have their own [prediction file](reports/stress_predictions.json).

## How I made the comparison

I use the source-slide identifiers supplied in PCam's public metadata, so I did not need a substitute grouping or dataset. I split the official validation archive into training and testing rounds. Each row gets a prediction from a model that did not train on that row. The random split can still put an exact copy of its image in training; the report exposes repeated images and checks whether copies cross folds. Random splitting mixes slides across the two sides; the other method keeps every patch from each slide together. The classifier sees only image color and texture measurements. It never sees a slide name, coordinate or metadata label as an input.

The official PCam benchmark already separates slides. My re-splitting experiment is not a benchmark submission or a finding of leakage in that published benchmark. It tests what happens when I ignore source identity within one archive. I do not infer hospitals from slide names or claim hospital-level validation.

I fix the classifier and quality rules before measuring performance. The confidence intervals use paired slide resampling, meaning the same sampled slides contribute to both methods and to their difference. They describe uncertainty in these saved predictions; they do not cover retraining on new data. Fold sizes and class balance appear in the report because changes in training composition can also affect the gap.

## What the quality flags mean

Blur is low image sharpness; possible pen marks are connected regions of unusually saturated color; background is a patch dominated by pale, nearly colorless pixels. These are screening rules, not expert-confirmed defects. Natural tissue or stain can trigger them, and defects can be missed. I keep flagged patches in the evaluation and compare their error rates with unflagged patches, including uncertainty and empty groups.

I separately add blur, a blue line and a white covering to real held-out patches. That checks whether the rules react to known changes and whether predictions change. It does not establish how well the flags find natural defects. The [method notes](docs/method.md) give the exact rules, feature definitions, split design, bootstrap procedure and limitations.

## Public sources

- [PatchCamelyon release](https://github.com/basveeling/pcam): images, central-region label definition, slide metadata, sampling and CC0 data license.
- [Official Zenodo archive](https://zenodo.org/records/2546921): the downloaded validation image, label and metadata files. I pin the record's checksums because the upstream GitHub valid/test checksum table differs from the archive filenames. The run checks row IDs and labels before training.
- [Veeling et al., Rotation Equivariant CNNs for Digital Pathology](https://arxiv.org/abs/1806.03962): the PCam paper.
- [Ehteshami Bejnordi et al., original CAMELYON16 study](https://doi.org/10.1001/jama.2017.14585): the underlying public slide dataset.
- [StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html): the splitter that preserves slide groups while trying to balance labels.

I used public data only. Raw archives are excluded from git. The saved patch examples are derived from the CC0 release; the project code is [MIT licensed](LICENSE).

## Tests and limits

There are 4 test source files in this copy. That source count does not establish an execution result. This is a source snapshot, and running it on a new machine may need environment-specific setup.



Generated output omitted from this copy: reports/ (generated output). Regenerate it with the project scripts.

The relocated workflow failed in GitHub Actions and was removed to keep the hub checks independent. Local test sources remain available.

No project-specific GitHub Actions workflow was retained. The local test sources remain in this folder.
