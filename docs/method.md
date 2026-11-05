# What this experiment can answer

I compare two ways of dividing the **same official PCam validation archive** into training and testing rounds. This keeps the evaluated patch population identical. It is a development experiment within one archive, not a replacement for the official PCam benchmark and not a test of hospital generalization.

## Data identity and labels

The download comes from [Zenodo record 2546921](https://zenodo.org/records/2546921), linked by the [PCam authors](https://github.com/basveeling/pcam). I pin the record's MD5 checksums and retain SHA-256 fingerprints in `reports/results.json`. The valid/test checksums in the upstream GitHub table differ from the Zenodo filenames. I use Zenodo's files and corresponding record checksums consistently, and check that every metadata row index and central-region label agrees with the HDF5 image/label ordering. The run command also checks the unpacked files against the compressed archives.

`wsi` is the original CAMELYON16 source slide, not a slide guessed from image appearance. Coordinates and both metadata tumor fields are used only for audit and alignment. They are never classifier inputs. The positive label means at least one tumor pixel in the central 32 by 32 region of a 96 by 96 patch. Tumor outside that region does not determine the label. I do not infer a hospital or a patient identity from the slide name. Keeping slides separate protects that grouping, but cannot guarantee independence of unidentified higher-level groups.

Only public, CC0 PCam images are used. No substitute dataset is used. Raw archives stay out of git. A few derived CC0 patch examples are saved with row IDs and source-slide IDs so they can be traced to the release. Source-slide identifiers are public dataset identifiers, not personal records.

## Fixed baseline

For each image I extract 84 color and texture features: channel means, standard deviations, quantiles, eight-bin channel histograms, gradient magnitude mean/standard deviation and Laplacian variance. I do this for the whole patch and its central region. The classifier is a random forest with 96 trees, maximum depth 16, at least five training observations per leaf, square-root feature sampling and two workers. The positive threshold is 0.5. There is no fitted image preprocessing, feature selection, threshold optimization, or hyperparameter search on these results.

Five-fold stratified random splitting gives a familiar optimistic comparison. Five-fold stratified **group** splitting keeps all patches from a source slide in the same fold. Stratification tries to maintain the class balance, but group constraints take priority. The report exposes training sizes, label balance, shared slide counts and shared image hashes per fold. Training data is the complement of the recorded test fold, so the membership can be reconstructed from `predictions.csv`. Both protocols make one out-of-fold prediction for every row. An exact image copy can still occur in the random protocol's training rows; row separation alone does not prevent that leakage. Each model is discarded after its held-out predictions are made.

Random splitting intentionally allows similar patches from the same source into both training and testing. The official PCam benchmark already separates source slides. A higher random-split score in this experiment does not show that the official benchmark leaks. The difference also reflects training composition and finite-sample variation. It is an observed protocol gap, not an isolated causal effect.

## Uncertainty

The accuracy, balanced accuracy and ROC AUC intervals use 1,000 source-slide bootstrap draws with a fixed seed. Each draw samples as many slides as the archive contains, with replacement, and retains all patches in each sampled slide. The same draw is used for both protocols. The random-minus-slide interval is computed from those paired differences, rather than subtracting the endpoints of two intervals. Metrics are patch-weighted within a draw: larger slides contribute more patches. This is not a macro-average that gives each slide equal importance.

The bootstrap holds out-of-fold model fits fixed. It captures slide-sampling variation in the saved predictions, not uncertainty from retraining on new datasets or choices of splits. Predictions from cross-validation have dependencies through shared training sets. These intervals are descriptive and can understate full procedure uncertainty. Repeating grouped cross-validation and bootstrapping model refits would be a more expensive follow-up. A bootstrap draw missing either label is omitted for classification metrics; usable draw counts are saved. QC strata with no observations return null, never a made-up zero rate. If a stratum has observations but no errors, its nonparametric bootstrap interval can collapse to zero. That reflects the observed sample only and cannot rule out future errors.

## Quality flags

The rules are fixed before model training, independent of labels, and operate on original RGB images:

- **Blur:** variance of the discrete grayscale Laplacian below 80, with grayscale on the 0 to 255 scale. Pale or naturally smooth tissue can also trigger it.
- **Background:** more than 80% of pixels have every channel above 220 and saturation below 0.12, where saturation is `(max - min) / max`.
- **Possible pen:** more than 0.5% of pixels belong to connected components of at least 12 candidate pixels. Blue candidates have B > 130, B > 1.5 R, B > 1.2 G and R < 100; green candidates have G > 130, G > 1.4 R and G > 1.3 B; red candidates have R > 180, G < 80 and B < 100.

These are transparent screening cues, not pathologist-confirmed defects. Natural tissue and stain can cause false positives; other pen colors or subtle blur can be missed. PCam already filters background during patch selection, so a small or empty background stratum is expected. Flags can overlap. I never discard flagged patches to improve headline scores. I compare flagged versus unflagged errors using slide-separated predictions, with cluster intervals for both rates and their difference. Flag counts, errors, and numbers of contributing slides are saved. These are associations, not causal effects or validated defect sensitivity/specificity.

## Controlled damage check

A seed-selected subset of up to 256 real patches is selected before fitting or examining errors. I create Gaussian-blurred images (radius 3), an 8-pixel blue line, and a white rectangle covering columns 0 through 87. The same held-out slide model scores the clean and damaged version. The report separates this experiment from naturally flagged patches. Original labels are retained for a sensitivity check; a large occlusion can remove the label's visual evidence. Synthetic flag rates do not validate detection of natural defects. `stress_predictions.json` saves every clean/damaged score and flag.

## Reproducibility and boundaries

`uv run tissue-patch-eval fetch` downloads and verifies the files. `run` checks their provenance, extracts features, trains, bootstraps and writes reports. `verify` checks artifact hashes and reconstructs the entire saved statistical summary from the row-level predictions without downloading raw images or training. It also reconstructs every fold audit field, duplicate count and example caption, checks example image fingerprints and exact seeded split membership, and recomputes controlled-damage summaries. A readable report must exactly match fresh generation from the checked results. The lockfile records dependencies; versions and all settings appear in the report JSON. Across platforms, floating-point calculations can vary at rounding precision.

This is a small baseline, without expert QC annotations, external validation, probability calibration or clinical testing. It is not a diagnostic tool and must not guide patient care.

## Sources

- [PCam authors' release, labeling, sampling and CC0 license](https://github.com/basveeling/pcam)
- [Official downloadable archive and checksums](https://zenodo.org/records/2546921)
- [Veeling et al., Rotation Equivariant CNNs for Digital Pathology](https://arxiv.org/abs/1806.03962)
- [Ehteshami Bejnordi et al., original CAMELYON16 study](https://doi.org/10.1001/jama.2017.14585)
- [scikit-learn StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
