import numpy as np
import pytest

from tissue_patch_eval.evaluation import cluster_draw, cross_validate, make_folds, summarize


def population():
    groups = np.repeat(np.array([f"slide_{i}" for i in range(12)]), 12)
    y = np.tile([0, 1], len(groups) // 2)
    x = np.random.default_rng(11).normal(size=(len(y), 6))
    x[:, 0] += y
    return x, y, groups


def test_group_folds_have_zero_slide_overlap_and_test_every_row_once():
    _, y, slides = population()
    folds = make_folds(y, slides, "slide", 3, 5)
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(len(y)))
    for train, test in folds:
        assert not set(slides[train]) & set(slides[test])
        assert not set(train) & set(test)
    random = make_folds(y, slides, "random", 3, 5)
    assert any(set(slides[a]) & set(slides[b]) for a, b in random)


def test_cross_validation_is_aligned_reproducible_and_binary():
    x, y, slides = population()
    result = cross_validate(x, y, slides, n_splits=3, seed=5, trees=8)
    repeat = cross_validate(x, y, slides, n_splits=3, seed=5, trees=8)
    for mode in ("random", "slide"):
        p = result[mode]["probability"]
        assert p.shape == y.shape
        assert ((p >= 0) & (p <= 1)).all()
        np.testing.assert_allclose(p, repeat[mode]["probability"], rtol=0, atol=1e-14)
        assert (result[mode]["fold"] >= 0).all()
    assert all(f["overlap_slides"] == 0 for f in result["slide"]["audit"])


def test_invalid_eval_inputs_are_rejected():
    x, y, slides = population()
    with pytest.raises(ValueError):
        cross_validate(x[:-1], y, slides, trees=8)
    with pytest.raises(ValueError):
        make_folds(y, np.repeat("one_slide", len(y)), "slide", 3, 5)
    with pytest.raises(ValueError):
        make_folds(np.zeros_like(y), slides, "random", 3, 5)
    with pytest.raises(ValueError):
        make_folds(y, slides, "typo", 3, 5)


def test_bootstrap_draws_whole_slides_with_replacement():
    groups = np.repeat(np.arange(5), 4)
    draw = cluster_draw(groups, np.random.default_rng(2))
    assert len(draw) == len(groups)
    for g in range(5):
        counts = [np.count_nonzero(draw == i) for i in np.flatnonzero(groups == g)]
        assert len(set(counts)) == 1


def test_cluster_intervals_and_qc_empty_strata_are_honest():
    _, y, slides = population()
    p = y * 0.8 + 0.1
    q = {
        "blur": np.zeros(len(y), bool),
        "pen": y.astype(bool),
        "background": np.zeros(len(y), bool),
    }
    summary = summarize(y, {"random": p, "slide": p}, slides, q, 40, 8)
    assert summary["gap_random_minus_slide"]["accuracy"]["estimate"] == 0
    assert summary["gap_random_minus_slide"]["accuracy"]["ci95"] == [0, 0]
    assert summary["qc"]["blur"]["flagged"]["n"] == 0
    assert summary["qc"]["blur"]["flagged"]["error_rate"]["estimate"] is None
    assert summary["qc"]["blur"]["error_difference"]["ci95"] is None
    assert summary == summarize(y, {"random": p, "slide": p}, slides, q, 40, 8)


def test_slides_not_patches_determine_uncertainty():
    # Within each slide errors are perfectly correlated. Patch bootstrap would be narrow.
    slides = np.repeat(np.arange(10), 20)
    y = np.tile([0, 1], 100)
    p = np.where(np.repeat(np.arange(10) < 5, 20), y * 0.8 + 0.1, 0.9 - y * 0.8)
    q = {k: np.zeros(len(y), bool) for k in ("blur", "pen", "background")}
    s = summarize(y, {"random": p, "slide": p}, slides, q, 200, 9)
    low, high = s["protocols"]["slide"]["accuracy"]["ci95"]
    assert high - low > 0.35


def test_stress_predictions_use_only_the_matching_heldout_fold_model():
    x, y, slides = population()
    idx = np.array([0, 3, 15, 27, 101])
    result = cross_validate(
        x, y, slides, n_splits=3, seed=5, trees=8, stress_features={"unchanged": (idx, x[idx])}
    )
    np.testing.assert_allclose(
        result["slide"]["stress"]["unchanged"],
        result["slide"]["probability"][idx],
        rtol=0,
        atol=1e-14,
    )
