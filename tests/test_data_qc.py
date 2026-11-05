import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from tissue_patch_eval.data import load_data, verify_file
from tissue_patch_eval.qc import extract_features, quality_flags


def texture():
    rng = np.random.default_rng(8)
    return rng.integers(30, 220, (96, 96, 3), dtype=np.uint8)


def test_checksum_rejects_changed_file(tmp_path):
    p = tmp_path / "input"
    p.write_bytes(b"public")
    verify_file(p, hashlib.md5(b"public").hexdigest())
    with pytest.raises(ValueError, match="checksum"):
        verify_file(p, "0" * 32)


def fixture_data(root: Path, wrong_label=False):
    prefix = root / "camelyonpatch_level_2_split_valid"
    with h5py.File(str(prefix) + "_x.h5", "w") as f:
        f["x"] = np.stack([texture()] * 2)
    with h5py.File(str(prefix) + "_y.h5", "w") as f:
        f["y"] = np.array([0, 0 if wrong_label else 1]).reshape(2, 1, 1, 1)
    Path(str(prefix) + "_meta.csv").write_text(
        ",coord_y,coord_x,tumor_patch,center_tumor_patch,wsi\n"
        "0,0,0,False,False,slide_a\n1,96,0,True,True,slide_b\n"
    )


def test_loader_preserves_row_identity_and_checks_label_alignment(tmp_path):
    fixture_data(tmp_path)
    data = load_data(tmp_path)
    assert data.labels.tolist() == [0, 1]
    assert data.slides.tolist() == ["slide_a", "slide_b"]
    assert data.images.shape == (2, 96, 96, 3)
    fixture_data(tmp_path, wrong_label=True)
    with pytest.raises(ValueError, match="alignment"):
        load_data(tmp_path)


def test_loader_rejects_reordered_metadata(tmp_path):
    fixture_data(tmp_path)
    p = tmp_path / "camelyonpatch_level_2_split_valid_meta.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2], lines[1]]))
    with pytest.raises(ValueError, match="row"):
        load_data(tmp_path)


def test_quality_flags_detect_controlled_damage():
    a = texture()
    assert not quality_flags(a)["blur"]
    blurred = np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(4)))
    assert quality_flags(blurred)["blur"]
    assert quality_flags(np.full_like(a, 255))["background"]
    pen = Image.fromarray(a)
    ImageDraw.Draw(pen).line((5, 5, 85, 85), fill=(0, 40, 255), width=8)
    assert quality_flags(np.array(pen))["pen"]
    # Natural purple tissue must not be treated as blue pen just for being purple.
    purple = np.full_like(a, [145, 80, 170])
    assert not quality_flags(purple)["pen"]


@pytest.mark.parametrize(
    "a", [np.zeros((96, 96)), np.zeros((96, 96, 3)), np.zeros((32, 32, 3), dtype=np.uint8)]
)
def test_rgb_contract(a):
    with pytest.raises(ValueError, match="96.*uint8"):
        quality_flags(a)


def test_features_are_finite_deterministic_and_image_only():
    a = texture()
    f = extract_features(a)
    assert f.ndim == 1 and len(f) >= 30
    assert np.isfinite(f).all()
    np.testing.assert_array_equal(f, extract_features(a))
    assert not np.array_equal(f, extract_features(np.full_like(a, 255)))


def test_fetch_is_atomic_verified_and_replaces_stale_unpacked_files(tmp_path, monkeypatch):
    import gzip
    import io

    from tissue_patch_eval import data

    content = gzip.compress(b"a tiny public-format fixture")
    md5 = hashlib.md5(content).hexdigest()
    monkeypatch.setattr(data, "FILES", {"fixture.h5.gz": md5})
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content))
    (tmp_path / "fixture.h5").write_bytes(b"stale")
    data.fetch_data(tmp_path)
    assert (tmp_path / "fixture.h5").read_bytes() == b"a tiny public-format fixture"
    assert not list(tmp_path.glob("*.part"))


def test_failed_download_leaves_no_valid_or_partial_archive(tmp_path, monkeypatch):
    import io

    from tissue_patch_eval import data

    monkeypatch.setattr(data, "FILES", {"fixture.h5.gz": "0" * 32})
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="checksum"):
        data.fetch_data(tmp_path)
    assert not list(tmp_path.iterdir())


def test_provenance_rejects_tampered_unpacked_data(tmp_path, monkeypatch):
    import gzip

    from tissue_patch_eval import cli

    packed = gzip.compress(b"known public fixture")
    name = "fixture.h5.gz"
    (tmp_path / name).write_bytes(packed)
    (tmp_path / "fixture.h5").write_bytes(b"known public fixture")
    monkeypatch.setattr(cli, "FILES", {name: hashlib.md5(packed).hexdigest()})
    manifest = cli.provenance(tmp_path)
    assert manifest["substitute"] is None
    (tmp_path / "fixture.h5").write_bytes(b"changed outside download")
    with pytest.raises(ValueError, match="unpacked checksum"):
        cli.provenance(tmp_path)
