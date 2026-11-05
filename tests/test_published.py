"""Published claims must match the generated public-data evidence."""

import json
from pathlib import Path

from tissue_patch_eval.report import readme_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_readme_measured_block_matches_command_output():
    r = json.loads((ROOT / "reports/results.json").read_text())
    assert r["dataset"]["substitute"] is None
    assert r["dataset"]["record"] == "https://zenodo.org/records/2546921"
    readme = (ROOT / "README.md").read_text()
    block = readme.split("<!-- results:start -->\n")[1].split("\n<!-- results:end -->")[0]
    assert block == readme_metrics(r)
    assert "skills demonstrated" not in readme.lower()
    assert "—" not in readme
    assert all(f["overlap_slides"] == 0 for f in r["fold_audit"]["slide"])
