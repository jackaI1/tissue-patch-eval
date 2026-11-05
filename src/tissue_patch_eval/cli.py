"""CPU-only public-data workflow with reproducible defaults."""

import argparse
import gzip
import hashlib
from pathlib import Path

from .data import FILES, RECORD, digest, fetch_data, load_data, verify_file
from .pipeline import run_pipeline, verify_report
from .report import readme_metrics


def provenance(root):
    files = {}
    for name, md5 in FILES.items():
        path = root / name
        verify_file(path, md5)
        files[name] = {"url": f"{RECORD}/files/{name}", "md5": md5, "sha256": digest(path)}
        if name.endswith(".gz"):
            raw = path.with_suffix("")
            h = hashlib.sha256()
            with gzip.open(path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            raw_sha = digest(raw)
            if h.hexdigest() != raw_sha:
                raise ValueError(f"unpacked checksum mismatch: {raw.name}; fetch again")
            files[raw.name] = {"sha256": raw_sha}
    return {
        "source": "Official PCam validation archive from Zenodo record 2546921, "
        "with original CAMELYON16 wsi metadata",
        "record": RECORD,
        "license": "CC0-1.0",
        "files": files,
        "substitute": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download verified official validation files")
    fetch.add_argument("--data", type=Path, default=Path("data"))
    run = sub.add_parser("run", help="train both protocols and save the public-data report")
    run.add_argument("--data", type=Path, default=Path("data"))
    run.add_argument("--out", type=Path, default=Path("reports"))
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--trees", type=int, default=96)
    run.add_argument("--bootstrap", type=int, default=1000)
    run.add_argument("--seed", type=int, default=2026)
    verify = sub.add_parser("verify", help="replay all report statistics from saved predictions")
    verify.add_argument("--report", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_data(args.data)
    elif args.command == "run":
        manifest = provenance(args.data)
        run_pipeline(
            load_data(args.data),
            args.out,
            manifest,
            args.folds,
            args.trees,
            args.bootstrap,
            args.seed,
        )
        print(f"Saved {args.out / 'report.md'} and offline report.html")
    else:
        result = verify_report(args.report)
        print(readme_metrics(result))


if __name__ == "__main__":
    main()
