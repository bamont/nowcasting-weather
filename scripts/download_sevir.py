"""Download a subset of SEVIR (VIL modality) from the public S3 bucket.

No AWS account is required (unsigned requests). Always run with --dry-run first
to see how much data you are about to download.

Examples:
    uv run python scripts/download_sevir.py --years 2019 --dry-run
    uv run python scripts/download_sevir.py --years 2019 --max-files 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import fnmatch

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

BUCKET = "sevir"
REGION = "us-west-2"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def list_files(s3, prefix: str) -> list[dict]:
    paginator = s3.get_paginator("list_objects_v2")
    out: list[dict] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        out.extend(o for o in page.get("Contents", []) if o["Key"].endswith(".h5"))
    return out


def download(s3, key: str, dest: Path, size: int) -> None:
    if dest.exists() and dest.stat().st_size == size:
        print(f"skip (already complete): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=size, unit="B", unit_scale=True, desc=dest.name) as bar:
        s3.download_file(BUCKET, key, str(dest), Callback=bar.update)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/sevir"))
    parser.add_argument("--modality", default="vil")
    parser.add_argument("--years", nargs="+", default=["2019"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pattern", default="*", help="glob on file name, e.g. '*STORMEVENTS*'")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))

    files: list[dict] = []
    for year in args.years:
        files.extend(list_files(s3, f"data/{args.modality}/{year}/"))
    files.sort(key=lambda o: o["Key"])
    files = [o for o in files if fnmatch.fnmatch(Path(o["Key"]).name, args.pattern)]
    if args.max_files:
        files = files[: args.max_files]

    total = sum(o["Size"] for o in files)
    print(f"{len(files)} file(s), total {human(total)}")
    for o in files:
        print(f"  {o['Key']}  ({human(o['Size'])})")
    if args.dry_run:
        return

    catalog_size = s3.head_object(Bucket=BUCKET, Key="CATALOG.csv")["ContentLength"]
    download(s3, "CATALOG.csv", args.dest / "CATALOG.csv", catalog_size)
    for o in files:
        download(s3, o["Key"], args.dest / o["Key"], o["Size"])


if __name__ == "__main__":
    main()
