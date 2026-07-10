"""Download public ECG datasets (PTB-XL) reproducibly.

Raw data is NOT committed to git (see .gitignore). Run this once to fetch it.

Usage:
    python scripts/download_data.py --dataset ptbxl

Why HuggingFace and not PhysioNet directly
------------------------------------------
PhysioNet's HTTPS endpoint is unreachable from this machine's network (direct
connections are dropped, and the local proxy resets the TLS handshake to
physionet.org specifically).  HuggingFace / hf-mirror.com *are* reachable, and
``longisland3/ptb-xl`` is a byte-complete WFDB mirror of PTB-XL 1.0.3 (records100/,
records500/, ptbxl_database.csv, scp_statements.csv, plus a single 1.83 GB
``ptb-xl-data.zip`` bundle).  We pull the single zip (one resumable file) via the
HuggingFace hub with ``HF_ENDPOINT=https://hf-mirror.com`` and extract it.

If your network reaches PhysioNet directly, ``--source physionet`` restores the
original single-URL download.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HF_REPO = "longisland3/ptb-xl"
HF_MIRROR = "https://hf-mirror.com"
PHYSIONET_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)


def _hf_download(repo: str, filename: str, dest_dir: Path) -> Path:
    """Resumable single-file download from the hf-mirror resolve URL via curl.

    ``huggingface_hub``'s HEAD/etag resolution fails through the local proxy, and
    Windows' ``curl`` (schannel) resets on physionet.org but works fine for
    hf-mirror.com.  So we point curl at the LFS resolve URL and let it follow the
    redirect chain (hf-mirror -> huggingface.co -> CDN) with resume + retries.
    """
    import subprocess

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name
    url = f"{HF_MIRROR}/datasets/{repo}/resolve/main/{filename}"
    print(f"[curl] {url}\n       -> {dest}")
    cmd = [
        "curl", "-L", "-C", "-", "--fail",
        "--retry", "20", "--retry-delay", "5", "--retry-all-errors",
        "--connect-timeout", "30",
        "-o", str(dest), url,
    ]
    subprocess.run(cmd, check=True)
    print(f"[done] {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def _extract(zip_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path.name} -> {out}")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        root = names[0].split("/")[0] if "/" in names[0] else ""
        for n in names:
            rel = n[len(root) + 1 :] if root and n.startswith(root + "/") else n
            if not rel:
                continue
            target = out / rel
            if target.exists() and not n.endswith("/"):
                continue
            if n.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    print("[ok] extracted to", out)


def download_ptbxl(source: str) -> None:
    out = DATA_DIR / "ptbxl"
    out.mkdir(parents=True, exist_ok=True)
    if source == "hf":
        # Label CSVs first (small, needed for stratification), then the signal bundle.
        for csv in ("ptbxl_database.csv", "scp_statements.csv"):
            _hf_download(HF_REPO, csv, out)
        zip_path = _hf_download(HF_REPO, "ptb-xl-data.zip", DATA_DIR / "_ptbxl_zip")
        _extract(zip_path, out)
    elif source == "physionet":
        import urllib.request

        zip_path = DATA_DIR / "ptbxl.zip"
        if not zip_path.exists():
            print(f"[get] {PHYSIONET_URL}")
            urllib.request.urlretrieve(PHYSIONET_URL, zip_path)
        _extract(zip_path, out)
    else:
        raise ValueError(source)
    print("[ready] PTB-XL at", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ptbxl"], default="ptbxl")
    ap.add_argument("--source", choices=["hf", "physionet"], default="hf",
                    help="hf = HuggingFace mirror (default, GFW-friendly); physionet = direct")
    args = ap.parse_args()
    if args.dataset == "ptbxl":
        download_ptbxl(args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
