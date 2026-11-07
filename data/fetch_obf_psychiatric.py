"""Fetch OBF-Psychiatric actigraphy dataset from Zenodo.

DOI: 10.1038/s41597-025-04384-3
Data: https://zenodo.org/records/13754984
License: CC BY 4.0

Motor activity recordings, 162 individuals, 1565 days.
Groups: depression, schizophrenia, ADHD, control.

Usage: python data/fetch_obf_psychiatric.py
"""
import hashlib
import json
import os
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "obf_psychiatric")
MANIFEST_PATH = os.path.join(ROOT, "manifest_obf_psychiatric.json")

ZENODO_URL = "https://zenodo.org/records/13754984/files/OBF-Psychiatric%20Dataset.zip"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    zip_path = os.path.join(ROOT, "obf_psychiatric.zip")

    if not os.path.exists(zip_path):
        print("Downloading OBF-Psychiatric Dataset.zip...")
        urllib.request.urlretrieve(ZENODO_URL, zip_path)
        print(f"  Downloaded: {os.path.getsize(zip_path)} bytes")
    else:
        print(f"Already downloaded: {zip_path}")

    sha = sha256_file(zip_path)
    print(f"  SHA-256: {sha}")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(OUT_DIR)
        members = zf.namelist()
    print(f"  Extracted: {len(members)} files")

    # List contents
    for m in members[:10]:
        print(f"    {m}")
    if len(members) > 10:
        print(f"    ... ({len(members)} total)")

    # Manifest
    manifest = {
        "dataset": "OBF-Psychiatric",
        "source": "https://zenodo.org/records/13754984",
        "doi": "10.1038/s41597-025-04384-3",
        "license": "CC BY 4.0",
        "citation": ("OBF-Psychiatric, a motor activity dataset. "
                     "Scientific Data 12 (2025)."),
        "date_fetched": time.strftime("%Y-%m-%d"),
        "zip_sha256": sha,
        "zip_size": os.path.getsize(zip_path),
        "n_files": len(members),
        "files": members[:20],
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
