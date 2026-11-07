"""Fetch Confidence Database from OSF (https://osf.io/s46pr/).

Downloads data CSVs and readme files for perceptual 2AFC datasets
with >=200 trials per subject. Computes SHA-256 for manifest.

Rahnev et al. (2020), Nature Human Behaviour, DOI: 10.1038/s41562-019-0813-1.
License: CC-By Attribution 4.0 International.

Usage: python data/fetch_confidence_db.py
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "confidence_db")
MANIFEST_PATH = os.path.join(ROOT, "manifest_confidence_db.json")

OSF_PROJECT = "s46pr"
OSF_API = "https://api.osf.io/v2"
HEADERS = {"Accept": "application/json"}

# Surviving datasets from step 0a (perceptual, >=200 trials)
# Will be populated from Database_Information.xlsx
METADATA_URL = "https://osf.io/download/xkc7b/"


def get_folder_files(folder_id):
    """Get all files in an OSF folder (paginated)."""
    files = []
    url = (f"{OSF_API}/nodes/{OSF_PROJECT}/files/osfstorage/"
           f"{folder_id}/?page[size]=200")
    while url:
        req = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        files.extend(data["data"])
        url = data.get("links", {}).get("next")
        if url:
            time.sleep(0.5)  # rate limit
    return files


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, path):
    """Download with retry."""
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, path)
            return True
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2)
    return False


def load_surviving_datasets():
    """Load list of surviving dataset names from metadata."""
    try:
        import openpyxl
    except ImportError:
        print("pip install openpyxl")
        sys.exit(1)

    xlsx_path = os.path.join(ROOT, "Database_Information.xlsx")
    if not os.path.exists(xlsx_path):
        print("Downloading Database_Information.xlsx...")
        urllib.request.urlretrieve(METADATA_URL, xlsx_path)

    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]

    surviving = []
    for r in range(2, ws.max_row + 1):
        row = {headers[c]: ws.cell(r, c + 1).value for c in range(len(headers))}
        cat = str(row.get("Category", ""))
        if "Percep" not in cat:
            continue
        try:
            min_trials = int(row.get("Min_trials_per_subject", 0))
        except (TypeError, ValueError):
            min_trials = 0
        if min_trials < 200:
            continue
        try:
            n_cond = int(row.get("Num_tasks_x_conditions", 1))
        except (TypeError, ValueError):
            n_cond = 1
        surviving.append({
            "name": row["Name_in_database"],
            "n_conditions": n_cond,
            "min_trials": min_trials,
            "stimuli": row.get("Stimuli", ""),
        })
    return surviving


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading surviving dataset list...")
    datasets = load_surviving_datasets()
    print(f"  {len(datasets)} datasets to fetch")

    # Build set of needed file names
    needed_data = {f"data_{d['name']}.csv" for d in datasets}
    needed_readme = {f"readme_{d['name']}.txt" for d in datasets}
    needed = needed_data | needed_readme

    # Get OSF file listing
    print("Fetching OSF file listing...")
    # Find the Confidence Database folder ID
    url = f"{OSF_API}/nodes/{OSF_PROJECT}/files/osfstorage/?page[size]=100"
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=30)
    top = json.loads(resp.read())
    folder_id = None
    for item in top["data"]:
        if item["attributes"]["name"] == "Confidence Database":
            # Extract folder ID from path
            folder_id = item["id"]
            break
    if not folder_id:
        print("ERROR: Confidence Database folder not found")
        sys.exit(1)

    # Get all files in folder
    folder_url = (f"{OSF_API}/nodes/{OSF_PROJECT}/files/osfstorage/"
                  f"{folder_id}/?page[size]=200")
    all_files = []
    page_url = folder_url
    while page_url:
        req = urllib.request.Request(page_url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        all_files.extend(data["data"])
        page_url = data.get("links", {}).get("next")
        if page_url:
            time.sleep(0.5)
    print(f"  {len(all_files)} files in folder")

    # Match needed files
    to_download = []
    for f in all_files:
        name = f["attributes"]["name"]
        if name in needed:
            dl_url = f["links"]["download"]
            to_download.append((name, dl_url, f["attributes"].get("size", 0)))

    print(f"  {len(to_download)} files matched (of {len(needed)} needed)")
    missing = needed - {name for name, _, _ in to_download}
    if missing:
        print(f"  Missing: {len(missing)} files")
        for m in sorted(missing)[:10]:
            print(f"    {m}")

    # Download
    manifest_entries = []
    for i, (name, url, size) in enumerate(to_download):
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path):
            sha = sha256_file(path)
            manifest_entries.append({
                "file": name, "size": os.path.getsize(path),
                "sha256": sha, "url": url,
            })
            continue
        print(f"  [{i+1}/{len(to_download)}] {name} ({size} bytes)...",
              end="", flush=True)
        ok = download_file(url, path)
        if ok:
            sha = sha256_file(path)
            manifest_entries.append({
                "file": name, "size": os.path.getsize(path),
                "sha256": sha, "url": url,
            })
            print(" OK")
        else:
            print(" FAILED")
        time.sleep(0.3)

    # Write manifest
    manifest = {
        "dataset": "Confidence Database",
        "source": "https://osf.io/s46pr/",
        "doi": "10.1038/s41562-019-0813-1",
        "license": "CC-By Attribution 4.0 International",
        "citation": ("Rahnev, D. et al. The Confidence Database. "
                     "Nat Hum Behav 4, 317-325 (2020)."),
        "date_fetched": time.strftime("%Y-%m-%d"),
        "n_datasets": len(datasets),
        "datasets": [d["name"] for d in datasets],
        "files": manifest_entries,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Downloaded: {len(manifest_entries)} files")


if __name__ == "__main__":
    main()
