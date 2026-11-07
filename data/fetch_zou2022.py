"""Fetch Zou et al. 2022 probabilistic reversal learning data from OSF.

DOI: 10.3389/fpsyt.2022.800290
Data: https://osf.io/jde6m/
~70 healthy participants, probabilistic switch task.

Usage: python data/fetch_zou2022.py
"""
import hashlib
import json
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "zou2022")
MANIFEST_PATH = os.path.join(ROOT, "manifest_zou2022.json")

OSF_PROJECT = "jde6m"
OSF_API = "https://api.osf.io/v2"
HEADERS = {"Accept": "application/json"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # List files in OSF project
    print("Fetching file listing from OSF...")
    url = f"{OSF_API}/nodes/{OSF_PROJECT}/files/osfstorage/?page[size]=100"
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())

    files = []
    for item in data["data"]:
        a = item["attributes"]
        name = a["name"]
        kind = a["kind"]
        size = a.get("size", 0)
        dl = item["links"].get("download", "")
        print(f"  {kind}: {name} ({size} bytes)")
        files.append({"name": name, "kind": kind, "size": size,
                       "download": dl})

    # Check for next pages
    next_url = data.get("links", {}).get("next")
    while next_url:
        req = urllib.request.Request(next_url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        for item in data["data"]:
            a = item["attributes"]
            files.append({"name": a["name"], "kind": a["kind"],
                          "size": a.get("size", 0),
                          "download": item["links"].get("download", "")})
        next_url = data.get("links", {}).get("next")

    # Download all files (small dataset)
    manifest_entries = []
    for f in files:
        if f["kind"] != "file":
            # It's a folder, need to recurse
            print(f"  Folder: {f['name']} — skipping (check manually)")
            continue
        path = os.path.join(OUT_DIR, f["name"])
        if os.path.exists(path):
            sha = sha256_file(path)
        else:
            print(f"  Downloading {f['name']}...", end="", flush=True)
            try:
                urllib.request.urlretrieve(f["download"], path)
                sha = sha256_file(path)
                print(" OK")
            except Exception as e:
                print(f" FAILED: {e}")
                sha = None
        manifest_entries.append({
            "file": f["name"], "size": f["size"],
            "sha256": sha, "url": f["download"],
        })
        time.sleep(0.3)

    manifest = {
        "dataset": "Zou et al. 2022",
        "source": "https://osf.io/jde6m/",
        "doi": "10.3389/fpsyt.2022.800290",
        "license": "CC-By Attribution 4.0 International",
        "citation": ("Zou, A.R. et al. Impulsivity Relates to Multi-Trial "
                     "Choice Strategy in Probabilistic Reversal Learning. "
                     "Front. Psychiatry 13 (2022)."),
        "date_fetched": time.strftime("%Y-%m-%d"),
        "files": manifest_entries,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
