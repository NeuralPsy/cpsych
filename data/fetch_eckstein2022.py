"""Fetch Eckstein et al. 2022 reversal learning data from OSF.

Two sources:
1. Trial-level: osf.io/7wuh4 (DevCogNeurosci, 291 participant CSVs)
2. Summary + demographics: osf.io/h4qr6 (eLife, 4 consolidated CSVs)

DOI (DevCogNeurosci): 10.1016/j.dcn.2022.101106
DOI (eLife): 10.7554/eLife.75474
License: CC-By Attribution 4.0 International

Usage: python data/fetch_eckstein2022.py
"""
import hashlib
import json
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "eckstein2022")
MANIFEST_PATH = os.path.join(ROOT, "manifest_eckstein2022.json")

OSF_API = "https://api.osf.io/v2"
HEADERS = {"Accept": "application/json"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_osf_project(project_id, out_dir, label):
    """Download all files from an OSF project (flat, no folders)."""
    entries = []
    url = f"{OSF_API}/nodes/{project_id}/files/osfstorage/?page[size]=200"

    while url:
        req = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())

        for item in data["data"]:
            a = item["attributes"]
            if a["kind"] != "file":
                continue
            name = a["name"]
            dl = item["links"]["download"]
            size = a.get("size", 0)
            path = os.path.join(out_dir, name)

            if os.path.exists(path) and os.path.getsize(path) > 0:
                sha = sha256_file(path)
            else:
                print(f"  [{label}] {name} ({size})...",
                      end="", flush=True)
                try:
                    opener = urllib.request.build_opener(
                        urllib.request.HTTPRedirectHandler)
                    req2 = urllib.request.Request(dl)
                    resp2 = opener.open(req2, timeout=60)
                    with open(path, "wb") as f:
                        f.write(resp2.read())
                    sha = sha256_file(path)
                    actual = os.path.getsize(path)
                    print(f" OK ({actual})")
                except Exception as e:
                    print(f" FAILED: {e}")
                    sha = None
                time.sleep(0.3)

            entries.append({
                "file": name, "size": os.path.getsize(path),
                "sha256": sha, "url": dl, "project": project_id,
            })

        url = data.get("links", {}).get("next")
        if url:
            time.sleep(0.5)

    return entries


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Downloading eLife summary files (osf.io/h4qr6)...")
    elife_entries = download_osf_project("h4qr6", OUT_DIR, "eLife")

    print(f"\nDownloading trial-level files (osf.io/7wuh4)...")
    trial_entries = download_osf_project("7wuh4", OUT_DIR, "trial")

    all_entries = elife_entries + trial_entries
    manifest = {
        "dataset": "Eckstein et al. 2022",
        "sources": [
            {"project": "7wuh4", "doi": "10.1016/j.dcn.2022.101106",
             "description": "Trial-level data (DevCogNeurosci)"},
            {"project": "h4qr6", "doi": "10.7554/eLife.75474",
             "description": "Summary + demographics (eLife)"},
        ],
        "license": "CC-By Attribution 4.0 International",
        "citation": (
            "Eckstein, M.K. et al. Reinforcement learning and Bayesian "
            "inference provide complementary models for the unique advantage "
            "of adolescents in stochastic reversal. Dev Cogn Neurosci 55, "
            "101106 (2022)."),
        "date_fetched": time.strftime("%Y-%m-%d"),
        "n_files": len(all_entries),
        "files": all_entries,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Total files: {len(all_entries)}")


if __name__ == "__main__":
    main()
