"""Verify numbers.json against results/ files.

For each entry: loads source file, extracts value at path, compares.
Mismatch or missing → explicit error. Rule 1 enforcement.

Usage: python article/verify_numbers.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
NUMBERS = ROOT / "article" / "numbers.json"


def extract(data, path):
    """Extract value from nested dict/list by dot-separated path."""
    parts = path.split(".")
    v = data
    for p in parts:
        if isinstance(v, list):
            v = v[int(p)]
        elif isinstance(v, dict):
            # Try int key first (for stringified int keys in JSON)
            if p not in v and p.isdigit():
                v = v[int(p)]
            else:
                v = v[p]
        else:
            raise KeyError(f"Cannot traverse into {type(v)} at '{p}'")
    return v


def verify():
    with open(NUMBERS) as f:
        numbers = json.load(f)

    n_ok = 0
    n_fail = 0
    failures = []

    for name, entry in numbers.items():
        source = entry["source"]
        path = entry["path"]
        expected = entry["value"]

        source_path = RESULTS / source
        if not source_path.exists():
            failures.append(f"{name}: source {source} NOT FOUND")
            n_fail += 1
            continue

        with open(source_path) as f:
            data = json.load(f)

        # Special case: absent_key means value=0 iff key missing from dict
        if "absent_key" in entry:
            try:
                container = extract(data, path)
                absent = entry["absent_key"]
                if absent in container:
                    failures.append(
                        f"{name}: key '{absent}' should be absent but "
                        f"found with value {container[absent]}")
                    n_fail += 1
                else:
                    n_ok += 1
                continue
            except (KeyError, IndexError, TypeError) as e:
                failures.append(f"{name}: path '{path}' in {source}: {e}")
                n_fail += 1
                continue

        try:
            actual = extract(data, path)
        except (KeyError, IndexError, TypeError) as e:
            failures.append(f"{name}: path '{path}' in {source}: {e}")
            n_fail += 1
            continue

        # Compare with tolerance
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if abs(expected - actual) > max(abs(expected) * 1e-6, 1e-10):
                failures.append(
                    f"{name}: expected {expected}, got {actual} "
                    f"(diff {actual - expected})")
                n_fail += 1
                continue
        elif expected != actual:
            failures.append(
                f"{name}: expected {expected!r}, got {actual!r}")
            n_fail += 1
            continue

        n_ok += 1

    print(f"Verified: {n_ok}/{n_ok + n_fail}")
    if failures:
        print(f"\nFAILURES ({n_fail}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All numbers verified against source files.")


if __name__ == "__main__":
    verify()
