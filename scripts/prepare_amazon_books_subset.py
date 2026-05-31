#!/usr/bin/env python3
"""Prepare the local Amazon Books subset for profile-before-ranking experiments."""

from __future__ import annotations

import argparse
import ast
import gzip
import html
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("/root/autodl-tmp/amazon-books-subset"))
    parser.add_argument("--full-root", type=Path, default=Path("/root/autodl-tmp/amazon-books-processed"))
    parser.add_argument("--raw-meta", type=Path, default=Path("/root/autodl-tmp/amazon-books-raw/meta_Books.json.gz"))
    parser.add_argument("--outdir", type=Path, default=ROOT / "data/amazon_books_subset")
    parser.add_argument("--progress-every", type=int, default=500000)
    return parser.parse_args()


def normalize_split(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(columns={"user_idx": "uid", "item_idx": "iid"})[["uid", "iid", "timestamp", "rating"]].copy()
    out["uid"] = out["uid"].astype("int64")
    out["iid"] = out["iid"].astype("int64")
    out["timestamp"] = out["timestamp"].astype("int64")
    out["rating"] = out["rating"].astype("float32")
    return out.sort_values(["timestamp", "uid", "iid"], kind="mergesort").reset_index(drop=True)


def flatten_categories(value: object) -> List[str]:
    labels: List[str] = []
    if isinstance(value, list):
        for path in value:
            if isinstance(path, list):
                for entry in path:
                    text = str(entry).strip()
                    if text and text not in labels:
                        labels.append(text)
            else:
                text = str(path).strip()
                if text and text not in labels:
                    labels.append(text)
    return labels


def clean_text(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = html.unescape(str(value)).replace("\n", " ").strip()
    return " ".join(text.split()) or fallback


def load_needed_asins(subset_mappings: Mapping[str, Mapping[object, object]], full_mappings: Mapping[str, Mapping[object, object]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    full_id2item = full_mappings["id2item"]
    for subset_iid, full_iid in subset_mappings["id2item"].items():
        asin = full_id2item.get(int(full_iid))
        if asin is not None:
            out[int(subset_iid)] = str(asin)
    return out


def parse_metadata(raw_meta: Path, iid_to_asin: Mapping[int, str], progress_every: int) -> Dict[str, Dict[str, object]]:
    needed = set(iid_to_asin.values())
    found: Dict[str, Dict[str, object]] = {}
    with gzip.open(raw_meta, "rt", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no % progress_every == 0:
                print(f"metadata scan lines={line_no:,} found={len(found):,}/{len(needed):,}", flush=True)
            try:
                row = ast.literal_eval(line)
            except Exception:
                continue
            asin = str(row.get("asin", ""))
            if asin not in needed or asin in found:
                continue
            title = clean_text(row.get("title"), fallback="Unknown Book")
            description = row.get("description", "")
            if isinstance(description, list):
                description = " ".join(str(x) for x in description)
            found[asin] = {
                "asin": asin,
                "title": title,
                "categories": flatten_categories(row.get("categories", [])),
                "description": clean_text(description)[:1000],
            }
            if len(found) == len(needed):
                break
    print(f"metadata filled for {len(found):,}/{len(needed):,} asins", flush=True)
    return found


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    train = normalize_split(pickle.load(open(args.subset_root / "train.pkl", "rb")))
    val = normalize_split(pickle.load(open(args.subset_root / "val.pkl", "rb")))
    test = normalize_split(pickle.load(open(args.subset_root / "test.pkl", "rb")))
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        frame.to_csv(args.outdir / f"{name}.csv", index=False)
    all_events = pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ], ignore_index=True).sort_values(["timestamp", "uid", "iid"], kind="mergesort")
    all_events.to_csv(args.outdir / "all_events_log_observable.csv", index=False)

    subset_mappings = pickle.load(open(args.subset_root / "mappings.pkl", "rb"))
    full_mappings = pickle.load(open(args.full_root / "mappings.pkl", "rb"))
    iid_to_asin = load_needed_asins(subset_mappings, full_mappings)
    meta_by_asin = parse_metadata(args.raw_meta, iid_to_asin, args.progress_every)

    metadata_rows = []
    item_texts = {}
    item_categories = {}
    for iid in range(max(iid_to_asin) + 1):
        asin = iid_to_asin.get(iid, "")
        row = meta_by_asin.get(asin, {"asin": asin, "title": "Unknown Book", "categories": [], "description": ""})
        categories = [str(x) for x in row.get("categories", []) if str(x).strip()]
        title = clean_text(row.get("title"), fallback="Unknown Book")
        description = clean_text(row.get("description", ""))
        text = title
        if categories:
            text += " | " + ", ".join(categories[:6])
        if description:
            text += " | " + description[:300]
        item_texts[iid] = text
        item_categories[iid] = categories
        metadata_rows.append({"iid": iid, "asin": asin, "title": title, "categories": categories, "description": description})
    with (args.outdir / "item_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    mappings = {
        "uid_to_raw": {str(int(k)): str(v) for k, v in subset_mappings["id2user"].items()},
        "iid_to_asin": {str(k): v for k, v in iid_to_asin.items()},
    }
    (args.outdir / "mappings.json").write_text(json.dumps(mappings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stats = {
        "users": int(max(train["uid"].max(), val["uid"].max(), test["uid"].max()) + 1),
        "items": int(max(train["iid"].max(), val["iid"].max(), test["iid"].max()) + 1),
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
        "known_titles": int(sum(1 for row in metadata_rows if row["title"] != "Unknown Book")),
        "nonempty_categories": int(sum(1 for row in metadata_rows if row["categories"])),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
