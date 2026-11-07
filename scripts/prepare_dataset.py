#!/usr/bin/env python3
"""
Build ``strawberry_open_v13`` for ImageFolder training (``data/processed/strawberry``).

Modes
-----
copy_layout
    Source already has ``train/``, ``val/``, ``test/`` with 14 class subfolders.

exact
    Pooled images per class under ``data/raw/pooled/<class_name>/``;
    split with fixed per-class counts from ``configs/dataset/strawberry_open_v13_expected_splits.json``.

manifest
    CSV with columns ``path,class,split`` (paths relative to ``--manifest-root``).

collect_sources
    Walk ``source_mapping.json`` (see ``configs/dataset/source_mapping.example.json``),
    copy into ``data/raw/pooled/<target_class>/``, then run ``--mode exact``.

Examples
--------
  python scripts/prepare_dataset.py --mode copy_layout --src data/raw/strawberry_open_v13
  python scripts/prepare_dataset.py --mode collect_sources --mapping data/raw/source_mapping.json
  python scripts/prepare_dataset.py --mode exact --pooled-root data/raw/pooled --seed 42 --verify
  python scripts/prepare_dataset.py --mode manifest --manifest data/splits/manifest.csv --verify
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ag_convnext import STRAWBERRY_14_CLASSES  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_OUT = ROOT / "data" / "processed" / "strawberry"
DEFAULT_SPLITS = ROOT / "configs" / "dataset" / "strawberry_open_v13_expected_splits.json"


def load_expected_splits(path: Path) -> dict[str, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["classes"]


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


def ensure_empty_split_tree(out_dir: Path, classes: Iterable[str], splits: Iterable[str]) -> None:
    for split in splits:
        for cls in classes:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def copy_class_split_files(
    files: list[Path],
    cls: str,
    split: str,
    out_dir: Path,
    use_symlink: bool,
    name_prefix: str = "",
) -> int:
    n = 0
    for i, src in enumerate(files):
        dst_name = f"{name_prefix}{src.stem}_{i:05d}{src.suffix.lower()}"
        link_or_copy(src, out_dir / split / cls / dst_name, use_symlink)
        n += 1
    return n


def split_exact(
    pooled_root: Path,
    out_dir: Path,
    expected: dict[str, dict[str, int]],
    seed: int,
    use_symlink: bool,
) -> dict[str, dict[str, int]]:
    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cls in STRAWBERRY_14_CLASSES:
        if cls not in expected:
            raise KeyError(f"Missing class in split config: {cls}")
        src_dir = pooled_root / cls
        images = list_images(src_dir)
        spec = expected[cls]
        need = spec["train"] + spec["val"] + spec["test"]
        if len(images) < need:
            raise ValueError(
                f"{cls}: need {need} images in {src_dir}, found {len(images)}. "
                "Add sources or run collect_sources first."
            )
        if len(images) > need:
            print(f"Warning: {cls} has {len(images)} files; using first {need} after shuffle (seed={seed}).")
        rng.shuffle(images)
        images = images[:need]

        n_tr, n_va, n_te = spec["train"], spec["val"], spec["test"]
        tr, va, te = images[:n_tr], images[n_tr : n_tr + n_va], images[n_tr + n_va :]

        counts[cls]["train"] = copy_class_split_files(tr, cls, "train", out_dir, use_symlink, f"{cls}_")
        counts[cls]["val"] = copy_class_split_files(va, cls, "val", out_dir, use_symlink, f"{cls}_")
        counts[cls]["test"] = copy_class_split_files(te, cls, "test", out_dir, use_symlink, f"{cls}_")

    return counts


def mode_copy_layout(src: Path, out_dir: Path, use_symlink: bool) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split in ("train", "val", "test"):
        split_src = src / split
        if not split_src.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_src}")
        for cls_dir in sorted(split_src.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            images = list_images(cls_dir)
            copy_class_split_files(images, cls, split, out_dir, use_symlink)
            counts[cls][split] = len(images)
    return counts


def mode_manifest(
    manifest: Path,
    manifest_root: Path,
    out_dir: Path,
    use_symlink: bool,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with manifest.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "class", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest CSV must contain columns: {sorted(required)}")
        for row in reader:
            cls = row["class"].strip()
            split = row["split"].strip()
            if split not in ("train", "val", "test"):
                raise ValueError(f"Invalid split '{split}' for {row['path']}")
            if cls not in STRAWBERRY_14_CLASSES:
                raise ValueError(f"Unknown class '{cls}' (not in STRAWBERRY_14_CLASSES)")
            src = (manifest_root / row["path"].strip()).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"Manifest image not found: {src}")
            dst = out_dir / split / cls / f"{src.stem}_{counts[cls][split]:05d}{src.suffix.lower()}"
            link_or_copy(src, dst, use_symlink)
            counts[cls][split] += 1
    return counts


def collect_sources(mapping_path: Path, pooled_root: Path) -> dict[str, int]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    per_class: dict[str, int] = defaultdict(int)

    for source in mapping.get("sources", []):
        root = Path(source["root"])
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        class_map: dict[str, str] = source["class_map"]
        if not root.is_dir():
            print(f"Skip missing source root: {root}")
            continue
        for src_label, tgt_cls in class_map.items():
            if tgt_cls not in STRAWBERRY_14_CLASSES:
                raise ValueError(f"Bad target class {tgt_cls} in mapping")
            candidates = [root / src_label, root / src_label.replace(" ", "_")]
            src_dirs = [d for d in candidates if d.is_dir()]
            if not src_dirs:
                print(f"Warning: no folder for '{src_label}' under {root}")
                continue
            dst_dir = pooled_root / tgt_cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src_dir in src_dirs:
                for img in list_images(src_dir):
                    dst = dst_dir / f"{source['name']}_{img.stem}_{per_class[tgt_cls]:05d}{img.suffix.lower()}"
                    shutil.copy2(img, dst)
                    per_class[tgt_cls] += 1
    return per_class


def verify_counts(counts: dict[str, dict[str, int]], expected_path: Path) -> bool:
    expected = load_expected_splits(expected_path)
    ok = True
    for cls in STRAWBERRY_14_CLASSES:
        spec = expected[cls]
        got = counts.get(cls, {})
        for split in ("train", "val", "test"):
            g = got.get(split, 0)
            e = spec[split]
            if g != e:
                print(f"MISMATCH {cls}/{split}: expected {e}, got {g}")
                ok = False
    if ok:
        print("Split counts match strawberry_open_v13 expected table.")
    return ok


def print_summary(counts: dict[str, dict[str, int]]) -> None:
    totals = {"train": 0, "val": 0, "test": 0}
    for cls in STRAWBERRY_14_CLASSES:
        c = counts.get(cls, {})
        print(
            f"  {cls:26s}  train={c.get('train', 0):4d}  "
            f"val={c.get('val', 0):3d}  test={c.get('test', 0):4d}"
        )
        for k in totals:
            totals[k] += c.get(k, 0)
    print(f"TOTAL  train={totals['train']}  val={totals['val']}  test={totals['test']}  all={sum(totals.values())}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare strawberry_open_v13 ImageFolder layout")
    p.add_argument(
        "--mode",
        choices=("copy_layout", "exact", "manifest", "collect_sources"),
        required=True,
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--src", type=Path, help="For copy_layout: root with train/val/test")
    p.add_argument("--pooled-root", type=Path, default=ROOT / "data" / "raw" / "pooled")
    p.add_argument("--manifest", type=Path, help="CSV path,class,split")
    p.add_argument("--manifest-root", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--mapping", type=Path, help="source_mapping.json for collect_sources")
    p.add_argument("--splits-config", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symlink", action="store_true", help="Symlink instead of copy (default: copy)")
    p.add_argument("--verify", action="store_true", help="Compare counts to paper table")
    p.add_argument("--clean", action="store_true", help="Remove existing out-dir train/val/test first")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)

    if args.clean and out_dir.exists():
        for split in ("train", "val", "test"):
            split_dir = out_dir / split
            if split_dir.is_dir():
                shutil.rmtree(split_dir)

    ensure_empty_split_tree(out_dir, STRAWBERRY_14_CLASSES, ("train", "val", "test"))

    if args.mode == "collect_sources":
        if not args.mapping:
            raise SystemExit("--mapping required for collect_sources")
        mapping = args.mapping if args.mapping.is_absolute() else (ROOT / args.mapping)
        pooled = args.pooled_root if args.pooled_root.is_absolute() else (ROOT / args.pooled_root)
        collected = collect_sources(mapping, pooled)
        print(f"Collected into {pooled}:")
        for cls in STRAWBERRY_14_CLASSES:
            print(f"  {cls}: {collected.get(cls, 0)}")
        print("Run next: python scripts/prepare_dataset.py --mode exact --verify")
        return

    if args.mode == "copy_layout":
        if not args.src:
            raise SystemExit("--src required for copy_layout")
        src = args.src if args.src.is_absolute() else (ROOT / args.src)
        counts = mode_copy_layout(src, out_dir, args.symlink)
    elif args.mode == "exact":
        pooled = args.pooled_root if args.pooled_root.is_absolute() else (ROOT / args.pooled_root)
        expected = load_expected_splits(args.splits_config)
        counts = split_exact(pooled, out_dir, expected, args.seed, args.symlink)
    elif args.mode == "manifest":
        if not args.manifest:
            raise SystemExit("--manifest required for manifest mode")
        manifest = args.manifest if args.manifest.is_absolute() else (ROOT / args.manifest)
        root = args.manifest_root if args.manifest_root.is_absolute() else (ROOT / args.manifest_root)
        counts = mode_manifest(manifest, root, out_dir, args.symlink)
    else:
        raise SystemExit(f"Unhandled mode: {args.mode}")

    print(f"\nDataset written to: {out_dir}")
    print_summary(counts)
    if args.verify:
        verify_counts(counts, args.splits_config)


if __name__ == "__main__":
    main()
