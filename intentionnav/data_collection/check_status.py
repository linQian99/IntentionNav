"""Pre-flight status check for Phase 2 annotation.

For each scene with a capture_manifest.json, compares the expected set of
(surface_id, target_category) pairs against what's actually present in
intent_annotations.json. Reports scene-level and entry-level status.

Usage:
    python -m intentionnav.data_collection.check_status                 # summary
    python -m intentionnav.data_collection.check_status --per-scene     # list every scene's state
    python -m intentionnav.data_collection.check_status --only-missing  # only list scenes with gaps
    python -m intentionnav.data_collection.check_status --json          # machine-readable
    python -m intentionnav.data_collection.check_status --capture-dir <path> --output-dir <path>
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def check_scenes(capture_dir, output_dir):
    """Returns list of per-scene status dicts."""
    if not os.path.isdir(capture_dir):
        return []
    scenes = sorted(d for d in os.listdir(capture_dir)
                    if d.startswith("kujiale_") and
                    os.path.isfile(os.path.join(capture_dir, d, "capture_manifest.json")))

    report = []
    for sid in scenes:
        manifest = _load_json(os.path.join(capture_dir, sid, "capture_manifest.json")) or []
        expected = {(e["surface_id"], e["target_category"]) for e in manifest}
        ann_path = os.path.join(output_dir, sid, "intent_annotations.json")
        annotations = _load_json(ann_path) or []
        done = {(e["surface_id"], e["target_category"]) for e in annotations}
        missing = expected - done
        extra = done - expected  # annotated but no longer in manifest (shouldn't happen)

        if not expected:
            status = "empty-manifest"
        elif not done:
            status = "untouched"
        elif missing:
            status = "partial"
        elif extra:
            status = "extra"
        else:
            status = "complete"

        report.append({
            "scene_id": sid,
            "status": status,
            "expected": len(expected),
            "done": len(done),
            "missing": len(missing),
            "extra": len(extra),
            "missing_items": sorted(missing),
            "extra_items": sorted(extra),
        })
    return report


def summarize(report):
    total_scenes = len(report)
    by_status = defaultdict(int)
    total_expected = total_done = total_missing = 0
    for r in report:
        by_status[r["status"]] += 1
        total_expected += r["expected"]
        total_done += r["done"]
        total_missing += r["missing"]
    return {
        "total_scenes": total_scenes,
        "scenes_by_status": dict(by_status),
        "total_expected_entries": total_expected,
        "total_done_entries": total_done,
        "total_missing_entries": total_missing,
        "completion_pct": (100 * total_done / total_expected) if total_expected else 0.0,
    }


def _fmt_pct(pct):
    return f"{pct:5.1f}%"


def print_summary(summary):
    print(f"Scenes:      {summary['total_scenes']}")
    print(f"  complete:  {summary['scenes_by_status'].get('complete', 0)}")
    print(f"  partial:   {summary['scenes_by_status'].get('partial', 0)}")
    print(f"  untouched: {summary['scenes_by_status'].get('untouched', 0)}")
    print(f"  extra:     {summary['scenes_by_status'].get('extra', 0)}")
    print(f"  other:     {summary['scenes_by_status'].get('empty-manifest', 0)}")
    print(f"Entries:     {summary['total_done_entries']} / {summary['total_expected_entries']}  "
          f"({_fmt_pct(summary['completion_pct'])}  "
          f"{summary['total_missing_entries']} missing)")


def print_per_scene(report, only_missing=False):
    col_status = {"complete": "✓", "partial": "…", "untouched": "·", "extra": "!",
                  "empty-manifest": "?"}
    for r in report:
        if only_missing and r["status"] == "complete":
            continue
        mark = col_status.get(r["status"], "?")
        print(f"  {mark} {r['scene_id']:18s}  {r['done']:>3d}/{r['expected']:<3d}  "
              f"[{r['status']}]")
        if r["missing"]:
            for sid, cat in r["missing_items"]:
                print(f"      - missing: {sid} → {cat}")
        if r["extra"]:
            for sid, cat in r["extra_items"]:
                print(f"      - extra (not in manifest): {sid} → {cat}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir",
                    default="work_dirs",
                    help="Phase 1 output root (contains kujiale_*/capture_manifest.json)")
    ap.add_argument("--output-dir",
                    default="work_dir_phase2",
                    help="Phase 2 output root (contains kujiale_*/intent_annotations.json)")
    ap.add_argument("--per-scene", action="store_true", help="list every scene")
    ap.add_argument("--only-missing", action="store_true",
                    help="implies --per-scene; hide fully-complete scenes")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = ap.parse_args()

    report = check_scenes(args.capture_dir, args.output_dir)
    summary = summarize(report)

    if args.json:
        print(json.dumps({"summary": summary, "per_scene": report},
                         ensure_ascii=False, indent=2))
        return

    print("=" * 60)
    print("IntentionNav Phase 2 status")
    print("=" * 60)
    print_summary(summary)
    if args.per_scene or args.only_missing:
        print()
        print("Per-scene breakdown:")
        print_per_scene(report, only_missing=args.only_missing)

    # Exit code 0 = all complete, 1 = gaps exist, useful in shell pipelines
    sys.exit(0 if summary["total_missing_entries"] == 0 else 1)


if __name__ == "__main__":
    main()
