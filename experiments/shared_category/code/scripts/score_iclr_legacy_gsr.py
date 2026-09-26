"""Supplement completed ICLR trajectories with the original paper's GSR@1m.

CPU only. Replay the immutable historical scorer, validate all 6,000 historical
labels, then score all 2,000 completed category-input trajectories. Original
navigation records, evidence reports, and metric sources remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import types

from analyze_iclr_paper_evidence import estimate


REPO = Path(__file__).resolve().parents[1]
REVISION = "c3c9ee08438bb445d49a0ee6e8f553c54db44bca"
CORE = REPO / "results/iclr_6day_core_20260919"
OLD_DATA = REPO / "results/dataset"
NEW_DATA = REPO / "results/dataset_ai_reviewed_20260918_v3"
SCENES = Path("/path/to/workspace/datasets/vlntube/SceneSummary/kujiale_scene_summary")
ARMS = ("r055_implicit_category/a", "r055_explicit/a",
        "mtu3d_implicit_category/a", "mtu3d_explicit/a")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pins = {}

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        pins[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    source = subprocess.check_output(
        ["git", "show", REVISION + ":eval/aggregate/visibility.py"], cwd=REPO)
    source_path = args.out / "legacy_visibility.py"
    source_path.write_bytes(source)
    vis = types.ModuleType("legacy_visibility")
    vis.__file__ = str(source_path.resolve())
    exec(compile(source, str(source_path), "exec"), vis.__dict__)
    os.environ["INTENTEQA_SCENE_SUMMARY"] = str(SCENES)

    def configure(root):
        vis.DATASET_ROOT = root
        vis.DATASET_JSONL = root / "selected_500_intents.jsonl"
        vis._items_by_selection_id.cache_clear()
        vis._scene_manifest.cache_clear()
        items = [json.loads(line) for line in vis.DATASET_JSONL.read_text().splitlines()]
        pins[str(vis.DATASET_JSONL)] = digest(vis.DATASET_JSONL)
        assert len(items) == len({item["selection_id"] for item in items}) == 500
        for scene in {item["scene_id"] for item in items}:
            read(root / scene / "manifest.json")
            objects = read(SCENES / scene / "object_dict.json")
            assert objects and vis._scene_objects(scene), scene
        return {item["selection_id"]: item for item in items}

    def score(record):
        trajectory = record["trajectory"]
        poses = [step for step in trajectory if step.get("position")]
        last = poses[-1]
        assert last is trajectory[-1] and last.get("yaw") is not None
        pos, target = last["position"], record["episode_meta"]["target_position"]
        assert len(pos) == len(target) == 3
        assert all(math.isfinite(x) for x in [*pos, *target, last["yaw"]])
        entry = vis._target_entry(record)
        assert entry and len(entry["target_details"]) == 1
        assert entry["target_details"][0]["object_id"] == record["episode_meta"]["target_object_id"]
        bmin, bmax = vis._combined_target_bbox(record)
        assert bmin and bmax and all(math.isfinite(x) for x in [*bmin, *bmax])
        assert all(lo < hi for lo, hi in zip(bmin, bmax))
        d = math.dist(pos[:2], target[:2])
        v = vis.trajectory_visibility(record)
        assert v, record["selection_id"]
        return dict(center_final_m=d, paper_SR_1m=int(d <= 1.0),
                    paper_GSR_1m=int(d <= 1.0 and v["G_seen"]), **v)

    configure(OLD_DATA)
    old_csv = REPO / "results/final_6000_metrics_20260507/strict_r1m/per_item.csv"
    pins[str(old_csv)] = digest(old_csv)
    old = {(r["model"], r["style"], r["selection_id"]): r
           for r in csv.DictReader(old_csv.open())}
    assert len(old) == 6000
    calibration, seen = {}, set()
    for path in sorted((REPO / "results/final_6000_merged_20260507").rglob("record.json")):
        record = read(path)
        key = record["model"], record["style"], record["selection_id"]
        assert key not in seen
        seen.add(key)
        value = score(record)
        assert value["paper_SR_1m"] == int(old[key]["SR_hit"]), key
        assert value["paper_GSR_1m"] == int(old[key]["GSR_hit"]), key
        counts = calibration.setdefault(key[0], dict(n=0, SR_1m_count=0, GSR_1m_count=0))
        counts["n"] += 1
        counts["SR_1m_count"] += value["paper_SR_1m"]
        counts["GSR_1m_count"] += value["paper_GSR_1m"]
    assert seen == set(old)
    print("Historical calibration passed: 6000/6000 exact labels", flush=True)

    items = configure(NEW_DATA)
    ep_path = NEW_DATA / "episodes_explicit_category.jsonl"
    pins[str(ep_path)] = digest(ep_path)
    episodes = {e["selection_id"]: e for e in map(json.loads, ep_path.read_text().splitlines())}
    evidence_path = CORE / "paper_evidence_20260920/full500/evidence.json"
    evidence = read(evidence_path)
    assert evidence["passed"]
    for path, expected in evidence["source_hashes"].items():
        assert digest(path) == expected, path
    csv_path = evidence_path.parent / "episodes.csv"
    pins[str(csv_path)] = digest(csv_path)
    rows, by_arm = [], {arm: {} for arm in ARMS}
    for row in csv.DictReader(csv_path.open()):
        record = read(row["record_path"])
        sid, arm = row["selection_id"], row["arm_repeat"]
        assert sid == record["selection_id"] and sid not in by_arm[arm]
        ep = episodes[sid]
        for field in ("scene_id", "target_object_id", "target_position", "start_position"):
            assert record["episode_meta"][field] == ep[field], (arm, sid, field)
        assert items[sid]["target_representative"] == ep["target_object_id"]
        assert record["trajectory"][0]["position"] == ep["start_position"]
        value = score(record)
        assert value["paper_SR_1m"] == int(row["paper_SR_1m"])
        assert abs(value["center_final_m"] - float(row["center_final_m"])) < 1e-10
        result = {k: row[k] for k in ("arm_repeat", "selection_id", "scene_id",
                                      "target_object_id", "record_path")}
        result.update(value)
        rows.append(result)
        by_arm[arm][sid] = result
    assert len(rows) == 2000
    table = {}
    for arm, arm_rows in by_arm.items():
        assert set(arm_rows) == set(items)
        ordered = [arm_rows[sid] for sid in sorted(items)]
        stat = estimate([r["paper_GSR_1m"] for r in ordered], ordered)
        stat["success_count"] = sum(r["paper_GSR_1m"] for r in ordered)
        stat["SR_1m_count"] = sum(r["paper_SR_1m"] for r in ordered)
        stat["success_ids"] = [r["selection_id"] for r in ordered if r["paper_GSR_1m"]]
        table[arm] = stat
    output_csv = args.out / "episodes.csv"
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pins[str(Path(__file__).resolve())] = digest(__file__)
    for name in ("analyze_iclr_paper_evidence.py", "score_iclr_core_comparison.py"):
        path = REPO / "scripts" / name
        pins[str(path)] = digest(path)
    assert all(digest(path) == value for path, value in pins.items())
    report = dict(passed=True, created_at=datetime.now(timezone.utc).isoformat(),
                  definition="GSR@1m = final target-center XY distance <= 1m AND legacy final-pose geometric visibility",
                  visibility=dict(revision=REVISION, source_sha256=digest(source_path),
                                  projection_size=1024, min_effective_visible_px=1.0,
                                  max_occlusion_ratio=0.95, visibility_radius_m=3.0,
                                  scene_summary=str(SCENES), missing_geometry=0,
                                  explicit_stop_required=False),
                  calibration=dict(records=6000, per_item_mismatches=0, by_model=calibration),
                  main_table=table, records=2000, source_hashes=pins)
    write_json(args.out / "gsr_evidence.json", report)
    # Publish a supplemented copy, preserving the original evidence artifact.
    evidence["definitions"]["GSR"] = report["definition"]
    for arm in ARMS:
        evidence["main_table"][arm]["GSR"] = table[arm]
    evidence["GSR_supplement"] = dict(path=str((args.out / "gsr_evidence.json").resolve()),
                                       sha256=digest(args.out / "gsr_evidence.json"))
    write_json(args.out / "evidence_with_gsr.json", evidence)
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
