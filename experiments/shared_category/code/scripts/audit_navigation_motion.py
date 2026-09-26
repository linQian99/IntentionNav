"""Audit full MOVE segments for either navigation system before accepting scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval/agents"))
sys.path.insert(0, str(ROOT / "eval/simulator"))
from navigation_motion import audit_navigation_motion
from walkable_map import WalkableMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--metaroot", required=True, type=Path)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--maximum-m", type=float, default=1.7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.run.glob("episodes/**/record.json"))
    if (args.expected_records < 1 or len(paths) != args.expected_records
            or not (args.run / ".RUN_SUCCESS").is_file()):
        raise ValueError("Require the exact completed run, including its completion marker")
    import math
    if not math.isfinite(args.maximum_m) or args.maximum_m <= 0:
        raise ValueError("Invalid MOVE budget")
    maps, seen, results, pins = {}, set(), [], {}

    def pin(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = str(path.resolve())
        if key in pins and pins[key] != digest:
            raise ValueError(f"Input changed during audit: {path}")
        pins[key] = digest

    for path in [Path(__file__), ROOT / "eval/agents/navigation_motion.py",
                 ROOT / "eval/agents/mtu3d_strict_grid.py", ROOT / "eval/simulator/walkable_map.py",
                 args.run / ".RUN_SUCCESS"]:
        pin(path)
    for path in paths:
        pin(path)
        record = json.loads(path.read_text())
        identity = (record["scene_id"], record["selection_id"], record["style"])
        if identity in seen:
            raise ValueError(f"Duplicate episode identity: {identity}")
        seen.add(identity)
        scene = record["scene_id"]
        if scene not in maps:
            pin(args.metaroot / scene / "freemap.npy")
            maps[scene] = WalkableMap.load(scene, metaroot=args.metaroot)
            if maps[scene] is None:
                raise FileNotFoundError(f"Missing map for {scene}")
        try:
            result = audit_navigation_motion(record, maps[scene], maximum=args.maximum_m)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            result = {"passed": False, "error": str(exc)}
        results.append({"record": str(path.resolve()), "selection_id": record["selection_id"], **result})
        pin(path)
    report = {"passed": all(r["passed"] for r in results), "records": len(results),
              "failed_records": sum(not r["passed"] for r in results),
              "scope": "Grid-motion integrity only; not perception audit or performance acceptance",
              "maximum_m": args.maximum_m, "results": results, "source_hashes": pins,
              "checked_at": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"results", "source_hashes"}}))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
