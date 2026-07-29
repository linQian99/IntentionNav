"""Aggregate per-episode judged outputs into headline metrics.

ACTIVE tiers (vlm / vlm_engine / fbe / random / oracle / blind): produces
IM / SR / OSR / SPL / G + per-style and per-intent_mode breakdowns.
See `report/make_tables.py` for the 5-table paper layout.

Scans `eval_out/<tier>/<model>/<style>/*.json` and produces:
  - results/<tag>/metrics.json      — all aggregate numbers
  - results/<tag>/per_item.csv      — per-(tier,model,style,item) scores
  - prints a summary table to stdout

Usage:
  python aggregate/compute_metrics.py --out results/current
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "agents"))
from common import EVAL_DIR, EPISODES_OUT, STYLES, load_items
from visibility import trajectory_visibility


# ---- Category normalization / synonym match ----
def _load_synonyms() -> dict[str, set[str]]:
    """Load category synonym dict for intent-match scoring."""
    try:
        import yaml
    except ImportError:
        return {}
    path = EVAL_DIR / "vocab/category_synonyms.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for cat, aliases in raw.items():
        norm_cat = _norm(cat)
        s = {norm_cat}
        for a in (aliases or []):
            s.add(_norm(a))
        out[norm_cat] = s
    return out


def _norm(s: str) -> str:
    """Normalize string for IM matching: lower + snake + strip articles."""
    s = (s or "").lower().strip()
    s = re.sub(r"[\s_\-]+", "_", s)
    s = re.sub(r"^(a_|an_|the_)", "", s)
    return s


def intent_match(pred: str, target: str, synonyms: dict[str, set[str]]) -> bool:
    """IM: did LLM's plan target_guess match the dataset target_category?

    Measures intent comprehension separately from navigation success.
    Uses synonym dict for semantic equivalence (bed↔bedframe, basin↔sink).
    """
    p, t = _norm(pred), _norm(target)
    if not p or not t:
        return False
    if p == t:
        return True
    if t in synonyms and p in synonyms[t]:
        return True
    if p in synonyms and t in synonyms[p]:
        return True
    return False


# ---- Episode loading ----
def iter_episode_files(tier_dir: Path):
    for p in sorted(tier_dir.rglob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            yield p, json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] skip unreadable {p}: {e}", file=sys.stderr)


def nav_metrics(rec: dict, radius_m: float = 2.0) -> dict:
    """Compute geometric d_final / d_min / OSR_hit / path_length from an
    active-tier record. SR_hit is computed in `aggregate()` after combining
    geometric proximity with G_hit (visibility). Returns {} if no trajectory.

    Note: σ_T (LLM-judge target match) was removed from SR. Visual
    confirmation of "agent stopped facing target" is now handled by G
    (geometric occlusion-aware projection of target bbox onto final-frame
    camera) — deterministic, no LLM call, doesn't truncate.
    """
    import math
    traj = rec.get("trajectory") or []
    meta = rec.get("episode_meta") or {}
    target_xy = meta.get("target_position")
    if not traj:
        return {}
    if target_xy is None or len(target_xy) < 2:
        nav_metrics._dropped = getattr(nav_metrics, "_dropped", 0) + 1
        return {}
    target = (float(target_xy[0]), float(target_xy[1]))

    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    positions = []
    for t in traj:
        p = t.get("position")
        if p and len(p) >= 2:
            positions.append((float(p[0]), float(p[1])))
    if not positions:
        return {}

    d_final = d(positions[-1], target)
    d_min = min(d(p, target) for p in positions)
    path_len = sum(d(positions[i], positions[i - 1]) for i in range(1, len(positions)))
    geo_len = float(meta.get("geodesic_to_target") or 0.0)

    SR_hit = int(d_final <= radius_m)  # SR = geometric reach (d_final ≤ 2m)
    OSR_hit = int(d_min <= radius_m)
    SPL = (SR_hit * geo_len / max(path_len, geo_len)) if geo_len > 0 else 0.0

    return {
        "d_final": round(d_final, 3),
        "d_min": round(d_min, 3),
        "path_length": round(path_len, 3),
        "geodesic_length": round(geo_len, 3),
        "SR_hit": SR_hit,           # geometric success (d_final ≤ 2m)
        "OSR_hit": OSR_hit,
        "SPL": round(SPL, 4),
        "steps": len(traj) - 1,
        # GSR_hit = SR_hit AND G_hit, filled in aggregate() (needs visibility)
    }


def collect_records(split_ids: set[str]):
    """Yield one (tier, model, style, record) per judged episode.

    New layout: results/eval_out/<scene>/<model>/<tier>_<style>_<sel>.json
    We just walk all .json files and read tier/model/style/selection_id
    from inside the record (which is authoritative).
    """
    if not EPISODES_OUT.exists():
        return
    for p in sorted(EPISODES_OUT.rglob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] skip unreadable {p}: {e}", file=sys.stderr)
            continue
        if rec.get("selection_id") not in split_ids:
            continue
        # judge no longer required (SR uses geometric d + G visibility, no σ_T).
        tier = rec.get("tier", "")
        model = rec.get("model", "")
        style = rec.get("style", "")
        if style not in STYLES:
            continue
        yield tier, model, style, rec


# ---- Bootstrap helpers ----
def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    # Clamp index to valid range [0, n_boot-1]. With small n_boot or extreme
    # alpha, int(alpha/2 * n_boot) could be 0 (using min instead of true 2.5%)
    # which under-reports CI width — and at n_boot - 1 we don't want to read
    # past the array end.
    lo_idx = max(0, min(n_boot - 1, int(alpha / 2 * n_boot)))
    hi_idx = max(0, min(n_boot - 1, int((1 - alpha / 2) * n_boot)))
    return (means[lo_idx], means[hi_idx])


def mean(x):
    return sum(x) / len(x) if x else float("nan")




# ---- Per-intent_mode stratification (paper Table 3) ----
def stratify_by_intent_mode(rows: list[dict], items_lookup: dict) -> dict:
    """Join SR_hit + _intent_mode.mode from dataset jsonl.
    Output per (tier, model, mode) → {n, SR, SR_ci95, SPL, T}.

    Logs how many rows were dropped due to missing `_intent_mode` so a
    half-labeled dataset doesn't silently bias the per-mode stratification
    (e.g., if labeling is in progress and only some SELs have the field)."""
    by_tmm = defaultdict(list)
    dropped = 0
    for r in rows:
        sel = r["selection_id"]
        item = items_lookup.get(sel) or {}
        mode = (item.get("_intent_mode") or {}).get("mode")
        if not mode:
            dropped += 1
            continue
        by_tmm[(r["tier"], r["model"], mode)].append(r)
    if dropped > 0:
        print(f"[metrics] per_intent_mode: dropped {dropped}/{len(rows)} rows "
              f"with no _intent_mode label (label dataset items to include them)")
    out = {}
    for (tier, model, mode), rs in sorted(by_tmm.items()):
        im_vals = [r["IM"] for r in rs if "IM" in r]
        sr_vals = [r["SR_hit"] for r in rs if "SR_hit" in r]
        spl_vals = [r["SPL"] for r in rs if "SPL" in r]
        entry = {
            "n": len(rs),
            "IM": round(mean(im_vals), 4) if im_vals else None,
        }
        if sr_vals:
            entry["SR"] = round(mean(sr_vals), 4)
            sr_ci = bootstrap_ci(sr_vals)
            entry["SR_ci95"] = [round(sr_ci[0], 4), round(sr_ci[1], 4)]
        if spl_vals:
            entry["SPL"] = round(mean(spl_vals), 4)
        out[f"{tier}/{model}/{mode}"] = entry
    return out


# ---- Main aggregation ----
def aggregate(out_dir: Path, radius_m: float = 1.0):
    all_items = load_items()
    split_ids = {it["selection_id"] for it in all_items}
    items_lookup = {it["selection_id"]: it for it in all_items}
    nav_metrics._dropped = 0  # reset counter for this aggregate pass
    print(f"[metrics] items={len(split_ids)} radius_m={radius_m}")

    # IM (Intent Match) — synonym-aware string match between agent's
    # predicted target and dataset's target_category. Measures pure intent
    # comprehension, independent of nav execution.
    synonyms = _load_synonyms()

    # Gather per-(tier,model,item,style) rows
    rows = []  # list of dicts
    for tier, model, style, rec in collect_records(split_ids):
        target_cat = rec.get("target_category", "")
        prediction = (rec.get("prediction") or {}).get("target", "")
        row = {
            "tier": tier, "model": model, "style": style,
            "selection_id": rec["selection_id"],
            "scene_id": rec.get("scene_id", ""),
            "room_type": rec.get("room_type", "") or _derive_room_type(rec),
            "target_category": target_cat,
            "prediction": prediction,
            "IM": int(intent_match(prediction, target_cat, synonyms)),
        }
        # Extra active-tier metrics if this record has a trajectory.
        nav = nav_metrics(rec, radius_m=radius_m)
        row.update(nav)  # adds SR_hit (geometric), OSR_hit, SPL
        if nav:
            vis = trajectory_visibility(rec)
            g_hit = int(vis.get("G_seen", 0)) if vis else 0
            # GSR = geometric reach AND target visible in final frame
            row["GSR_hit"] = int(row.get("SR_hit", 0) and g_hit)
        rows.append(row)
    print(f"[metrics] collected {len(rows)} judged records")
    if not rows:
        print("[metrics] no records — aborting")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Per-item CSV ----
    # Build fieldnames from union of all row keys so active rows (with nav metrics)
    # don't break on csv.DictWriter.
    fieldnames = list({k for r in rows for k in r.keys()})
    # Stable ordering: put common columns first, nav columns last
    preferred = ["tier", "model", "style", "selection_id", "scene_id",
                 "room_type", "target_category", "prediction",
                 "IM",
                 "SR_hit", "GSR_hit",
                 "OSR_hit", "SPL", "d_final", "d_min",
                 "path_length", "geodesic_length", "steps"]
    fieldnames = [k for k in preferred if k in fieldnames] + \
                 [k for k in fieldnames if k not in preferred]
    csv_path = out_dir / "per_item.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[metrics] wrote {csv_path}")
    if nav_metrics._dropped > 0:
        print(f"[metrics] WARN: {nav_metrics._dropped} record(s) dropped from "
              f"nav metrics due to malformed/missing target_position in episode_meta")

    # ---- Headline: per (tier, model, style) means ----
    by_tms = defaultdict(list)
    for r in rows:
        by_tms[(r["tier"], r["model"], r["style"])].append(r)

    headline = {}
    for (tier, model, style), rs in sorted(by_tms.items()):
        im_vals = [r["IM"] for r in rs if "IM" in r]
        entry = {
            "n": len(rs),
            "IM": round(mean(im_vals), 4) if im_vals else None,
        }
        # Active-tier nav metrics: SR (geometric reach), GSR (reach + visible),
        # OSR, SPL — paper table columns.
        sr_vals = [r["SR_hit"] for r in rs if "SR_hit" in r]
        if sr_vals:
            entry["SR"] = round(mean(sr_vals), 4)
            sr_ci = bootstrap_ci(sr_vals)
            entry["SR_ci95"] = [round(sr_ci[0], 4), round(sr_ci[1], 4)]
            gsr_vals = [r["GSR_hit"] for r in rs if "GSR_hit" in r]
            if gsr_vals:
                entry["GSR"] = round(mean(gsr_vals), 4)
            entry["OSR"] = round(mean([r["OSR_hit"] for r in rs if "OSR_hit" in r]), 4)
            entry["SPL"] = round(mean([r["SPL"] for r in rs if "SPL" in r]), 4)
            # Diagnostic only (not in paper tables); kept for appendix.
            entry["TL_mean_m"] = round(mean([r["path_length"] for r in rs if "path_length" in r]), 2)
            entry["steps_mean"] = round(mean([r["steps"] for r in rs if "steps" in r]), 2)
        headline[f"{tier}/{model}/{style}"] = entry

    # ---- Style sensitivity per (tier, model): SR + SPL only ----
    style_results = {}
    for (tier, model), group_key in _group_keys(rows, ("tier", "model")):
        rs = [r for r in rows if (r["tier"], r["model"]) == group_key]
        per_item_SR = defaultdict(dict)
        per_item_SPL = defaultdict(dict)
        for r in rs:
            if "SR_hit" in r:
                per_item_SR[r["selection_id"]][r["style"]] = r["SR_hit"]
            if "SPL" in r:
                per_item_SPL[r["selection_id"]][r["style"]] = r["SPL"]

        # CSR: items where SR_hit == 1 across all 4 styles (cross-style robustness)
        csr_items = [i for i, s in per_item_SR.items() if len(s) == len(STYLES)]
        csr = None
        per_style_SR = None
        delta_SR_style = None
        per_style_SPL = None
        delta_SPL_style = None
        if csr_items:
            csr = round(sum(1 for i in csr_items if all(v == 1 for v in per_item_SR[i].values()))
                        / len(csr_items), 4)
            per_style_SR_lists = {st: [per_item_SR[i][st] for i in per_item_SR if st in per_item_SR[i]]
                                  for st in STYLES}
            per_style_SR = {st: round(mean(v), 4) if v else None
                            for st, v in per_style_SR_lists.items()}
            valid_sr = [per_style_SR[st] for st in STYLES if per_style_SR[st] is not None]
            if valid_sr:
                delta_SR_style = round(max(valid_sr) - min(valid_sr), 4)

            per_style_SPL_lists = {st: [per_item_SPL[i][st] for i in per_item_SPL if st in per_item_SPL[i]]
                                   for st in STYLES}
            per_style_SPL = {st: round(mean(v), 4) if v else None
                             for st, v in per_style_SPL_lists.items()}
            valid_spl = [per_style_SPL[st] for st in STYLES if per_style_SPL[st] is not None]
            if valid_spl:
                delta_SPL_style = round(max(valid_spl) - min(valid_spl), 4)

        style_results[f"{tier}/{model}"] = {
            "per_style_SR": per_style_SR,
            "per_style_SPL": per_style_SPL,
            "delta_SR_style": delta_SR_style,
            "delta_SPL_style": delta_SPL_style,
            "CSR_n_items": len(csr_items),
            "CSR": csr,  # cross-style navigation consistency
        }

    # ---- Per-intent_mode stratification (Table 3) ----
    per_intent_mode = stratify_by_intent_mode(rows, items_lookup)

    # ---- Write metrics.json ----
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({
        "n_records": len(rows),
        "radius_m": radius_m,
        "headline_per_tier_model_style": headline,
        "style_sensitivity": style_results,
        "per_intent_mode": per_intent_mode,
    }, indent=2))
    print(f"[metrics] wrote {metrics_path}")

    # ---- Stdout summary ----
    print("\n===== Headline per (tier/model/style) =====")
    print(f"{'group':<50} {'n':>4} {'SR':>6} {'OSR':>6} {'SPL':>6} {'GSR':>6}")
    for key, v in sorted(headline.items()):
        sr = f"{v.get('SR','-'):>6}" if "SR" in v else f"{'-':>6}"
        osr = f"{v.get('OSR','-'):>6}" if "OSR" in v else f"{'-':>6}"
        spl = f"{v.get('SPL','-'):>6}" if "SPL" in v else f"{'-':>6}"
        g = f"{v.get('GSR','-'):>6}" if "GSR" in v else f"{'-':>6}"
        print(f"{key:<50} {v['n']:>4} {sr} {osr} {spl} {g}")

    print("\n===== Style sensitivity per (tier/model) =====")
    for key, v in sorted(style_results.items()):
        ps_sr = v.get("per_style_SR") or {}
        ps_str = " ".join(f"{st[0]}={ps_sr[st]:.3f}" if ps_sr.get(st) is not None
                           else f"{st[0]}=?" for st in STYLES)
        print(f"{key:<35} ΔSR={v.get('delta_SR_style')}  CSR={v.get('CSR')} "
              f"(n={v.get('CSR_n_items', 0)})  {ps_str}")


def _group_keys(rows, fields):
    seen = set()
    for r in rows:
        key = tuple(r[f] for f in fields)
        if key in seen:
            continue
        seen.add(key)
        yield key, key


def _derive_room_type(rec: dict) -> str:
    rt = (rec.get("room_type") or "").strip()
    if rt:
        return rt.lower()
    room = (rec.get("room") or "").strip()
    stripped = re.sub(r"_\d+$", "", room)
    return stripped.lower() if stripped else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius-m", type=float, default=None,
                    help="If set, run a single aggregate at this radius. "
                         "Otherwise run primary (1.0m) + sensitivity (3.0m).")
    args = ap.parse_args()
    if args.radius_m is not None:
        aggregate(args.out, radius_m=args.radius_m)
    else:
        # Habitat-ObjectNav-aligned reporting:
        #   primary 2.0m  — matches the "viewpoint-radius ≈ object-center
        #                   distance" of Habitat 0.1m-to-VIEW_POINTS
        #                   (viewpoints are pre-sampled 1-2m from object).
        #   strict  1.0m  — Anderson 2018 / Batra 2020 to-object-center.
        #   loose   3.0m  — sensitivity / context.
        aggregate(args.out / "primary_r2m", radius_m=2.0)
        print()
        aggregate(args.out / "strict_r1m", radius_m=1.0)
        print()
        aggregate(args.out / "sensitivity_r3m", radius_m=3.0)


if __name__ == "__main__":
    main()
