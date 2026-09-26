"""Aggregate token / latency usage across episode records.

Scans `results/eval_out/<tier>/<model>/<style>/*.json` for records that
carry a `usage` (passive tiers) or `usage_total` (active tiers) field, and
produces per-(tier, model, style) statistics plus a projected cost for a
full 500×{styles} run using pricing from `configs/models.yaml`.

Usage:
  python aggregate/token_stats.py --out ../results/bench_pilot/token_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "agents"))
from common import EPISODES_OUT, EVAL_DIR  # noqa: E402


def _load_pricing() -> dict[str, dict[str, float]]:
    """Return {model_key: {"in": usd_per_1m_input, "out": usd_per_1m_output}}.
    Aggregated across vlms + blind_llm sections of models.yaml."""
    try:
        import yaml
    except ImportError:
        print("[warn] PyYAML missing — cost projection disabled", file=sys.stderr)
        return {}
    path = EVAL_DIR / "configs/models.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pricing: dict[str, dict[str, float]] = {}
    for section in ("vlms", "blind_llm"):
        for key, spec in (cfg.get(section) or {}).items():
            p_in = spec.get("pricing_usd_per_1m_input")
            p_out = spec.get("pricing_usd_per_1m_output")
            if p_in is not None and p_out is not None:
                pricing[key] = {"in": float(p_in), "out": float(p_out)}
    return pricing


def _iter_records():
    for tier_dir in sorted(EPISODES_OUT.iterdir()):
        if not tier_dir.is_dir():
            continue
        for model_dir in sorted(tier_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for style_dir in sorted(model_dir.iterdir()):
                if not style_dir.is_dir():
                    continue
                for p in sorted(style_dir.rglob("*.json")):
                    if p.name.startswith("_"):
                        continue
                    try:
                        rec = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    yield tier_dir.name, model_dir.name, style_dir.name, rec


def _episode_usage(rec: dict) -> dict | None:
    """Extract per-episode usage totals. Handles both passive (`usage`) and
    active (`usage_total`) records. Returns None if neither is present."""
    if "usage_total" in rec and rec["usage_total"]:
        u = rec["usage_total"]
        return {
            "n_calls": int(u.get("n_calls", 0) or 0),
            "input_tokens": int(u.get("input_tokens_total", 0) or 0),
            "output_tokens": int(u.get("output_tokens_total", 0) or 0),
            "latency_s": float(u.get("latency_s_total", 0.0) or 0.0),
        }
    if "usage" in rec and rec["usage"]:
        u = rec["usage"]
        return {
            "n_calls": 1,
            "input_tokens": int(u.get("input_tokens", 0) or 0),
            "output_tokens": int(u.get("output_tokens", 0) or 0),
            "latency_s": float(u.get("latency_s", 0.0) or 0.0),
        }
    return None


def _policy_model_for_cost(tier: str, model_key: str) -> str | None:
    """Map (tier, model) → the model_key whose pricing table should be used
    for the DOMINANT usage of that tier.

    For tiers where the model column encodes a compound name (e.g.,
    "random_walk+gemini_flash_vlm"), extract the VLM component.
    """
    if tier in ("blind", "oracle"):
        return model_key
    if tier == "vlm":
        return model_key
    # random / fbe use compound names like "random_walk+gemini_flash_vlm"
    if "+" in model_key:
        suffix = model_key.split("+", 1)[1]
        # strip common suffixes
        for suff in ("_vlm", "_blind"):
            if suffix.endswith(suff):
                return suffix[:-len(suff)]
        return suffix
    return model_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="markdown report path (optional)")
    ap.add_argument("--project-to", type=int, default=500,
                    help="project cost to N items (default 500)")
    ap.add_argument("--styles-project", type=int, default=1,
                    help="multiply projection by this many styles (default 1)")
    args = ap.parse_args()

    pricing = _load_pricing()

    # Gather per-(tier, model, style) lists of per-episode usage dicts
    by_group: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for tier, model, style, rec in _iter_records():
        u = _episode_usage(rec)
        if u is None:
            continue
        by_group[(tier, model, style)].append(u)

    if not by_group:
        print("[token_stats] no records with usage/usage_total found. "
              "Run agents after the token instrumentation is in place.",
              file=sys.stderr)
        return

    lines = [
        "# IntentEQA Token / Latency Report",
        "",
        f"Projection target: {args.project_to} items × {args.styles_project} style(s) = "
        f"{args.project_to * args.styles_project} episodes.",
        "",
        "Per-episode means (across pilot runs):",
        "",
        "| Tier/Model/Style | n | calls | in tok | out tok | latency s | $ proj |",
        "|---|---|---|---|---|---|---|",
    ]
    print(f"{'group':<60} {'n':>3} {'calls':>6} {'in_tok':>8} "
          f"{'out_tok':>8} {'lat_s':>7} {'$_proj':>8}")
    grand_cost = 0.0
    for (tier, model, style), usages in sorted(by_group.items()):
        n = len(usages)
        mean_in = statistics.mean(u["input_tokens"] for u in usages)
        mean_out = statistics.mean(u["output_tokens"] for u in usages)
        mean_cal = statistics.mean(u["n_calls"] for u in usages)
        mean_lat = statistics.mean(u["latency_s"] for u in usages)

        pricing_key = _policy_model_for_cost(tier, model)
        proj_cost = None
        if pricing_key and pricing_key in pricing:
            p_in = pricing[pricing_key]["in"]
            p_out = pricing[pricing_key]["out"]
            total_episodes = args.project_to * args.styles_project
            proj_cost = total_episodes * (mean_in * p_in + mean_out * p_out) / 1e6
            grand_cost += proj_cost

        key = f"{tier}/{model}/{style}"
        cost_str = f"${proj_cost:.3f}" if proj_cost is not None else "—"
        print(f"{key:<60} {n:>3d} {mean_cal:>6.1f} {mean_in:>8.0f} "
              f"{mean_out:>8.0f} {mean_lat:>7.2f} {cost_str:>8}")
        lines.append(
            f"| {key} | {n} | {mean_cal:.1f} | {int(mean_in)} | {int(mean_out)} | "
            f"{mean_lat:.2f} | {cost_str} |"
        )

    lines.append("")
    lines.append(f"**Grand projected cost ({args.project_to}×{args.styles_project}): "
                 f"${grand_cost:.2f}**")
    print()
    print(f"Grand projected cost ({args.project_to}×{args.styles_project}): "
          f"${grand_cost:.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines), encoding="utf-8")
        print(f"[token_stats] wrote {args.out}")


if __name__ == "__main__":
    main()
