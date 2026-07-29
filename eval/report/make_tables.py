"""Render markdown tables from metrics.json for the dataset benchmark paper.

Five tables, symmetric structure (compreh × nav) × (style × intent_mode):

  Table 1 — Headline (IM/SR/GSR/OSR/SPL per LLM, averaged)
  Table 2 — SR per phrasing style    (nav × style)
  Table 3 — SR per intent_mode       (nav × intent)
  Table 4 — IM per phrasing style    (compreh × style)
  Table 5 — IM per intent_mode       (compreh × intent)

IM = Intent Match: did LLM's plan target_guess match dataset target_category
(synonym-aware string match). Measures pure intent comprehension,
independent of nav execution. SR < IM = nav failures. SR > IM impossible.

Usage:
  python report/make_tables.py --metrics results/<run>/metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean

STYLE_ORDER = ["formal", "natural", "casual", "emotional"]
INTENT_MODE_ORDER = ["EVENT_SCRIPT", "INNER_STATE", "PHYSICAL_STATE", "AFFORDANCE"]


def fmt_pct(v: float | None, d: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{d}f}"


def _group_by_model(headline: dict, want_active: bool) -> dict:
    """Group `tier/model/style` entries by (tier, model). Filter by active flag."""
    out = defaultdict(dict)
    for key, v in headline.items():
        parts = key.split("/")
        if len(parts) != 3:
            continue
        tier, model, style = parts
        is_active = "SR" in v
        if is_active != want_active:
            continue
        out[(tier, model)][style] = v
    return out


def table1_active_headline(headline: dict) -> str:
    """Method-level IM/SR/GSR/OSR/SPL, averaged across the 4 styles."""
    by_model = _group_by_model(headline, want_active=True)
    lines = []
    lines.append("### Table 1 — Headline\n")
    lines.append("*Per LLM backbone, averaged across 4 styles. "
                 "SR = geometric reach (d≤2m). GSR = SR ∧ target visible in final frame.*\n")
    lines.append("| Tier | LLM | n | IM | SR | GSR | OSR | SPL |")
    lines.append("|------|-----|---|----|----|-----|-----|-----|")
    body = 0
    for (tier, model), per_style in sorted(by_model.items()):
        n = sum(v.get("n", 0) for v in per_style.values())
        im_vals  = [v["IM"]  for v in per_style.values() if "IM"  in v and v["IM"] is not None]
        sr_vals  = [v["SR"]  for v in per_style.values() if "SR"  in v]
        gsr_vals = [v["GSR"] for v in per_style.values() if "GSR" in v]
        osr_vals = [v["OSR"] for v in per_style.values() if "OSR" in v]
        spl_vals = [v["SPL"] for v in per_style.values() if "SPL" in v]
        if not sr_vals:
            continue
        lines.append(
            f"| {tier} | {model} | {n} | "
            f"{fmt_pct(_mean(im_vals)) if im_vals else '—'} | "
            f"**{fmt_pct(_mean(sr_vals))}** | "
            f"{fmt_pct(_mean(gsr_vals)) if gsr_vals else '—'} | "
            f"{fmt_pct(_mean(osr_vals)) if osr_vals else '—'} | "
            f"{fmt_pct(_mean(spl_vals)) if spl_vals else '—'} |"
        )
        body += 1
    if body == 0:
        lines.append("| — | — | — | — | — | — | — | — |")
    return "\n".join(lines) + "\n"


def table2_active_per_style(headline: dict) -> str:
    """SR per phrasing style for active LLMs (style robustness)."""
    by_model = _group_by_model(headline, want_active=True)
    lines = []
    lines.append("### Table 2 — SR × Style\n")
    lines.append("*SR per phrasing style. Δ_style = max − min across 4 styles.*\n")
    lines.append("| Tier | LLM | n | Formal SR | Natural SR | Casual SR | Emotional SR | Mean SR | Δ_style |")
    lines.append("|------|-----|---|-----------|------------|-----------|--------------|---------|---------|")
    body = 0
    for (tier, model), per_style in sorted(by_model.items()):
        srs = [per_style.get(s, {}).get("SR") for s in STYLE_ORDER]
        vals = [s for s in srs if s is not None]
        if not vals:
            continue
        n = sum(v.get("n", 0) for v in per_style.values())
        mean_sr = _mean(vals)
        delta = max(vals) - min(vals) if len(vals) >= 2 else None
        lines.append(
            f"| {tier} | {model} | {n} | "
            + " | ".join(fmt_pct(s) for s in srs)
            + f" | **{fmt_pct(mean_sr)}** | {fmt_pct(delta)} |"
        )
        body += 1
    if body == 0:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    return "\n".join(lines) + "\n"


def table4_intent_comprehension(headline: dict) -> str:
    """IM per phrasing style — pure intent comprehension by LLM (no nav)."""
    by_model = _group_by_model(headline, want_active=True)
    lines = []
    lines.append("### Table 4 — IM × Style\n")
    lines.append("*IM per style: did LLM's plan target_guess match dataset target_category. Δ_style = max − min.*\n")
    lines.append("| Tier | LLM | n | Formal IM | Natural IM | Casual IM | Emotional IM | Mean IM | Δ_style |")
    lines.append("|------|-----|---|-----------|------------|-----------|--------------|---------|---------|")
    body = 0
    for (tier, model), per_style in sorted(by_model.items()):
        ims = [per_style.get(s, {}).get("IM") for s in STYLE_ORDER]
        vals = [v for v in ims if v is not None]
        if not vals:
            continue
        n = sum(v.get("n", 0) for v in per_style.values())
        mean_im = _mean(vals)
        delta = max(vals) - min(vals) if len(vals) >= 2 else None
        lines.append(
            f"| {tier} | {model} | {n} | "
            + " | ".join(fmt_pct(v) for v in ims)
            + f" | **{fmt_pct(mean_im)}** | {fmt_pct(delta)} |"
        )
        body += 1
    if body == 0:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    return "\n".join(lines) + "\n"


def table5_im_per_intent_mode(per_intent_mode: dict) -> str:
    """IM per intent_mode — which intent type is hardest for LLM to comprehend."""
    by_model = defaultdict(dict)
    for key, v in per_intent_mode.items():
        parts = key.split("/")
        if len(parts) != 3:
            continue
        tier, model, mode = parts
        if v.get("IM") is None:
            continue
        by_model[(tier, model)][mode] = v
    lines = []
    lines.append("### Table 5 — IM × Intent Mode\n")
    lines.append("*IM per intent type: LLM understanding vs intent category. Δ_intent = max − min.*\n")
    lines.append("| Tier | LLM | n | EVENT_SCRIPT | INNER_STATE | PHYSICAL_STATE | AFFORDANCE | Mean IM | Δ_intent |")
    lines.append("|------|-----|---|--------------|-------------|----------------|------------|---------|----------|")
    body = 0
    for (tier, model), per_mode in sorted(by_model.items()):
        ims = [per_mode.get(m, {}).get("IM") for m in INTENT_MODE_ORDER]
        ns  = [per_mode.get(m, {}).get("n", 0) for m in INTENT_MODE_ORDER]
        vals = [v for v in ims if v is not None]
        if not vals:
            continue
        n = sum(ns)
        mean_im = _mean(vals)
        delta = max(vals) - min(vals) if len(vals) >= 2 else None
        cells = []
        for v, count in zip(ims, ns):
            if v is None or count == 0:
                cells.append("—")
            else:
                cells.append(f"{fmt_pct(v)} (n={count})")
        lines.append(
            f"| {tier} | {model} | {n} | "
            + " | ".join(cells)
            + f" | **{fmt_pct(mean_im)}** | {fmt_pct(delta)} |"
        )
        body += 1
    if body == 0:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    return "\n".join(lines) + "\n"


def table3_active_per_intent_mode(per_intent_mode: dict) -> str:
    """SR per intent_mode for active LLMs (DDN difficulty across intent types)."""
    by_model = defaultdict(dict)
    for key, v in per_intent_mode.items():
        parts = key.split("/")
        if len(parts) != 3:
            continue
        tier, model, mode = parts
        if "SR" not in v:
            continue
        by_model[(tier, model)][mode] = v
    lines = []
    lines.append("### Table 3 — SR × Intent Mode\n")
    lines.append("*SR per intent type. Δ_intent = max − min across 4 modes.*\n")
    lines.append("| Tier | LLM | n | EVENT_SCRIPT | INNER_STATE | PHYSICAL_STATE | AFFORDANCE | Mean SR | Δ_intent |")
    lines.append("|------|-----|---|--------------|-------------|----------------|------------|---------|----------|")
    body = 0
    for (tier, model), per_mode in sorted(by_model.items()):
        srs = [per_mode.get(m, {}).get("SR") for m in INTENT_MODE_ORDER]
        ns  = [per_mode.get(m, {}).get("n", 0) for m in INTENT_MODE_ORDER]
        vals = [s for s in srs if s is not None]
        if not vals:
            continue
        n = sum(ns)
        mean_sr = _mean(vals)
        delta = max(vals) - min(vals) if len(vals) >= 2 else None
        cells = []
        for s, count in zip(srs, ns):
            if s is None or count == 0:
                cells.append("—")
            else:
                cells.append(f"{fmt_pct(s)} (n={count})")
        lines.append(
            f"| {tier} | {model} | {n} | "
            + " | ".join(cells)
            + f" | **{fmt_pct(mean_sr)}** | {fmt_pct(delta)} |"
        )
        body += 1
    if body == 0:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True,
                    help="path to metrics.json from aggregate/compute_metrics.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional output .md path (default: <metrics dir>/tables.md)")
    args = ap.parse_args()
    data = json.loads(args.metrics.read_text())
    headline = data.get("headline_per_tier_model_style", {})
    per_intent_mode = data.get("per_intent_mode", {})

    md = []
    md.append("# DDN Benchmark Tables\n")
    md.append(f"*{data.get('n_records', '?')} records, success radius = "
              f"{data.get('radius_m', '?')}m. All values in % unless noted.*\n\n")
    md.append(table1_active_headline(headline))
    md.append("\n")
    md.append(table2_active_per_style(headline))
    md.append("\n")
    md.append(table3_active_per_intent_mode(per_intent_mode))
    md.append("\n")
    md.append(table4_intent_comprehension(headline))
    md.append("\n")
    md.append(table5_im_per_intent_mode(per_intent_mode))

    out_path = args.out or args.metrics.parent / "tables.md"
    out_path.write_text("".join(md))
    print(f"[tables] wrote {out_path}")
    print("".join(md))


if __name__ == "__main__":
    main()
