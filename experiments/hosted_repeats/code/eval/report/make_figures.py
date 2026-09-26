"""Render figures from per_item.csv:
  - Per-style T bar chart with bootstrap CI (one subplot per model/tier)
  - Per-room-type T heatmap
  - Per-coarse-category radar (optional)

Usage:
  python report/make_figures.py --csv results/pilot_dev/per_item.csv \
      --out results/pilot_dev/figures/
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLES = ["formal", "natural", "casual", "emotional"]
STYLE_COLOR = {"formal": "#3b5998", "natural": "#2b8a3e",
               "casual":  "#c49102", "emotional": "#b23a3a"}


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot)]


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ("em", "direct_score", "reasoning_score"):
                if k in r:
                    r[k] = int(float(r[k])) if r[k] else 0
            for k in ("T", "R", "C", "SPL", "d_final", "d_min", "path_length"):
                if k in r and r[k] != "":
                    try:
                        r[k] = float(r[k])
                    except Exception:
                        pass
            rows.append(r)
    return rows


def fig_style_bars(rows: list[dict], out_path: Path):
    """Grouped bar chart: per (tier/model), T mean across styles with CI."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = f"{r['tier']}/{r['model']}"
        groups[key][r["style"]].append(r["T"])

    keys = sorted(groups.keys())
    if not keys:
        print("[figures] no rows for style bars")
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4.2), sharey=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        xs = range(len(STYLES))
        ys, cis = [], []
        for s in STYLES:
            vals = groups[key].get(s, [])
            ys.append(sum(vals) / len(vals) if vals else 0.0)
            cis.append(bootstrap_ci(vals))
        lows = [y - c[0] for y, c in zip(ys, cis)]
        highs = [c[1] - y for y, c in zip(ys, cis)]
        colors = [STYLE_COLOR[s] for s in STYLES]
        ax.bar(xs, ys, yerr=[lows, highs], color=colors, capsize=4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(STYLES, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(key, fontsize=10)
        ax.set_ylabel("Target Score T")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-style target score (95% bootstrap CI)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out_path}")


def fig_room_heatmap(rows: list[dict], out_path: Path):
    """Heatmap: rows = room_type, cols = tier/model, values = mean T."""
    cell = defaultdict(list)
    room_types = set()
    groups = set()
    for r in rows:
        rt = (r.get("room_type") or "unknown").lower()
        key = f"{r['tier']}/{r['model']}"
        cell[(rt, key)].append(r["T"])
        room_types.add(rt)
        groups.add(key)
    rt_list = sorted(room_types)
    g_list = sorted(groups)
    if not rt_list or not g_list:
        print("[figures] no rows for room heatmap")
        return

    import numpy as np
    mat = np.full((len(rt_list), len(g_list)), float("nan"))
    for i, rt in enumerate(rt_list):
        for j, key in enumerate(g_list):
            vals = cell.get((rt, key), [])
            if vals:
                mat[i, j] = sum(vals) / len(vals)

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(g_list)),
                                       max(3, 0.4 * len(rt_list) + 2)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(g_list)))
    ax.set_xticklabels(g_list, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(rt_list)))
    ax.set_yticklabels(rt_list, fontsize=9)
    for i in range(len(rt_list)):
        for j in range(len(g_list)):
            v = mat[i, j]
            if not (v != v):  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Target Score T")
    ax.set_title("Target score by room type × tier/model")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    print(f"[figures] {len(rows)} rows loaded")
    args.out.mkdir(parents=True, exist_ok=True)
    fig_style_bars(rows, args.out / "style_bars.png")
    fig_room_heatmap(rows, args.out / "room_heatmap.png")


if __name__ == "__main__":
    main()
