"""T_blind: text-only LLM using only the intent string. No image.

This is the language-prior floor. Records identical schema to oracle so the
judge + aggregation can process both uniformly. `final_frame` is null, so the
judge will not attach an image (that's the intended behavior for blind).

Usage:
  python agents/agent_blind.py --model gemini_3_1_flash --style all --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from clients import call_vlm, MODEL_CATALOG
from common import (
    EPISODES_OUT, STYLES, render_prompt, load_items, tolerant_json_parse,
    intent_for_style, episode_path, save_atomic, now_iso,
)

BLIND_MODELS = tuple(MODEL_CATALOG.keys())


def run_one(item: dict, model: str, style: str, force: bool = False) -> tuple[str, str | None]:
    sel_id = item["selection_id"]
    out_path = episode_path(item["scene_id"], "blind", model, style, sel_id)
    if out_path.exists() and not force:
        return sel_id, "skip"

    intent = intent_for_style(item, style)
    if not intent:
        return sel_id, f"empty intent"

    system_prompt = "You are a helpful embodied-QA assistant. You output only valid JSON. No prose."
    user_prompt = render_prompt("blind_system", intent=intent)

    response, err, usage = call_vlm(
        model_key=model,
        system=system_prompt,
        user_text=user_prompt,
        image_bytes=None,
        temperature=0.0,
        json_mode=True,
    )
    if response is None:
        return sel_id, f"api: {err}"

    parsed, perr = tolerant_json_parse(response)
    pred_target = ""
    if parsed:
        pred_target = str(parsed.get("target", "") or "").strip()

    record = {
        "selection_id": sel_id,
        "tier": "blind",
        "model": model,
        "style": style,
        "scene_id": item["scene_id"],
        "target_category": item["target_category"],
        "intent": intent,
        "photo": item.get("photo"),
        "final_frame": None,  # blind — judge sees no image
        "prediction": {"target": pred_target},
        "model_meta": {
            "provider": MODEL_CATALOG[model]["provider"],
            "model": MODEL_CATALOG[model]["model"],
            "temperature": 0.0,
            "image_used": False,
        },
        "raw_response": response,
        "usage": usage,
        "timestamp": now_iso(),
    }
    save_atomic(record, out_path)
    return sel_id, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini_3_1_flash", choices=BLIND_MODELS)
    ap.add_argument("--style", default="all",
                    choices=["formal", "natural", "casual", "emotional", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated selection_ids to run (skip everything else)")
    args = ap.parse_args()

    items = load_items()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        items = [it for it in items if it["selection_id"] in wanted]
    if args.limit:
        items = items[:args.limit]
    styles = STYLES if args.style == "all" else (args.style,)

    jobs = [(it, args.model, st) for it in items for st in styles]
    print(f"[blind] model={args.model} items={len(items)} "
          f"styles={styles} jobs={len(jobs)} workers={args.workers}")

    stats = {"ok": 0, "skip": 0, "err": 0}
    lock = threading.Lock()
    failure_log = EPISODES_OUT / "_blind_failures.jsonl"
    failure_log.parent.mkdir(parents=True, exist_ok=True)

    def work(job):
        item, model, style = job
        sid, err = run_one(item, model, style, force=args.force)
        return sid, style, err

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            sid, style, err = fut.result()
            with lock:
                if err == "skip":
                    stats["skip"] += 1
                elif err is None:
                    stats["ok"] += 1
                else:
                    stats["err"] += 1
                    with failure_log.open("a") as ff:
                        ff.write(json.dumps({"selection_id": sid, "style": style, "error": err}) + "\n")
            if i % 25 == 0 or i == len(jobs):
                print(f"[blind] {i}/{len(jobs)} ok={stats['ok']} skip={stats['skip']} err={stats['err']}")

    print(f"[blind] done. {stats}")


if __name__ == "__main__":
    main()
