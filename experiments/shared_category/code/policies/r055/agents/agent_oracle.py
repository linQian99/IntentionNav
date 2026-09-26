"""T_oracle: single VLM call per (item, style) using the pre-rendered best photo.

This is the ceiling baseline — measures target-ID accuracy when the agent is
"teleported" to the best viewpoint. No Isaac Sim involved.

Usage:
  python agents/agent_oracle.py --model gpt5_4 --style all --workers 8
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
from clients import call_vlm, load_image_bytes, MODEL_CATALOG
from common import (
    EPISODES_OUT, STYLES, render_prompt, load_items, tolerant_json_parse,
    intent_for_style, episode_path, save_atomic, now_iso,
    scene_photo_path,
)

ORACLE_MODELS = tuple(MODEL_CATALOG.keys())


def run_one(item: dict, model: str, style: str, force: bool = False) -> tuple[str, str | None]:
    sel_id = item["selection_id"]
    out_path = episode_path(item["scene_id"], "oracle", model, style, sel_id)
    if out_path.exists() and not force:
        return sel_id, "skip"

    intent = intent_for_style(item, style)
    if not intent:
        return sel_id, f"empty intent for style {style}"

    scene_id = item["scene_id"]
    photo_rel = item.get("photo")
    if not photo_rel:
        return sel_id, "no photo field in dataset item"
    photo_abs = scene_photo_path(scene_id, photo_rel)
    img_bytes, img_err = load_image_bytes(photo_abs)
    if img_bytes is None:
        return sel_id, f"photo: {img_err}"

    system_prompt = "You are a helpful embodied-QA assistant. You output only valid JSON. No prose."
    user_prompt = render_prompt("oracle_system", intent=intent)

    response, err, usage = call_vlm(
        model_key=model,
        system=system_prompt,
        user_text=user_prompt,
        image_bytes=img_bytes,
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
        "tier": "oracle",
        "model": model,
        "style": style,
        "scene_id": scene_id,
        "target_category": item["target_category"],
        "intent": intent,
        "photo": photo_rel,
        "final_frame": photo_rel,  # oracle = best pre-rendered photo
        "prediction": {"target": pred_target},
        "model_meta": {
            "provider": MODEL_CATALOG[model]["provider"],
            "model": MODEL_CATALOG[model]["model"],
            "temperature": 0.0,
        },
        "raw_response": response,
        "usage": usage,
        "timestamp": now_iso(),
    }
    save_atomic(record, out_path)
    return sel_id, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ORACLE_MODELS)
    ap.add_argument("--style", default="all",
                    choices=["formal", "natural", "casual", "emotional", "all"])
    ap.add_argument("--limit", type=int, default=None, help="max items for quick testing")
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
    print(f"[oracle] model={args.model} items={len(items)} "
          f"styles={styles} jobs={len(jobs)} workers={args.workers}")

    stats = {"ok": 0, "skip": 0, "err": 0}
    lock = threading.Lock()
    failure_log = EPISODES_OUT / "_oracle_failures.jsonl"
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
                print(f"[oracle] {i}/{len(jobs)} ok={stats['ok']} skip={stats['skip']} err={stats['err']}")

    print(f"[oracle] done. {stats}")


if __name__ == "__main__":
    main()
