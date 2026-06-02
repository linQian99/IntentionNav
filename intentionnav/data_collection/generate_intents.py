"""Phase 2: Generate human intent annotations using Gemini.

Sends surface photos + scene graph context to Gemini 3.0 Flash,
which proposes implicit intent descriptions for each surface.

Follows the Gemini API pattern from instube/gemini_aug_goal_image_enhance.py:
- JSON response mode
- 1.1s rate limiting
- Checkpoint/resume every 10 records
"""

import argparse
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from intentionnav.data_collection.intent_prompt import format_prompt


def get_api_key():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: neither GOOGLE_API_KEY nor GEMINI_API_KEY is set.")
        exit(1)
    return api_key


def load_scene_data(scene_summary_dir, scene_id):
    """Load object_dict and room_dict for a scene."""
    obj_path = os.path.join(scene_summary_dir, scene_id, "object_dict.json")
    room_path = os.path.join(scene_summary_dir, scene_id, "room_dict.json")

    object_dict = {}
    room_dict = {}

    if os.path.exists(obj_path):
        with open(obj_path, "r", encoding="utf-8") as f:
            object_dict = json.load(f)
    if os.path.exists(room_path):
        with open(room_path, "r", encoding="utf-8") as f:
            room_dict = json.load(f)

    return object_dict, room_dict


def find_best_photo(surface_entry):
    """Find the best quality photo path for a surface from manifest."""
    for photo in surface_entry.get("photos", []):
        if photo.get("quality_ok", False):
            return photo["path"]
    # Fallback: return first photo even if quality not confirmed
    if surface_entry.get("photos"):
        return surface_entry["photos"][0]["path"]
    return None


def _tolerant_json_parse(text):
    """Parse the first complete JSON object in text, ignoring trailing content.

    Gemini sometimes emits extra text after the JSON (markdown fence close,
    comments, a second JSON object). `raw_decode` returns the first valid
    object and we drop the rest.

    Returns (result, None) on success, (None, error_msg) on failure.
    """
    # Strip common noise: leading markdown fences + whitespace.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove ```json ... ``` fences
        cleaned = cleaned.lstrip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    # Find the first { and try raw_decode from there
    start = cleaned.find("{")
    if start < 0:
        return None, "no opening brace found"
    try:
        result, _end = json.JSONDecoder().raw_decode(cleaned[start:])
        return result, None
    except json.JSONDecodeError as e:
        return None, f"{e}"


_RETRY_HINTS = [
    "429", "500", "502", "503", "504",
    "deadline", "timeout", "unavailable", "resource exhausted",
    "rate limit", "connection", "transient", "temporarily",
]


def _is_retryable_error(exc) -> bool:
    msg = str(exc).lower()
    return any(h in msg for h in _RETRY_HINTS)


def generate_intents_for_entry(model, manifest_entry, surface_targets_entry,
                                  photo_path, object_dict, room_dict,
                                  max_retries: int = 6,
                                  log_prefix: str = ""):
    """Call Gemini for a per-category manifest entry.

    Returns (result, error_msg):
      (dict, None)  — success
      (None, str)   — permanent / exhausted-retries failure; `error_msg` is
                      the last exception or validation-error text for logs.

    Retries transient errors (504/503/429/timeout/etc.) with exponential
    backoff (4 / 8 / 16 / 32 / 60 / 60 s) up to max_retries times; gives up
    immediately on permanent errors (malformed JSON, 400/404, content policy).
    """
    from intentionnav.data_collection.intent_prompt import format_per_category_prompt

    prompt_text = format_per_category_prompt(
        manifest_entry, surface_targets_entry, room_dict, object_dict
    )

    parts = [prompt_text]
    if photo_path and os.path.exists(photo_path):
        try:
            from PIL import Image
            parts.append(Image.open(photo_path))
        except Exception as e:
            print(f"{log_prefix}  Warning: Could not load photo {photo_path}: {e}")

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(
                parts,
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": 180},
            )
            if hasattr(response, "text") and response.text:
                result, parse_err = _tolerant_json_parse(response.text)
                if result is None:
                    # Malformed JSON is often transient (Gemini flakiness):
                    # treat it as retryable unless it's the last attempt.
                    is_last = attempt == max_retries
                    if is_last:
                        err = f"malformed JSON after {max_retries} tries: {parse_err}"
                        print(f"{log_prefix}  [exhausted] {err}")
                        return None, err
                    backoff = min(60, 2 ** attempt)
                    print(f"{log_prefix}  [retry {attempt}/{max_retries - 1}] malformed JSON: {parse_err[:120]} — sleeping {backoff}s")
                    time.sleep(backoff)
                    continue
                if "intents" in result and isinstance(result["intents"], list):
                    return result, None
                err = f"unexpected JSON keys: {list(result.keys())}"
                print(f"{log_prefix}  [permanent] {err}")
                return None, err
            # Empty response
            feedback = getattr(response, "prompt_feedback", "(no feedback)")
            err = f"empty response (feedback: {feedback})"
            print(f"{log_prefix}  [permanent] {err}")
            return None, err
        except Exception as e:
            err_text = str(e)[:200]
            is_last = attempt == max_retries
            if not _is_retryable_error(e):
                print(f"{log_prefix}  [permanent] non-retryable error (attempt {attempt}): {err_text}")
                return None, f"non-retryable: {err_text}"
            if is_last:
                print(f"{log_prefix}  [exhausted] gave up after {max_retries} retries: {err_text}")
                return None, f"exhausted after {max_retries} retries: {err_text}"
            backoff = min(60, 2 ** attempt)
            print(f"{log_prefix}  [retry {attempt}/{max_retries - 1}] transient: {err_text[:120]} — sleeping {backoff}s")
            time.sleep(backoff)

    return None, "unknown (loop fell through)"


def _append_failure(failure_log_path, record):
    """Append a single failure record as one JSON line."""
    os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)
    with open(failure_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _one_entry_task(model, scene_id, entry, st_entry, photo_abs, photo_rel,
                     object_dict, room_dict):
    """Worker: call Gemini for one (surface, category) entry.
    Returns (entry_dict_for_save | None, error_msg | None, meta)."""
    sid = entry["surface_id"]
    cat = entry["target_category"]
    result, err_msg = generate_intents_for_entry(
        model, entry, st_entry, photo_abs, object_dict, room_dict,
        log_prefix=f"  [{scene_id} {sid}→{cat}]",
    )
    meta = {"scene_id": scene_id, "surface_id": sid, "target_category": cat}
    if result:
        return ({
            "scene_id": scene_id,
            "sequence_id": entry["sequence_id"],
            "surface_id": sid,
            "surface_category": entry["surface_category"],
            "room": entry["room"],
            "target_category": cat,
            "target_representative": entry["target_representative"],
            "all_targets_in_category": entry["all_targets_in_category"],
            "photo": photo_rel,
            "specific_description": result.get("specific_description", {}),
            "intents": result["intents"],
        }, None, meta)
    return (None, err_msg, meta)


def process_scene(scene_id, args, model):
    """Process all manifest entries (per category) in a single scene.
    Concurrency: `args.workers` parallel Gemini calls via ThreadPoolExecutor.
    Thread-safety: one lock around the shared result list + checkpoint write."""
    manifest_path = os.path.join(args.photos_dir, scene_id, "capture_manifest.json")
    surface_targets_path = os.path.join(args.surface_targets_dir, scene_id, "surface_targets.json")

    if not os.path.exists(manifest_path):
        print(f"[SKIP] {scene_id}: no capture_manifest.json")
        return None
    if not os.path.exists(surface_targets_path):
        print(f"[SKIP] {scene_id}: no surface_targets.json")
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(surface_targets_path, "r", encoding="utf-8") as f:
        surface_targets = json.load(f)

    st_by_id = {s["surface_id"]: s for s in surface_targets}
    object_dict, room_dict = load_scene_data(args.scene_summary, scene_id)

    output_dir = os.path.join(args.output_dir, scene_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "intent_annotations.json")
    failure_log = os.path.join(args.output_dir, "logs", "failures.jsonl")

    # Resume: load existing
    existing = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    done_keys = {(e["surface_id"], e["target_category"]) for e in existing}

    results = list(existing)
    n_success = 0
    n_failed = 0
    fail_details = []
    lock = threading.Lock()

    # Build pending task list
    pending = []
    for entry in manifest:
        sid = entry["surface_id"]
        cat = entry["target_category"]
        if (sid, cat) in done_keys:
            print(f"  [CACHED] {sid} → {cat}")
            continue
        st_entry = st_by_id.get(sid)
        if not st_entry:
            msg = "surface not in surface_targets"
            print(f"  [SKIP] {sid}: {msg}")
            _append_failure(failure_log, {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "scene_id": scene_id, "surface_id": sid, "target_category": cat,
                "stage": "lookup", "error": msg,
            })
            fail_details.append((sid, cat, msg))
            n_failed += 1
            continue
        photos = entry.get("photos", [])
        photo_rel = photos[0]["path"] if photos else None
        photo_abs = os.path.join(args.photos_dir, scene_id, photo_rel) if photo_rel else None
        pending.append((entry, st_entry, photo_abs, photo_rel))

    if not pending:
        print(f"[OK] {scene_id}: nothing new to do ({len(results)} cached)")
        return {"scene_id": scene_id, "num_surfaces": len(results),
                "num_failed_this_pass": n_failed}

    workers = max(1, int(getattr(args, "workers", 1)))
    print(f"  [PARALLEL] {len(pending)} tasks, {workers} worker(s)")
    completed = 0

    def _run_one(task_tuple):
        entry, st_entry, photo_abs, photo_rel = task_tuple
        return _one_entry_task(
            model, scene_id, entry, st_entry, photo_abs, photo_rel,
            object_dict, room_dict,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, t) for t in pending]
        for fut in as_completed(futures):
            try:
                entry_dict, err_msg, meta = fut.result()
            except Exception as e:
                # Defensive: generate_intents_for_entry should not raise, but guard anyway
                print(f"  [UNCAUGHT] {e}")
                continue
            sid, cat = meta["surface_id"], meta["target_category"]
            with lock:
                if entry_dict is not None:
                    results.append(entry_dict)
                    n_success += 1
                    n_intents = len(entry_dict["intents"])
                    print(f"  [OK] {sid} → {cat}: {n_intents} intents")
                else:
                    _append_failure(failure_log, {
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                        "scene_id": scene_id, "surface_id": sid, "target_category": cat,
                        "stage": "gemini_call", "error": err_msg,
                    })
                    fail_details.append((sid, cat, err_msg))
                    n_failed += 1
                    print(f"  [FAIL] {sid} → {cat}: {str(err_msg)[:120]}")
                completed += 1
                if completed % 10 == 0:
                    _save_atomic(results, output_path)
                    print(f"  [CHECKPOINT] saved {len(results)} entries ({completed}/{len(pending)} done)")

    _save_atomic(results, output_path)
    print(f"[OK] {scene_id}: {len(results)} surfaces annotated  "
          f"(+{n_success} new, {n_failed} failed this pass)")
    if fail_details:
        print(f"[FAIL-SUMMARY] {scene_id}: {n_failed} failures")
        for sid, cat, err in fail_details:
            print(f"    - {sid} → {cat}: {str(err)[:120]}")
    return {
        "scene_id": scene_id,
        "num_surfaces": len(results),
        "num_failed_this_pass": n_failed,
    }


def _save_atomic(data, path):
    """Atomic save with temp file. Sorts by sequence_id (ties broken by
    surface_id + target_category) so on-disk ordering is stable regardless of
    the non-deterministic parallel completion order."""
    if isinstance(data, list):
        data = sorted(data, key=lambda e: (
            e.get("sequence_id", 0),
            e.get("surface_id", ""),
            e.get("target_category", ""),
        ))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Generate intent annotations with Gemini")
    parser.add_argument("--surface-targets-dir", required=True, help="Dir with surface_targets.json per scene")
    parser.add_argument("--scene-summary", required=True, help="Path to kujiale_scene_summary/")
    parser.add_argument("--photos-dir", required=True, help="Dir with capture_manifest.json + photos per scene")
    parser.add_argument("--output-dir", required=True, help="Output dir for intent annotations")
    parser.add_argument("--scene", default=None, help="Single scene ID (for testing)")
    parser.add_argument("--model", default="models/gemini-3-flash-preview", help="Gemini model name")
    parser.add_argument("--workers", type=int, default=8, help="Parallel Gemini calls per scene (default 8)")
    args = parser.parse_args()

    # Setup Gemini
    api_key = get_api_key()
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise SystemExit(
            "google-generativeai is required for annotation. "
            "Install it with `pip install google-generativeai`."
        ) from e
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=args.model)

    # Get scene list
    if args.scene:
        scene_ids = [args.scene]
    else:
        scene_ids = sorted([
            d for d in os.listdir(args.surface_targets_dir)
            if os.path.isdir(os.path.join(args.surface_targets_dir, d))
            and os.path.exists(os.path.join(args.surface_targets_dir, d, "surface_targets.json"))
        ])

    print(f"Processing {len(scene_ids)} scenes with {args.model}...")

    all_stats = []
    for sid in scene_ids:
        print(f"\n{'='*40} {sid} {'='*40}")
        stats = process_scene(sid, args, model)
        if stats:
            all_stats.append(stats)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {len(all_stats)}/{len(scene_ids)} scenes annotated")
    total = sum(s["num_surfaces"] for s in all_stats)
    print(f"Total surfaces with intents: {total}")


if __name__ == "__main__":
    main()
