"""Optional OpenAI-compatible LLM-as-judge scaffold.

Reusable helpers for image-grounded LLM judgments:
  - HTTP / retry layer (_call_judge)
  - Prompt loader (load_prompt / render_prompt)
  - Image loader + resolver (load_image_bytes, resolve_final_frame)
  - Atomic record write-back (_save_atomic)

No active judge task currently. σ_T (target match) was retired in favor of
deterministic IM (synonym match, eval/vocab/category_synonyms.yaml) and G
(occlusion-aware target visibility, eval/aggregate/visibility.py). This file
remains as a scaffold for future judge tasks (e.g. reasoning quality,
trajectory critique).

ENV:
  INTENTIONNAV_API_BASE_URL  required when calling _call_judge
  INTENTIONNAV_API_KEY       required when calling _call_judge
  JUDGE_MODEL       optional override; default gpt-5.5
  INTENTIONNAV_API_PROXY     optional per-call proxy

The original PP_API_* names remain accepted for compatibility with the
archived evaluation snapshot.
"""

from __future__ import annotations
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# Relative paths — derive project root from __file__ so the tree can relocate.
_THIS = Path(__file__).resolve()
REPO = _THIS.parents[2]
EVAL_DIR = REPO / "eval"
PROMPTS_DIR = EVAL_DIR / "judge/prompts"
DATASET_ROOT = Path(os.environ.get(
    "INTENTIONNAV_DATASET_ROOT",
    str(REPO / "data/benchmark"),
))
if not DATASET_ROOT.is_absolute():
    DATASET_ROOT = REPO / DATASET_ROOT

API_BASE = (
    os.environ.get("INTENTIONNAV_API_BASE_URL")
    or os.environ.get("PP_API_BASE_URL")
    or ""
).rstrip("/")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.5")
MAX_PHOTO_DIM = 512   # matches sim render + agent inputs; 1024 was VLM-token-overkill
_RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
_RETRY_BODY_HINTS = ["timeout", "unavailable", "rate limit", "overloaded",
                     "transient", "temporarily"]


def _api_key() -> str:
    k = os.environ.get("INTENTIONNAV_API_KEY") or os.environ.get("PP_API_KEY", "")
    if not k:
        raise RuntimeError("INTENTIONNAV_API_KEY is not set")
    if not API_BASE:
        raise RuntimeError("INTENTIONNAV_API_BASE_URL is not set")
    return k


def _tolerant_json_parse(text: str):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        for kw in ("json", "JSON"):
            if cleaned.startswith(kw):
                cleaned = cleaned[len(kw):]
                break
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    if start < 0:
        return None, "no opening brace"
    try:
        result, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        return result, None
    except json.JSONDecodeError as e:
        return None, str(e)


# ---------------- Prompt loader ----------------
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **vars) -> str:
    """Replace <<KEY>> markers with values — avoids brace-escape issues in prompts."""
    text = load_prompt(name)
    for k, v in vars.items():
        text = text.replace(f"<<{k.upper()}>>", str(v))
    return text


# ---------------- Photo loader ----------------
def load_image_bytes(path: Path) -> tuple[bytes | None, str | None]:
    if not path.exists():
        return None, f"image missing: {path}"
    try:
        from PIL import Image
    except ImportError:
        return None, "PIL not installed"
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_PHOTO_DIM:
            scale = MAX_PHOTO_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85, optimize=True)
        return buf.getvalue(), None
    except Exception as e:
        return None, f"image read error: {e}"


def resolve_final_frame(record: dict) -> Path | None:
    """Resolve the image a future judge should see for this episode.

    Order of preference:
      1. `final_frame` field — agent's actual last RGB observation, saved
         relative to REPO root by agent_vlm/random/fbe.
      2. Oracle-style `photo` field → dataset/<scene_id>/photos/<basename>
         (still valid for the oracle tier, which never has a final_frame).
      3. Blind tier has no image at all (final_frame=None) → return None.
    """
    ff = record.get("final_frame")
    if ff:
        p = Path(ff)
        if not p.is_absolute():
            p = REPO / ff
        if p.exists():
            return p
    scene = record.get("scene_id")
    photo = record.get("photo")
    if scene and photo:
        p = DATASET_ROOT / scene / "photos" / Path(photo).name
        if p.exists():
            return p
    return None


# ---------------- OpenAI-compatible HTTP helper ----------------
def _image_data_url(img_bytes: bytes) -> str:
    import base64
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _call_judge(user_prompt: str, img_bytes: bytes | None,
                max_retries: int = 5) -> tuple[dict | None, str | None]:
    """Single judge call through the configured compatible endpoint."""
    content: list = [{"type": "text", "text": user_prompt}]
    if img_bytes is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_data_url(img_bytes), "detail": "high"},
        })
    body = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You output only valid JSON. No prose."},
            {"role": "user", "content": content},
        ],
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }
    # gpt-5.x / o1 / o3 only support default temperature=1
    if not any(t in JUDGE_MODEL.lower() for t in ("gpt-5", "o1", "o3")):
        body["temperature"] = 0.0

    key = _api_key()
    url = f"{API_BASE}/chat/completions"
    last_err = "unknown"

    _proxy_url = (
        os.environ.get("INTENTIONNAV_API_PROXY")
        or os.environ.get("PP_API_PROXY")
    )
    if _proxy_url:
        _opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": _proxy_url, "https": _proxy_url})
        )
        _open = _opener.open
    else:
        _open = urllib.request.urlopen

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with _open(req, timeout=120) as r:
                payload = json.loads(r.read())
            text = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            result, perr = _tolerant_json_parse(text)
            if result is None:
                if attempt == max_retries:
                    return None, f"malformed JSON: {perr}"
                time.sleep(min(30, 2 ** attempt))
                continue
            return result, None
        except urllib.error.HTTPError as e:
            try:
                body_str = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                body_str = str(e)
            last_err = f"http={e.code} {body_str}"
            if e.code not in _RETRY_HTTP_CODES and not any(h in body_str.lower() for h in _RETRY_BODY_HINTS):
                return None, last_err
        except Exception as e:
            last_err = f"exc: {str(e)[:200]}"
        if attempt == max_retries:
            return None, f"exhausted ({max_retries}): {last_err}"
        time.sleep(min(30, 2 ** attempt))
    return None, last_err


# ---------------- Atomic record write-back ----------------
def _save_atomic(record: dict, path: Path, dedup_key: str | None = None):
    """Cross-process safe atomic write to a per-episode record.json.

    Future judges that run concurrently across the same record.json should
    use this to avoid losing each other's writes. If `dedup_key` is given,
    re-reads inside the lock and skips the write if another worker has
    already populated record["judge"][dedup_key] — useful for idempotent
    streaming judges.
    """
    import fcntl
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        with lock_path.open("a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            if dedup_key is not None:
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if (existing.get("judge", {}) or {}).get(dedup_key):
                        return  # another worker won; do not overwrite
                except Exception:
                    pass  # missing/corrupt — proceed with our write
            tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
