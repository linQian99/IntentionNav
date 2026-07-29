"""OpenAI-compatible VLM/LLM client used by all evaluation agents.

The reported runs used one gateway exposing the three hosted backends through
``/v1/chat/completions``. Any compatible gateway can be configured explicitly.

Provides call_vlm(model_key, system, user_text, image_bytes) → (text,
err, usage). Usage dict has keys {input_tokens, output_tokens, latency_s}.

Env vars:
  INTENTIONNAV_API_BASE_URL  required; OpenAI-compatible ``.../v1`` endpoint
  INTENTIONNAV_API_KEY       required
  INTENTIONNAV_API_PROXY     optional HTTP(S) proxy

The original ``PP_API_*`` names remain accepted for compatibility with the
frozen evaluation-code archives on the dataset page.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------- Config ----------------
API_BASE = (
    os.environ.get("INTENTIONNAV_API_BASE_URL")
    or os.environ.get("PP_API_BASE_URL")
    or ""
).rstrip("/")
MAX_PHOTO_DIM = 768   # raised from 512 — small-object recognition (plate, menorah, air_purifier)
                       # was bottlenecked at 512. 768 still lands in low-res VLM tier (e.g. Gemini
                       # Flash 258 tokens) — minimal cost increase, ~50% more pixels.
_MAX_TOKENS_INTERNAL = 4096
_RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
_RETRY_BODY_HINTS = ["timeout", "unavailable", "rate limit", "overloaded",
                     "transient", "temporarily"]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------- Model catalog ----------------
# Three hosted VLM backends used in the reported evaluation.
MODEL_CATALOG = {
    "gpt5_4":           {"provider": "openai_compatible", "model": "gpt-5.4"},
    "gemini_3_1_flash": {"provider": "openai_compatible", "model": "gemini-3.1-flash-lite-preview"},
    "qwen3_6_plus":     {"provider": "openai_compatible", "model": "qwen3.6-plus"},
}


def _api_key() -> str:
    k = os.environ.get("INTENTIONNAV_API_KEY") or os.environ.get("PP_API_KEY", "")
    if not k:
        raise RuntimeError("INTENTIONNAV_API_KEY is not set")
    if not API_BASE:
        raise RuntimeError("INTENTIONNAV_API_BASE_URL is not set")
    return k


# ---------------- Image encoding helpers ----------------
def load_image_bytes(path: Path) -> tuple[bytes | None, str | None]:
    if not Path(path).exists():
        return None, f"missing: {path}"
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_PHOTO_DIM:
            s = MAX_PHOTO_DIM / max(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90, optimize=True)
        return buf.getvalue(), None
    except Exception as e:
        return None, f"image error: {e}"


def _b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("ascii")


# ---------------- HTTP helper ----------------
# Optional per-call proxy. Set INTENTIONNAV_API_PROXY=http://host:port to route
# calls in this module
# through that proxy WITHOUT touching global http_proxy/https_proxy env
# vars (which would also redirect Isaac Sim's omni.client and break its
# Nucleus asset check).
#
# We prefer `requests.Session` when available — it keeps the underlying
# TCP/TLS connection alive across calls (HTTP/1.1 keep-alive) so each
# step does not pay a fresh handshake. Falls back to plain urllib when
# requests is not installed.
_REQUESTS_OK = False
try:
    import requests as _requests  # noqa: E402
    _REQUESTS_OK = True
except ImportError:
    _requests = None

import threading as _threading  # noqa: E402

_session_local = _threading.local()
_OPENER_CACHE: dict = {}


def _get_opener(proxy: str | None):
    """Build (and cache) a urllib opener for the given proxy URL."""
    if not proxy:
        return None
    if proxy in _OPENER_CACHE:
        return _OPENER_CACHE[proxy]
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    opener = urllib.request.build_opener(handler)
    _OPENER_CACHE[proxy] = opener
    return opener


def _get_session(proxy: str | None):
    """Per-thread requests.Session with keep-alive. Returns None when
    requests isn't installed (caller falls back to urllib)."""
    if not _REQUESTS_OK:
        return None
    sess = getattr(_session_local, "sess", None)
    if sess is None:
        sess = _requests.Session()
        _session_local.sess = sess
    if proxy:
        sess.proxies = {"http": proxy, "https": proxy}
    elif sess.proxies:
        sess.proxies = {}
    return sess


def _reset_session():
    sess = getattr(_session_local, "sess", None)
    if sess is not None:
        try:
            sess.close()
        except Exception:
            pass
    _session_local.sess = None


def _http_post_json(url: str, body: dict, key: str, timeout: int = 120):
    proxy = (
        os.environ.get("INTENTIONNAV_API_PROXY")
        or os.environ.get("PP_API_PROXY")
        or None
    )
    sess = _get_session(proxy)
    if sess is not None:
        # Keep-alive path. Mimic urllib's HTTPError-on-non-200 contract
        # so the caller's existing except branches still handle errors.
        try:
            r = sess.post(
                url,
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
        except _requests.exceptions.RequestException as e:
            # Surface as urllib.URLError so call_vlm's generic except retries.
            raise urllib.error.URLError(str(e)) from e
        if r.status_code != 200:
            try:
                msg = r.text[:300]
            except Exception:
                msg = ""
            raise urllib.error.HTTPError(url, r.status_code, msg, dict(r.headers), None)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"_raw": r.text}

    # Fallback: plain urllib (no keep-alive)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = _get_opener(proxy)
    do_open = opener.open if opener is not None else urllib.request.urlopen
    with do_open(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


# ---------------- Unified call ----------------
def call_vlm(
    model_key: str,
    system: str,
    user_text: str,
    image_bytes: bytes | list[bytes] | None = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    json_mode: bool = True,
) -> tuple[str | None, str | None, dict | None]:
    """Returns (text, err, usage).
    On success: (text, None, {"input_tokens","output_tokens","latency_s"}).
    On failure: (None, err, None).

    `image_bytes` accepts a single bytes (single image, legacy) or a list
    of bytes (multi-image, e.g. freemap + first-person RGB for the
    episode-start planner). Images are appended in order to the user
    content; compatible gateways accept multiple image_url parts in the same
    message.
    """
    if model_key not in MODEL_CATALOG:
        return None, f"unknown model_key: {model_key}", None
    spec = MODEL_CATALOG[model_key]
    model_id = spec["model"]

    # Build OpenAI-compatible body. Normalize image_bytes to a list.
    content: list = [{"type": "text", "text": user_text}]
    if image_bytes is not None:
        imgs = image_bytes if isinstance(image_bytes, list) else [image_bytes]
        for ib in imgs:
            if ib is None:
                continue
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_b64(ib)}",
                    "detail": "high",
                },
            })
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": _MAX_TOKENS_INTERNAL,
    }
    # GPT-5 / o1 / o3 family only supports default temperature=1, so skip.
    is_openai_reasoning = any(t in model_id.lower() for t in ("gpt-5", "o1", "o3"))
    if not is_openai_reasoning:
        body["temperature"] = temperature
    # Qwen-thinking variants: 97% of completion tokens go to hidden reasoning by
    # default. We don't need that for ObjectNav-style action prediction. Disable.
    if "qwen" in model_id.lower():
        body["enable_thinking"] = False
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    key = _api_key()
    url = f"{API_BASE}/chat/completions"

    if "PP_API_MAX_RETRIES" in os.environ:
        max_retries = max(1, _env_int("PP_API_MAX_RETRIES", max_retries))
    else:
        max_retries = max(1, max_retries)
    timeout_s = max(5, _env_int("PP_API_TIMEOUT_S", 120))
    retry_base_s = max(0.0, _env_float("PP_API_RETRY_BASE_SLEEP_S", 2.0))
    retry_max_s = max(retry_base_s, _env_float("PP_API_RETRY_MAX_SLEEP_S", 30.0))
    retry_log = os.environ.get("PP_API_RETRY_LOG", "0") == "1"

    last_err = "unknown"
    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            code, payload = _http_post_json(url, body, key, timeout=timeout_s)
            if code == 200 and isinstance(payload, dict):
                text = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
                u = payload.get("usage") or {}
                usage = {
                    "input_tokens": int(u.get("prompt_tokens") or 0),
                    "output_tokens": int(u.get("completion_tokens") or 0),
                    "latency_s": round(time.perf_counter() - t0, 4),
                }
                return text, None, usage
            # non-200: surface body
            last_err = f"http={code} body={str(payload)[:200]}"
        except urllib.error.HTTPError as e:
            try:
                body_str = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                body_str = str(e)
            last_err = f"http={e.code} {body_str}"
            if e.code not in _RETRY_HTTP_CODES and not any(h in body_str.lower() for h in _RETRY_BODY_HINTS):
                return None, last_err, None
            _reset_session()
        except Exception as e:
            last_err = f"exc: {str(e)[:200]}"
            _reset_session()

        if attempt == max_retries:
            return None, f"exhausted ({max_retries}): {last_err}", None
        sleep_s = min(retry_max_s, retry_base_s * (2 ** (attempt - 1)))
        if retry_log:
            print(f"[vlm/retry] attempt {attempt}/{max_retries} failed: "
                  f"{last_err}; sleep={sleep_s:.1f}s",
                  file=sys.stderr)
        time.sleep(sleep_s + random.uniform(0.0, min(0.5, retry_base_s * 0.1)))

    return None, last_err, None
