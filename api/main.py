"""Catdex web API for the AI Lodge demo (Cloud Run). API-only: the web app is in web/ on Vercel.

It runs the student's own pipeline unchanged (catdex_step1-4): YOLO11 finds and crops the cat, CLIP picks
the coat page, the pet re-identification model fingerprints each photo, and step 3's identify() decides
known / ask / new. Nothing is stored here: the browser keeps the collection (names, small photos,
fingerprints) and sends the fingerprints with each scan.

    GET  /api/health   {"status": "ok"}
    POST /api/scan     photos (multipart) + gallery (JSON) -> crops, coat, fingerprints, decision   [limit: scan]
    POST /api/card     {name, coat, photos} -> trading-card personality (vibe + language model)      [limit: card]
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
import re
import threading
from collections import Counter
from typing import Any

import cv2
import httpx
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

import catdex_step1 as step1
import catdex_step2 as step2
import catdex_step3 as step3
import catdex_step4 as step4  # imported before the text towers go: it embeds the vibe prompts at import
from api.rate_limit import check_rate_limit

try:  # iPhone .HEIC photos, as in the original app (optional)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

log = logging.getLogger("catdex")

# Memory (Cloud Run 2Gi): the coat and vibe prompts are already embedded, so both CLIP text towers can go.
# Only get_image_features() (vision tower + projection) runs from here on.
for _model in (step2.model, step3.model):
    for _part in ("text_model", "text_projection"):
        if hasattr(_model, _part):
            delattr(_model, _part)
gc.collect()
torch.set_grad_enabled(False)

MAX_PHOTOS = 8
MAX_PHOTO_BYTES = 12 * 1024 * 1024
MAX_SIDE = 2048            # big phone photos are shrunk first: the crop is all the models need
CROP_SIDE = 384            # the crop sent back for the album (and for the card writer)
MAX_GALLERY_CATS = 300
MAX_FPS_PER_CAT = 60
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
UNSURE = "unsure"
_lock = threading.Lock()   # one scan at a time per instance: the models share one CPU


def _cors_origins() -> list[str]:
    origins = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    return origins or DEV_ORIGINS


app = FastAPI(title="Catdex API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins(), allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])


# ---- encoding helpers ----------------------------------------------------------------------------
def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def _fp_from_b64(text: str) -> np.ndarray:
    vec = np.frombuffer(base64.b64decode(text), dtype=np.float32)
    if vec.shape != (512,):
        raise ValueError("bad fingerprint")
    return vec


def _thumb_from_b64(text: str) -> np.ndarray:
    th = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
    if th.shape != (1024,):
        raise ValueError("bad thumbnail")
    return th.reshape(32, 32).astype(float)


def _thumb32(crop: np.ndarray) -> np.ndarray:
    """Step 3's near-duplicate thumbnail (32x32 grey), kept by the browser for later sightings."""
    return step3._thumbnail(crop).round().clip(0, 255).astype(np.uint8)


def _jpeg(crop: np.ndarray, side: int = CROP_SIDE) -> str:
    h, w = crop.shape[:2]
    scale = min(1.0, side / max(h, w))
    small = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def _decode_photo(data: bytes) -> np.ndarray | None:
    """Any photo (JPEG, PNG, WebP, HEIC) -> an upright OpenCV BGR image, at most MAX_SIDE pixels."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        return None
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _decode_data_url(url: str) -> np.ndarray | None:
    m = re.match(r"^data:image/[a-z]+;base64,(.+)$", url or "")
    return _decode_photo(base64.b64decode(m.group(1))) if m else None


def coat_vote(crops: list[np.ndarray]) -> str:
    """Step 2 on every crop, then a majority vote (same as catdex_data.coat_vote in the original app)."""
    votes = Counter(step2.classify_coat(c)["coat"] for c in crops)
    best = max(votes.values())
    winners = [coat for coat, n in votes.items() if n == best]
    real = [coat for coat in winners if coat != UNSURE]
    return real[0] if real else winners[0]


def _locked(fn, *args):
    """Runs fn in a worker thread holding the model lock (never held across an await on the event loop)."""
    with _lock:
        return fn(*args)


def _is_dup(a: np.ndarray, b: np.ndarray) -> bool:
    return float(np.abs(a - b).mean()) < step3.DUPLICATE_LIMIT


# ---- routes --------------------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _scan(images: list[tuple[str, bytes]], gallery: list[dict[str, Any]]) -> dict[str, Any]:
    notes: list[str] = []
    crops: list[np.ndarray] = []
    for label, data in images:                                        # Step 1, one photo at a time
        image = _decode_photo(data)
        if image is None:
            notes.append(f"{label}: couldn't open this file. Is it a photo?")
            continue
        result = step1.find_cat_in(image)
        if result["ok"]:
            crops.append(result["crop"])
        else:
            notes.append(f"{label}: {result['message']}")
    keep = step3.distinct_photos(crops)                               # burst shots count once
    dropped = len(crops) - len(keep)
    if dropped:
        notes.append(f"{dropped} photo{'s' if dropped != 1 else ''} looked almost the same as another, so counted once")
    crops = [crops[i] for i in keep]
    if not crops:
        return {"found": False, "notes": notes, "min_new_photos": step3.MIN_ENROL_PHOTOS}

    coat = coat_vote(crops)                                           # Step 2
    db = {"cats": {g["id"]: {"name": g["name"], "fingerprints": np.stack(g["fps"])} for g in gallery if g["fps"]}}
    found = step3.identify(crops, db)                                 # Step 3 (the student's thresholds)
    thumbs = [_thumb32(c) for c in crops]
    by_id = {g["id"]: g for g in gallery}
    # For each of the top matches: which of these photos are new to that cat (not near-duplicates of its album).
    new_for = {}
    for cat_id in found["top3_ids"]:
        stored = by_id[cat_id]["ths"]
        kept: list[int] = []
        for i, th in enumerate(thumbs):
            t = th.astype(float)
            if any(_is_dup(t, s) for s in stored) or any(_is_dup(t, thumbs[k].astype(float)) for k in kept):
                continue
            kept.append(i)
        new_for[cat_id] = kept
    return {
        "found": True, "notes": notes, "coat": coat, "decision": found["decision"],
        "match": {"id": found["cat_id"], "name": found["name"], "similarity": round(found["similarity"], 3)},
        "top3": [{"id": cid, "score": s} for cid, s in zip(found["top3_ids"], found["top3_scores"])],
        "photos": [{"jpg": _jpeg(c), "fp": _b64(fp), "th": _b64(th)} for c, fp, th in zip(crops, found["fingerprints"], thumbs)],
        "new_for": new_for, "min_new_photos": step3.MIN_ENROL_PHOTOS,
        "thresholds": {"match": step3.MATCH_THRESHOLD, "ask": step3.ASK_THRESHOLD},
    }


@app.post("/api/scan")
async def scan(request: Request, photos: list[UploadFile] = File(...), gallery: str = Form("[]")) -> dict[str, Any]:
    if not 1 <= len(photos) <= MAX_PHOTOS:
        raise HTTPException(400, f"Add 1 to {MAX_PHOTOS} photos.")
    images = []
    for i, up in enumerate(photos, start=1):
        data = await up.read()
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(413, f"Photo {i} is larger than 12 MB.")
        images.append((f"Photo {i}", data))
    try:
        raw = json.loads(gallery)
        if not isinstance(raw, list) or len(raw) > MAX_GALLERY_CATS:
            raise ValueError
        cats = [{"id": str(g["id"])[:40], "name": str(g.get("name", ""))[:40],
                 "fps": [_fp_from_b64(f) for f in g.get("fps", [])[:MAX_FPS_PER_CAT]],
                 "ths": [_thumb_from_b64(t) for t in g.get("ths", [])[:MAX_FPS_PER_CAT]]} for g in raw]
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, "The Catdex sent with the scan could not be read.") from None
    check_rate_limit(request, "scan")  # after validation, right before the models run
    return await run_in_threadpool(_locked, _scan, images, cats)


class CardIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    coat: str = Field(min_length=1, max_length=20)
    photos: list[str] = Field(min_length=1, max_length=6)


JSON_REPLY = ("\n\nReply with only one JSON object with the keys title, traits (array of 3 strings), "
              "stats (object with integer stealth, zoomies, charm, chill, mischief), lore and favourite_thing. "
              "No other text and no code fence.")


# The card writer reads the photo through an OpenAI-compatible gateway. The AI Lodge default reads images; the student's
# own Anthropic-SDK script (catdex_step4.py) keeps LLM_MODEL = Claude Haiku 4.5.
DEFAULT_VISION_MODEL = "deepseek-v4-1-flash-260910"


def _llm_config() -> tuple[str, str, str] | None:
    key = (os.getenv("API_KEY") or "").strip()
    if not key:
        return None
    base = ((os.getenv("API_BASE_URL") or "").strip() or "https://api.apiyi.com/v1").rstrip("/")
    model = (os.getenv("VISION_MODEL_NAME") or os.getenv("MODEL_NAME") or "").strip() or DEFAULT_VISION_MODEL  # the card call sends the photo
    return key, base, model


def _parse_personality(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    body = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    raw = json.loads(body)
    missing = set(step4.PERSONALITY_SCHEMA["required"]) - set(raw)
    if missing:
        raise ValueError(f"reply was missing {sorted(missing)}")
    return step4._tidy(raw)


async def _write_personality(cfg: tuple[str, str, str], crop: np.ndarray, name: str, coat: str, vibe: str) -> dict[str, Any]:
    key, base, model = cfg
    image_url = _jpeg(crop, step4.CLAUDE_IMAGE_SIDE)
    payload = {"model": model, "max_tokens": 600, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": step4.card_prompt(name, coat, vibe) + JSON_REPLY}]}]}
    if "deepseek" in model.lower():
        payload["reasoning_effort"] = "none"  # DeepSeek thinks first by default; the card call has a short timeout
    async with httpx.AsyncClient(timeout=step4.CLAUDE_TIMEOUT + 10) as client:
        r = await client.post(f"{base}/chat/completions", json=payload, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"] or ""
    p = _parse_personality(text)
    p["source"] = model
    return p


@app.post("/api/card")
async def card(body: CardIn, request: Request) -> dict[str, Any]:
    crops = [c for c in (_decode_data_url(u) for u in body.photos) if c is not None]
    if not crops:
        raise HTTPException(400, "Send at least one photo of the cat.")
    vibe, _share = await run_in_threadpool(_locked, step4.read_vibe, crops)  # CLIP, from the student's step 4
    stars = step4.RARITY_STARS
    rarity = step4.RARITY.get(body.coat, "Common")
    extra = {"vibe": vibe, "rarity": rarity, "stars": stars.get(rarity, 1)}
    cfg = _llm_config()
    if cfg is None:  # no key (local dev): the student's template card, labelled as such
        return {**step4.personality_from_template(body.name, body.coat, vibe), **extra}
    check_rate_limit(request, "card")  # one hit per language-model card, right before the call
    try:
        return {**await _write_personality(cfg, crops[0], body.name, body.coat, vibe), **extra}
    except Exception as exc:  # provider error or a bad reply: the student's template, labelled (not after 429/503)
        log.warning("card writer failed: %s", type(exc).__name__)
        return {**step4.personality_from_template(body.name, body.coat, vibe), **extra,
                "note": "The AI card writer was unavailable, so this card uses a template."}
