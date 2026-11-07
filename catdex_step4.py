"""
Catdex - Step 4: a new cat gets a vibe, a personality and a trading card.

  1. VIBE (zero-shot, CLIP from Step 2): which posture does the cat show? loaf, sprawled, alert,
     crouching, grooming, exploring... -> a vibe like "Cozy Royalty". With several photos, majority vote.
  2. PERSONALITY (generated ONCE, then kept): Claude looks at the photo and writes a title, 3 traits,
     5 fun stats, a short lore line and a favourite thing - returned as structured JSON, so the card
     never breaks. No API key / no internet? A template personality is used instead, automatically.
  3. CARD: Pillow draws a trading card PNG: photo, name, coat page, rarity, vibe, stats, traits, lore.
Everything is stored with the cat in catdex_db (cats.json + cards/ folder).

Install once:  pip install anthropic pillow
API key: set an environment variable called ANTHROPIC_API_KEY, or put it in a .env file next to this
script (never paste the key into this file).

Try it:
  python catdex_step4.py --card cat_0001     -> (re)make the card for one cat already in your Catdex (id or name)
  python catdex_step4.py --card Bluestar --redo  -> same, but writes a NEW personality (e.g. after adding your API key)
  python catdex_step4.py --all               -> make cards for every cat that doesn't have one yet
New cats caught with catdex_step3.py get their card automatically.
"""

import base64
import json
import os
import random
import textwrap
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sys

# ---- API key from a .env file -------------------------------------------------
# Put a file called .env next to this script containing ONE line:  ANTHROPIC_API_KEY=sk-ant-...
# (No extra installs needed. Keep .env out of GitHub!)
HERE = Path(__file__).resolve().parent


# Lines the .env file may contain:
#   ANTHROPIC_API_KEY=...     your key (from Anthropic, or from another platform)
#   ANTHROPIC_BASE_URL=...    ONLY for a key from another platform: that platform's Anthropic-compatible
#                             address (their docs give it). Leave it out for a key from Anthropic itself.
#   ANTHROPIC_AUTH_TOKEN=...  only if that platform says to send the key as a "Bearer" token instead
ENV_SETTINGS = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")


def _load_env_file():
    env_file = HERE / ".env"
    if not env_file.exists():
        return
    raw = env_file.read_bytes()
    # PowerShell's  echo ... > .env  saves UTF-16, Notepad saves UTF-8 - accept both
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig", errors="ignore")
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        if sep and name in ENV_SETTINGS:
            value = value.strip().strip('"').strip("'").strip()
            if value:
                os.environ[name] = value                      # the .env file wins over an old/empty variable


_load_env_file()
# The SDK adds "/v1/messages" itself, so ".../v1" at the end of a base URL would become ".../v1/v1/messages"
if os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/").endswith("/v1"):
    os.environ["ANTHROPIC_BASE_URL"] = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")[:-3]

# python catdex_step4.py --check-key   -> says whether the key can be found (never prints the key itself)
if __name__ == "__main__" and "--check-key" in sys.argv:
    print(f"Script folder:   {HERE}")
    lookalikes = [p.name for p in HERE.iterdir() if p.name.lower().lstrip(".").startswith("env")]
    print(f".env file there: {'yes' if (HERE / '.env').exists() else 'NO'}"
          f"   (files with a similar name: {', '.join(lookalikes) or 'none'})")
    key = (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
    base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if not key:
        print("API key:         NOT found")
    else:
        hint = ""
        if not key.startswith("sk-ant-") and not base:
            hint = "   <- not an Anthropic key: add ANTHROPIC_BASE_URL=<your platform's address> to .env"
        print(f"API key:         found ({len(key)} characters, starts with {key[:3]}){hint}")
    print(f"Sends requests to: {base or 'https://api.anthropic.com (Anthropic itself)'}")
    sys.exit()

import catdex_step2 as step2                    # reuse its CLIP model for the vibe
import catdex_step3 as step3                    # the Catdex database

# ---- Settings you can tune -------------------------------------------------
LLM_MODEL = "claude-haiku-4-5-20251001"         # e.g. "claude-haiku-5-5" or "claude-sonnet-5-5" to compare
USE_LLM = True                                  # False = always use the template personality
CLAUDE_TIMEOUT = 20      # seconds to wait for one Claude reply before giving up and using the template
CLAUDE_RETRIES = 1       # extra tries if the API is busy or slow (SDK default: 2). Longest wait is
                         # roughly CLAUDE_TIMEOUT x (1 + CLAUDE_RETRIES); set 0 for a hard ~20 s limit
CLAUDE_IMAGE_SIDE = 512  # the photo sent to Claude is shrunk to this many pixels (smaller = quicker)
STAT_NAMES = ["stealth", "zoomies", "charm", "chill", "mischief"]

# Vibes: posture descriptions for CLIP (several per vibe, averaged - same trick as the coat pages)
VIBE_PROMPTS = {
    "Cozy Royalty":        ["a photo of a cat sitting in a loaf position with its paws tucked under",
                            "a cat curled up in a ball, sleeping"],
    "Professional Napper": ["a cat lying stretched out on its side",
                            "a cat sprawled on its back showing its belly"],
    "Dignified Sentinel":  ["a cat sitting upright and alert, looking around",
                            "a cat sitting tall with its ears pointed forward"],
    "Chaos Gremlin":       ["a cat crouched low, ready to pounce",
                            "a playful cat jumping or batting at a toy"],
    "Self-Care Icon":      ["a cat licking and grooming its fur",
                            "a cat washing its face with its paw"],
    "Curious Explorer":    ["a cat walking and exploring",
                            "a cat sniffing something curiously"],
}

# Rarity by coat page (how often you'd expect to meet one)
RARITY = {"tabby": "Common", "ginger": "Common", "black": "Common", "grey": "Common",
          "tuxedo": "Uncommon", "white": "Uncommon",
          "calico": "Rare", "tortoiseshell": "Rare", "colourpoint": "Rare"}
RARITY_COLOURS = {"Common": (96, 125, 139), "Uncommon": (46, 125, 50), "Rare": (191, 144, 0)}
RARITY_STARS = {"Common": 1, "Uncommon": 2, "Rare": 3}

CARDS = step3.DB / "cards"


# ---- 1. Vibe ------------------------------------------------------------------
VIBE_NAMES = list(VIBE_PROMPTS)
with torch.no_grad():
    _vecs = []
    for vibe in VIBE_NAMES:
        inputs = step2.processor(text=VIBE_PROMPTS[vibe], return_tensors="pt", padding=True)
        v = step2._normalise(step2._as_tensor(step2.model.get_text_features(**inputs)))
        _vecs.append(step2._normalise(v.mean(dim=0)))
    VIBE_VECTORS = torch.stack(_vecs)


def read_vibe(crops):
    """Majority vote of the posture vibe across the photos. Returns (vibe, share of photos that agreed)."""
    if not isinstance(crops, list):
        crops = [crops]
    votes = []
    for crop in crops:
        photo = step2.clip_features(crop)
        with torch.no_grad():
            votes.append(VIBE_NAMES[int(torch.argmax(VIBE_VECTORS @ photo))])
    vibe, count = Counter(votes).most_common(1)[0]
    return vibe, count / len(votes)


# ---- 2. Personality -----------------------------------------------------------
PERSONALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "traits": {"type": "array", "items": {"type": "string"}},
        "stats": {"type": "object",
                  "properties": {s: {"type": "integer"} for s in STAT_NAMES},
                  "required": STAT_NAMES, "additionalProperties": False},
        "lore": {"type": "string"},
        "favourite_thing": {"type": "string"},
    },
    "required": ["title", "traits", "stats", "lore", "favourite_thing"],
    "additionalProperties": False,
}


def _image_block(crop):
    """Turn an OpenCV crop into the image format the Claude API expects (small JPEG, base64)."""
    h, w = crop.shape[:2]
    scale = min(1.0, CLAUDE_IMAGE_SIDE / max(h, w))      # smaller image = fewer tokens = quicker + cheaper
    small = cv2.resize(crop, (int(w * scale), int(h * scale)))
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(jpg.tobytes()).decode()}}


def _cut(text, limit):
    """Shorten text to at most `limit` characters, at a word boundary, adding '...' if it was cut."""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit - 3].rsplit(" ", 1)[0].rstrip(",;:-") + "..."


def _tidy(p):
    """Make sure the personality fits on the card, whatever came back."""
    p["traits"] = [_cut(t, 28) for t in p.get("traits", [])][:3] or ["Mysterious"]
    p["stats"] = {s: max(1, min(99, int(p.get("stats", {}).get(s, 50)))) for s in STAT_NAMES}
    p["title"] = _cut(p.get("title", "The Enigma"), 40)
    p["lore"] = _cut(p.get("lore", ""), 220)
    p["favourite_thing"] = _cut(p.get("favourite_thing", ""), 40)
    return p


_client = None


def _claude():
    """One Claude client for the whole app (making a new one for every card wastes time on the connection)."""
    global _client
    if _client is None:
        import anthropic                                  # imported here so the template works without it
        _client = anthropic.Anthropic(timeout=CLAUDE_TIMEOUT,      # reads ANTHROPIC_API_KEY
                                      max_retries=CLAUDE_RETRIES)
    return _client


def card_prompt(name, coat, vibe):
    """The instructions for the card writer (shared by Claude here and the web API's OpenAI-compatible call)."""
    return (
        f"You write playful trading cards for Catdex, an app where people collect the cats they meet.\n"
        f"This cat is called {name}. Its Catdex coat page is '{coat}' and its vibe in this photo is '{vibe}'.\n"
        f"Look at the photo and write its card:\n"
        f"- title: 3 to 6 words, like 'The Windowsill Philosopher'\n"
        f"- traits: exactly 3, each 2 to 4 words\n"
        f"- stats: whole numbers from 1 to 99 for {', '.join(STAT_NAMES)}, fitting the photo and vibe\n"
        f"- lore: 1 or 2 fun sentences, at most 35 words\n"
        f"- favourite_thing: a short phrase\n"
        f"Keep it kind, funny and family-friendly. No claims about the cat's health or real behaviour."
    )


def personality_from_claude(crop, name, coat, vibe):
    """Ask Claude for a personality. Returns a dict, or raises an error if anything goes wrong."""
    client = _claude()
    started = time.perf_counter()
    prompt = card_prompt(name, coat, vibe)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": [_image_block(crop), {"type": "text", "text": prompt}]}],
        output_config={"format": {"type": "json_schema", "schema": PERSONALITY_SCHEMA}},
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"Claude stopped early ({response.stop_reason})")
    text = next(block.text for block in response.content if block.type == "text")
    raw = json.loads(text)
    missing = set(PERSONALITY_SCHEMA["required"]) - set(raw)
    if missing:
        raise ValueError(f"reply was missing {sorted(missing)}")
    p = _tidy(raw)
    p["source"] = LLM_MODEL
    print(f"  (Claude wrote the card in {time.perf_counter() - started:.1f} s)")   # see how long it really takes
    return p


TEMPLATES = {
    "Cozy Royalty":        (["The Loaf Monarch", "Duchess of the Cushion", "The Warm Spot Collector"],
                            ["Expert loafer", "Unbothered royalty", "Blanket connoisseur", "Slow blinker"],
                            "the sunniest cushion", dict(stealth=40, zoomies=15, charm=80, chill=95, mischief=25)),
    "Professional Napper": (["The Horizontal Champion", "Sleeper of Many Sofas", "The Belly Trap"],
                            ["Nap specialist", "Floor puddle", "Dreams of tuna", "Hard to wake"],
                            "a warm laptop", dict(stealth=30, zoomies=10, charm=75, chill=99, mischief=30)),
    "Dignified Sentinel":  (["The Windowsill Philosopher", "Keeper of the Corridor", "The Watchful One"],
                            ["Always on duty", "Judges quietly", "Ears like radar", "Very observant"],
                            "a good lookout spot", dict(stealth=75, zoomies=35, charm=70, chill=60, mischief=40)),
    "Chaos Gremlin":       (["The Midnight Sprinter", "Agent of Chaos", "The Pounce Prodigy"],
                            ["Zoomies on demand", "Fearless hunter", "Knocks things over", "Tiny tornado"],
                            "anything that moves", dict(stealth=65, zoomies=95, charm=70, chill=20, mischief=90)),
    "Self-Care Icon":      (["The Fluff Perfectionist", "Grooming Guru", "The Spa Regular"],
                            ["Immaculate coat", "Takes their time", "Fashionably late", "Glows up daily"],
                            "a long grooming session", dict(stealth=50, zoomies=30, charm=90, chill=75, mischief=35)),
    "Curious Explorer":    (["The Neighbourhood Inspector", "Seeker of Boxes", "The Wandering Scholar"],
                            ["Sniffs everything", "Box enthusiast", "Never lost", "Asks questions"],
                            "a brand-new cardboard box", dict(stealth=60, zoomies=60, charm=75, chill=50, mischief=65)),
}


def personality_from_template(name, coat, vibe):
    """No API needed: pick from lists, with small random changes. Same cat name -> same result."""
    rng = random.Random(f"{name}-{coat}-{vibe}")
    titles, traits, favourite, base = TEMPLATES.get(vibe, TEMPLATES["Curious Explorer"])
    title = rng.choice(titles)
    return _tidy({
        "title": title,
        "traits": rng.sample(traits, 3),
        "stats": {s: base[s] + rng.randint(-8, 8) for s in STAT_NAMES},
        "lore": f"{name} is a {coat} cat with {vibe.lower()} energy, best known as {title.lower()}. "
                f"Rumour has it {name} can't resist {favourite}.",
        "favourite_thing": favourite,
        "source": "template",
    })


def make_personality(crop, name, coat, vibe):
    """Claude if possible, otherwise the template - so the app never breaks."""
    if USE_LLM and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        try:
            return personality_from_claude(crop, name, coat, vibe)
        except Exception as e:                            # no internet, bad key, model busy...
            print(f"  (Claude unavailable: {type(e).__name__} - using the template personality)")
    elif USE_LLM:
        print("  (No ANTHROPIC_API_KEY found - using the template personality)")
    return personality_from_template(name, coat, vibe)


# ---- 3. Trading card ----------------------------------------------------------
def _font(size, bold=False):
    """A readable font on Windows, Mac or Linux - falls back to Pillow's built-in one."""
    names = (["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                     # very old Pillow
        return ImageFont.load_default()


def _fit(draw, text, max_width, size, bold=False, min_size=14):
    """Shrink the font until the text fits; if it still doesn't, shorten it at a word with '...'."""
    while size > min_size and draw.textlength(text, font=_font(size, bold)) > max_width:
        size -= 1
    font = _font(size, bold)
    if draw.textlength(text, font=font) > max_width:
        words = text.split()
        while words and draw.textlength(" ".join(words) + "...", font=font) > max_width:
            words.pop()
        text = " ".join(words) + "..."
    return text, font


def draw_card(cat_id, cat, crop):
    """Draw the trading card as a PNG. Returns its path."""
    W, H = 640, 960
    rarity = RARITY.get(cat.get("coat") or "", "Common")
    frame = RARITY_COLOURS[rarity]
    p = cat["personality"]
    card = Image.new("RGB", (W, H), frame)
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([16, 16, W - 16, H - 16], radius=24, fill=(253, 248, 238))      # cream inner card

    # Header: name + rarity stars
    name, name_font = _fit(d, cat["name"], W - 200, 40, bold=True, min_size=24)   # leave room for the stars
    d.text((40, 36), name, font=name_font, fill=(33, 33, 33))
    stars = "★" * RARITY_STARS[rarity] + "☆" * (3 - RARITY_STARS[rarity])
    d.text((W - 40, 44), stars, font=_font(30), fill=frame, anchor="ra")
    title, title_font = _fit(d, p["title"], W - 80, 22, min_size=16)
    d.text((40, 88), title, font=title_font, fill=(90, 90, 90))

    # Photo, cropped to fill a 560x380 box
    box_w, box_h = W - 80, 380
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    scale = max(box_w / img.width, box_h / img.height)
    img = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1))
    left, top = (img.width - box_w) // 2, (img.height - box_h) // 2
    card.paste(img.crop((left, top, left + box_w, top + box_h)), (40, 128))
    d.rectangle([40, 128, 40 + box_w, 128 + box_h], outline=frame, width=4)

    # Coat page, rarity, vibe
    y = 128 + box_h + 16
    d.text((40, y), f"{(cat.get('coat') or '?').title()} page  ·  {rarity}  ·  {cat.get('vibe', '?')}",
           font=_font(20, bold=True), fill=frame)

    # Stats as bars
    y += 40
    for s in STAT_NAMES:
        value = p["stats"][s]
        d.text((40, y), s.title(), font=_font(18), fill=(60, 60, 60))
        d.rounded_rectangle([160, y + 4, 160 + 360, y + 20], radius=8, fill=(225, 220, 210))
        d.rounded_rectangle([160, y + 4, 160 + int(360 * value / 99), y + 20], radius=8, fill=frame)
        d.text((W - 40, y), str(value), font=_font(18, bold=True), fill=(60, 60, 60), anchor="ra")
        y += 30

    # Traits and lore
    y += 6
    traits, traits_font = _fit(d, "  ·  ".join(p["traits"]), W - 80, 18, bold=True, min_size=13)
    d.text((40, y), traits, font=traits_font, fill=(33, 33, 33))
    y += 32
    lines = textwrap.wrap(p["lore"], width=52)
    if len(lines) > 4:                                     # too long: keep 4 lines, end the last one neatly
        lines = lines[:4]
        lines[3] = _fit(d, lines[3] + " ...", W - 80, 18, min_size=18)[0]
    for line in lines:
        d.text((40, y), line, font=_font(18), fill=(70, 70, 70))
        y += 24
    if p.get("favourite_thing"):
        fav, fav_font = _fit(d, f"Favourite thing: {p['favourite_thing']}", W - 80, 18, bold=True, min_size=14)
        d.text((40, y + 10), fav, font=fav_font, fill=frame)

    # Footer
    seen = (cat.get("first_seen") or "")[:10]
    d.text((W // 2, H - 40), f"Catdex {cat_id.replace('cat_', '#')}  ·  first seen {seen}  ·  stats just for fun",
           font=_font(14), fill=(130, 130, 130), anchor="mm")

    CARDS.mkdir(parents=True, exist_ok=True)
    path = CARDS / f"{cat_id}.png"
    card.save(path)
    return path


# ---- Putting it together ------------------------------------------------------
def welcome_new_cat(cat_id, crops, db=None):
    """Called right after a new cat is caught: vibe -> personality (once) -> card. Returns the card path."""
    db = db or step3.load_db()
    cat = db["cats"][cat_id]
    if not isinstance(crops, list):
        crops = [crops]
    if "vibe" not in cat:
        cat["vibe"], _ = read_vibe(crops)
    if "personality" not in cat:                          # generated ONCE, then kept forever
        cat["personality"] = make_personality(crops[0], cat["name"], cat.get("coat") or "mystery", cat["vibe"])
    path = draw_card(cat_id, cat, crops[0])
    cat["card"] = str(path.relative_to(step3.DB))
    step3.save_db(db)
    return path


def card_for_existing_cat(cat_id, db=None):
    """Make a card for a cat that's already in the Catdex, using its stored photos."""
    db = db or step3.load_db()
    cat = db["cats"][cat_id]
    crops = [cv2.imread(str(step3.DB / "photos" / cat_id / p)) for p in cat["photos"]]
    crops = [c for c in crops if c is not None]
    if not crops:
        raise SystemExit(f"No photos found for {cat_id}")
    return welcome_new_cat(cat_id, crops, db)


if __name__ == "__main__":
    import sys
    db = step3.load_db()
    redo = "--redo" in sys.argv                               # also write a brand-new personality
    args = [a for a in sys.argv[1:] if a != "--redo"]
    if len(args) >= 2 and args[0] == "--card":
        wanted = " ".join(args[1:])                           # an id (cat_0001) or a name (Bluestar)
        matches = step3.find_cat_id(wanted, db)
        if not matches:
            sys.exit(f"No cat called '{wanted}'. Run  python catdex_step3.py --list  to see the ids.")
        if len(matches) > 1:
            sys.exit(f"{len(matches)} cats are called '{wanted}' ({', '.join(matches)}). Use the id instead.")
        if redo:
            db["cats"][matches[0]].pop("personality", None)   # forget the old one, so a new one is written
        print(f"Card saved: {card_for_existing_cat(matches[0], db)}")
    elif len(sys.argv) == 2 and sys.argv[1] == "--all":
        todo = [cid for cid, c in db["cats"].items() if "card" not in c]
        for cid in todo:
            print(f"{cid} ({db['cats'][cid]['name']}): {card_for_existing_cat(cid, db)}")
        print(f"Made {len(todo)} card(s).")
    else:
        sys.exit("Usage: python catdex_step4.py --card <cat id or name> [--redo]   or   python catdex_step4.py --all")