/* Catdex web app (AI Lodge demo). The API (Cloud Run) runs the student's models; this page keeps the
   collection in localStorage: each cat's name, coat, card and photos, with every photo's re-ID
   fingerprint and a tiny near-duplicate thumbnail, which go back to the API with each scan. */
"use strict";

const API = (window.VITE_API_BASE || "http://localhost:8000").replace(/\/+$/, "");
const STORE = "catdex:v1";
const MAX_PHOTOS_PER_CAT = 20; // keeps localStorage within its few-MB limit
const COATS = {
  colourpoint: ["Colourpoint", "Pointy Pal", "🎭"], calico: ["Calico", "Patchwork Pal", "🧩"],
  tortoiseshell: ["Tortoiseshell", "Tortie Treasure", "🐢"], ginger: ["Ginger", "Ginger Snap", "🧡"],
  tuxedo: ["Tuxedo", "Dapper Tux", "🎩"], white: ["White", "Snowball", "☁️"], tabby: ["Tabby", "Stripe Spotter", "🐯"],
  black: ["Black", "Midnight Friend", "🌙"], grey: ["Grey", "Grey Cloud", "🐭"], unsure: ["Not sure yet", "", "❔"],
};
const STATS = ["stealth", "zoomies", "charm", "chill", "mischief"];
const MILESTONES = [["First Catch", "🐾", 1], ["Neighbourhood Regular", "🏘️", 5], ["Cat Whisperer", "✨", 10]];

/* ---------- storage ---------- */
function load() {
  try { const d = JSON.parse(localStorage.getItem(STORE) || "null"); if (d && d.cats) return d; } catch (e) { /* unreadable: start fresh */ }
  return { cats: {}, next: 1 };
}
let db = load();
function save() {
  try { localStorage.setItem(STORE, JSON.stringify(db)); return true; }
  catch (e) { toast("This browser is out of space for more photos. Remove a cat or some photos, then try again."); return false; }
}
const cats = () => Object.entries(db.cats).map(([id, c]) => ({ id, ...c }));
const now = () => new Date().toISOString();
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
const when = (iso) => new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });

/* ---------- API ---------- */
class Stop extends Error { constructor(status, body) { super(body?.detail?.message || body?.message || (typeof body?.detail === "string" ? body.detail : `HTTP ${status}`)); this.status = status; this.body = body; } }
async function call(path, init) {
  let res;
  try { res = await fetch(API + path, init); }
  catch (e) { throw new Stop(0, { detail: "Couldn't reach the Catdex server. It may be waking up: try again in a moment." }); }
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Stop(res.status, body);
  return body;
}
/** 429 -> the AI Lodge CTA, 503 -> the paused notice. True when handled. */
function quota(err) {
  if (!(err instanceof Stop)) return false;
  if (err.status === 429) {
    document.getElementById("rl-desc").textContent = "Want to build AI that recognises an animal's coat, breed and features from a single photo? AI Lodge students turn curiosity into deployed products, and the showcase is where they get seen.";
    openModal("rl-modal"); return true;
  }
  if (err.status === 503) {
    document.getElementById("paused-msg").textContent = err.message || "This demo is temporarily paused. Check back soon.";
    openModal("paused-modal"); return true;
  }
  return false;
}
let lastFocus = null;
function openModal(id) { lastFocus = document.activeElement; const m = document.getElementById(id); m.hidden = false; m.querySelector("a, button")?.focus(); }
function closeModals() { document.querySelectorAll(".rl-backdrop").forEach((m) => { m.hidden = true; }); lastFocus?.focus?.(); }
document.addEventListener("click", (e) => { if (e.target.matches(".rl-backdrop, [data-close]")) closeModals(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModals(); });

function toast(text) {
  const t = document.createElement("div"); t.className = "toast"; t.role = "status"; t.textContent = text;
  document.body.append(t); setTimeout(() => t.remove(), 5000);
}

/* ---------- catch ---------- */
let tray = []; // File objects waiting to be scanned
let scan = null; // the last scan result while the user decides

function gallery() {
  return cats().map((c) => ({ id: c.id, name: c.name, fps: c.photos.map((p) => p.fp), ths: c.photos.map((p) => p.th) }));
}

function viewCatch(message = "") {
  const urls = tray.map((f) => URL.createObjectURL(f));
  return `<section class="panel catch">
    <h1>Catch a cat</h1>
    <p class="hint">Take or add photos of <b>one</b> cat. A new cat needs <b>3 different photos</b>; a cat you've met before needs just one.</p>
    <div class="pick">
      <label class="btn primary"><input type="file" accept="image/*" capture="environment" data-add hidden>Take a photo</label>
      <label class="btn"><input type="file" accept="image/*,.heic,.heif" multiple data-add hidden>Add from gallery</label>
    </div>
    <ul class="tray" aria-label="Photos to scan">${urls.map((u, i) => `<li><img src="${u}" alt="Photo ${i + 1}"><button type="button" class="x" data-drop="${i}" aria-label="Remove photo ${i + 1}">×</button></li>`).join("")}</ul>
    <button type="button" class="btn big" data-scan ${tray.length ? "" : "disabled"}>${tray.length ? `Scan ${tray.length} photo${tray.length === 1 ? "" : "s"}` : "Add photos to scan"}</button>
    <div id="outcome" aria-live="polite">${message}</div>
  </section>`;
}

const notes = (list) => (list?.length ? `<ul class="notes">${list.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : "");
function coatChip(coat) { const c = COATS[coat] || COATS.unsure; return `<span class="chip">${c[2]} ${esc(c[0])}</span>`; }

async function doScan() {
  const out = document.getElementById("outcome");
  const btn = document.querySelector("[data-scan]");
  btn.disabled = true; btn.textContent = "Scanning…";
  out.innerHTML = `<p class="working">Sniffing out the cat, checking its coat, flipping through the Catdex…</p>`;
  const form = new FormData();
  tray.forEach((f) => form.append("photos", f, f.name || "photo.jpg"));
  form.append("gallery", JSON.stringify(gallery()));
  try {
    scan = await call("/api/scan", { method: "POST", body: form });
  } catch (err) {
    btn.disabled = false; btn.textContent = `Scan ${tray.length} photo${tray.length === 1 ? "" : "s"}`;
    if (quota(err)) { out.innerHTML = `<p class="hint">Scanning has stopped for now.</p>`; return; }
    out.innerHTML = `<p class="bad">${esc(err.message)}</p>`; return;
  }
  btn.textContent = "Scanned";
  if (!scan.found) { out.innerHTML = `<p class="bad"><b>No cat found in these photos.</b> Try a clearer photo of one cat.</p>${notes(scan.notes)}`; btn.disabled = false; btn.textContent = "Scan again"; return; }
  if (scan.decision === "known") return recordSighting(scan.match.id, false);
  if (scan.decision === "ask") return showAsk();
  showNew();
}

function showAsk() {
  const opts = scan.top3.filter((t) => db.cats[t.id]).map((t) => {
    const c = db.cats[t.id];
    return `<button type="button" class="pickcat" data-pick="${esc(t.id)}"><img src="${c.photos[0]?.jpg || ""}" alt=""><span>${esc(c.name)}</span><small>${Math.round(t.score * 100)}% alike</small></button>`;
  }).join("");
  document.getElementById("outcome").innerHTML = `<div class="result"><h2>Have we met?</h2><p>This cat looks a lot like one you've caught before. Is it one of these?</p>
    <div class="picks">${opts}</div><button type="button" class="btn" data-none>None of these: it's a new cat</button>${notes(scan.notes)}</div>`;
}

function showNew() {
  const n = scan.photos.length, need = scan.min_new_photos;
  const strip = scan.photos.map((p) => `<img src="${p.jpg}" alt="">`).join("");
  if (n < need) {
    document.getElementById("outcome").innerHTML = `<div class="result"><h2>A new cat!</h2><div class="strip">${strip}</div>
      <p>To add a new cat, Catdex needs <b>${need} different photos</b> (it has ${n}). Add ${need - n} more, then scan again, so it can recognise this cat next time.</p>${notes(scan.notes)}</div>`;
    document.querySelector("[data-scan]").disabled = false; document.querySelector("[data-scan]").textContent = "Scan again";
    return;
  }
  document.getElementById("outcome").innerHTML = `<div class="result"><h2>A new cat!</h2><div class="strip">${strip}</div><p>Coat page: ${coatChip(scan.coat)}</p>
    <form class="nameit" data-name><label for="catname">Give it a name</label><input id="catname" maxlength="40" required autocomplete="off" placeholder="e.g. Biscuit"><button class="btn primary">Catch!</button></form>${notes(scan.notes)}</div>`;
  document.getElementById("catname").focus();
}

function recordSighting(id, picked) {
  const cat = db.cats[id];
  const add = (scan.new_for[id] || []).map((i) => scan.photos[i]);
  cat.photos = [...cat.photos, ...add].slice(-MAX_PHOTOS_PER_CAT);
  cat.sightings += 1; cat.last_seen = now(); cat.dates.push(cat.last_seen);
  if (!save()) return;
  tray = []; scan = null;
  render(viewCatch(`<div class="result good"><h2>${picked ? "Got it:" : "Welcome back,"} ${esc(cat.name)}!</h2><p>Sighting number ${cat.sightings}.${add.length ? ` ${add.length} new photo${add.length === 1 ? "" : "s"} added, so recognition keeps improving.` : ""}</p><a class="btn" href="#cat/${esc(id)}">Open ${esc(cat.name)}'s profile</a></div>`));
}

async function catchNew(name) {
  const id = `cat_${String(db.next++).padStart(4, "0")}`;
  const t = now();
  db.cats[id] = { name: name.trim().slice(0, 40), coat: scan.coat, first_seen: t, last_seen: t, sightings: 1, dates: [t], photos: scan.photos.slice(0, MAX_PHOTOS_PER_CAT), card: null };
  if (!save()) { delete db.cats[id]; return; }
  tray = []; scan = null;
  location.hash = `#cat/${id}`;
  await makeCard(id);
}

async function makeCard(id) {
  const cat = db.cats[id];
  const slot = document.getElementById("card-slot");
  if (slot) slot.innerHTML = `<p class="working">Writing ${esc(cat.name)}'s trading card…</p>`;
  try {
    const card = await call("/api/card", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: cat.name, coat: cat.coat, photos: cat.photos.slice(0, 6).map((p) => p.jpg) }) });
    cat.card = card; save();
    if (card.note) toast(card.note);
  } catch (err) {
    // After a 429 or 503 the cat stays caught without a card; "Make card" tries again later.
    if (!quota(err)) toast(`The card couldn't be written: ${err.message}`);
  }
  if (location.hash === `#cat/${id}`) route();
}

/* ---------- catdex, profile, badges ---------- */
function tile(c) { return `<a class="tile rarity-${esc((c.card?.rarity || "Common").toLowerCase())}" href="#cat/${esc(c.id)}"><img src="${c.photos[0]?.jpg || ""}" alt=""><b>${esc(c.name)}</b><small>Seen ${c.sightings}×</small></a>`; }

function viewDex() {
  const all = cats();
  if (!all.length) return `<section class="panel empty"><h1>Your Catdex is empty</h1><p>Catch your first cat to start the collection.</p><a class="btn primary" href="#catch">Catch a cat</a></section>`;
  const pages = Object.keys(COATS).map((coat) => {
    const list = all.filter((c) => (c.coat || "unsure") === coat);
    return list.length ? `<section class="page"><h2>${COATS[coat][2]} ${esc(COATS[coat][0])} <small>${list.length}</small></h2><div class="tiles">${list.map(tile).join("")}</div></section>` : "";
  }).join("");
  return `<section class="panel"><h1>Catdex <small>${all.length} cat${all.length === 1 ? "" : "s"}</small></h1>${pages}</section>`;
}

function cardHtml(c) {
  const k = c.card;
  const stars = "★".repeat(k.stars || 1);
  return `<article class="tcard rarity-${esc((k.rarity || "Common").toLowerCase())}" aria-label="${esc(c.name)}'s trading card">
    <header><b>${esc(c.name)}</b><span title="${esc(k.rarity)}">${stars}</span></header>
    <img src="${c.photos[0]?.jpg || ""}" alt="${esc(c.name)}">
    <p class="ttitle">${esc(k.title)}</p>
    <p class="tmeta">${coatChip(c.coat)} <span class="chip">${esc(k.vibe)}</span></p>
    <ul class="traits">${(k.traits || []).map((t) => `<li>${esc(t)}</li>`).join("")}</ul>
    <dl class="stats">${STATS.map((s) => `<dt>${s}</dt><dd><i style="width:${Number(k.stats?.[s]) || 0}%"></i><span>${Number(k.stats?.[s]) || 0}</span></dd>`).join("")}</dl>
    <p class="lore">${esc(k.lore)}</p>
    <p class="fav">Favourite thing: ${esc(k.favourite_thing)}</p>
    <footer>${k.source === "template" ? "Template card" : "Written by AI"} · ${esc(k.rarity)}</footer>
  </article>`;
}

function viewCat(id) {
  const c = db.cats[id];
  if (!c) return `<section class="panel"><h1>Cat not found</h1><a class="btn" href="#dex">Back to the Catdex</a></section>`;
  const card = c.card ? cardHtml({ ...c }) : `<div class="nocard"><p>No trading card yet.</p><button type="button" class="btn primary" data-card="${esc(id)}">Make card</button></div>`;
  return `<section class="panel profile">
    <a class="back" href="#dex">← Catdex</a>
    <div class="cols"><div id="card-slot">${card}</div>
    <div><h1>${esc(c.name)}</h1><p>${coatChip(c.coat)} Seen ${c.sightings} time${c.sightings === 1 ? "" : "s"}. First met ${when(c.first_seen)}, last seen ${when(c.last_seen)}.</p>
      <h2>Photo album <small>${c.photos.length}</small></h2>
      <div class="album">${c.photos.map((p, i) => `<figure><img src="${p.jpg}" alt="Photo ${i + 1} of ${esc(c.name)}">${c.photos.length > 1 ? `<button type="button" class="x" data-unphoto="${i}" aria-label="Remove photo ${i + 1}">×</button>` : ""}</figure>`).join("")}</div>
      <h2>Sightings</h2><ol class="timeline">${c.dates.slice().reverse().map((d) => `<li>${when(d)}</li>`).join("")}</ol>
      <div class="actions"><button type="button" class="btn" data-rename="${esc(id)}">Rename</button>${c.card ? `<button type="button" class="btn" data-card="${esc(id)}">Remake card</button>` : ""}<button type="button" class="btn danger" data-remove="${esc(id)}">Remove cat</button></div>
    </div></div></section>`;
}

function viewBadges() {
  const all = cats();
  const coatBadges = Object.entries(COATS).filter(([k]) => k !== "unsure").map(([coat, [label, badge, emoji]]) => {
    const got = all.some((c) => c.coat === coat);
    return `<li class="${got ? "got" : ""}"><span>${got ? emoji : "🔒"}</span><b>${esc(badge)}</b><small>${got ? "Unlocked" : `Catch a ${label.toLowerCase()} cat`}</small></li>`;
  }).join("");
  const miles = MILESTONES.map(([name, emoji, n]) => `<li class="${all.length >= n ? "got" : ""}"><span>${all.length >= n ? emoji : "🔒"}</span><b>${esc(name)}</b><small>${n} cat${n === 1 ? "" : "s"}</small></li>`).join("");
  const sightings = all.reduce((s, c) => s + c.sightings, 0);
  return `<section class="panel"><h1>Badges</h1><p>${all.length} cats · ${sightings} sightings · ${new Set(all.map((c) => c.coat)).size} coat pages</p>
    <h2>Coat pages</h2><ul class="badges">${coatBadges}</ul><h2>Milestones</h2><ul class="badges">${miles}</ul>
    <details class="reset"><summary>Reset for demo</summary><p>Empties the Catdex in this browser. This can't be undone.</p><button type="button" class="btn danger" data-reset>Empty my Catdex</button></details></section>`;
}

/* ---------- routing and events ---------- */
function render(html) { const v = document.getElementById("view"); v.innerHTML = html; }
function route() {
  const [page, id] = (location.hash.slice(1) || "catch").split("/");
  document.querySelectorAll(".tabs a").forEach((a) => a.setAttribute("aria-current", a.dataset.tab === (page === "cat" ? "dex" : page) ? "page" : "false"));
  if (page === "dex") render(viewDex());
  else if (page === "badges") render(viewBadges());
  else if (page === "cat") render(viewCat(id));
  else render(viewCatch());
}
window.addEventListener("hashchange", route);

document.addEventListener("change", (e) => {
  if (e.target.matches("[data-add]")) { tray = [...tray, ...e.target.files].slice(0, 8); scan = null; render(viewCatch()); }
});
document.addEventListener("submit", (e) => {
  if (e.target.matches("[data-name]")) { e.preventDefault(); const v = document.getElementById("catname").value.trim(); if (v) void catchNew(v); }
});
document.addEventListener("click", (e) => {
  const t = e.target.closest("button");
  if (!t) return;
  if (t.matches("[data-drop]")) { tray.splice(Number(t.dataset.drop), 1); scan = null; render(viewCatch()); }
  else if (t.matches("[data-scan]")) void doScan();
  else if (t.matches("[data-pick]")) recordSighting(t.dataset.pick, true);
  else if (t.matches("[data-none]")) showNew();
  else if (t.matches("[data-card]")) void makeCard(t.dataset.card);
  else if (t.matches("[data-rename]")) {
    const c = db.cats[t.dataset.rename]; const v = prompt("New name", c.name);
    if (v && v.trim()) { c.name = v.trim().slice(0, 40); save(); route(); }
  } else if (t.matches("[data-remove]")) {
    const c = db.cats[t.dataset.remove];
    if (confirm(`Remove ${c.name} from your Catdex?`)) { delete db.cats[t.dataset.remove]; save(); location.hash = "#dex"; }
  } else if (t.matches("[data-unphoto]")) {
    const [, id] = location.hash.slice(1).split("/"); db.cats[id].photos.splice(Number(t.dataset.unphoto), 1); save(); route();
  } else if (t.matches("[data-reset]")) {
    if (confirm("Empty your whole Catdex in this browser?")) { db = { cats: {}, next: 1 }; save(); location.hash = "#catch"; route(); }
  }
});

route();
