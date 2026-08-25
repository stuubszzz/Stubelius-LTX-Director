// Stubelius LTX Seed Hunt — 2x2 gold slot panel writing the slots_json widget.
import { app } from "../../scripts/app.js";

const GOLD = "#FFD700", DIM = "#b8960a", TXT = "#f5e5a0", INK = "#0d0b06", BODY = "#14100a";
let SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "res_multistep", "uni_pc"];
let SCHEDULERS = ["linear_quadratic", "simple", "normal", "beta", "karras", "exponential"];

function _firstArray(x) {
  // object_info fields look like [ ["euler","dpmpp_2m",...], {tooltip:...} ]
  // dig out the first element that is actually an array of strings.
  if (Array.isArray(x)) {
    if (x.every(v => typeof v === "string")) return x;
    for (const el of x) { const r = _firstArray(el); if (r) return r; }
  }
  return null;
}
let LISTS_LOADED = false;
function loadListsFromRegistry() {
  // Node definitions are already client-side (same source as native combo widgets,
  // e.g. the Refine's sampler dropdown). No fetch, no timing issues.
  try {
    const LG = window.LiteGraph || globalThis.LiteGraph;
    const req = (cls) => LG?.registered_node_types?.[cls]?.nodeData?.input?.required || {};
    const s = _firstArray(req("StubeliusLTXRefine").sampler_name)
           || _firstArray(req("KSamplerSelect").sampler_name);
    const c = _firstArray(req("StubeliusLTXRefine").scheduler)
           || _firstArray(req("BasicScheduler").scheduler);
    if (Array.isArray(s) && s.length) { SAMPLERS = s; LISTS_LOADED = true; }
    if (Array.isArray(c) && c.length) { SCHEDULERS = c; }
    if (LISTS_LOADED) console.log("[StubeliusLTX] sampler lists from node registry:",
                                  SAMPLERS.length, "samplers,", SCHEDULERS.length, "schedulers");
  } catch (e) { console.warn("[StubeliusLTX] registry lookup failed", e); }
}
async function _tryFetch(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + " -> " + r.status);
  return r.json();
}
async function fetchLists() {
  try {
    const [ks, bs] = await Promise.all([
      _tryFetch("/object_info/KSamplerSelect").catch(() => _tryFetch("/api/object_info/KSamplerSelect")),
      _tryFetch("/object_info/BasicScheduler").catch(() => _tryFetch("/api/object_info/BasicScheduler")),
    ]);
    const sEntry = ks?.KSamplerSelect?.input?.required?.sampler_name;
    const cEntry = bs?.BasicScheduler?.input?.required?.scheduler;
    const s = _firstArray(sEntry);
    const c = _firstArray(cEntry);
    if (Array.isArray(s) && s.length) { SAMPLERS = s; LISTS_LOADED = true; }
    if (Array.isArray(c) && c.length) { SCHEDULERS = c; }
    console.log("[StubeliusLTX] sampler list loaded:", SAMPLERS.length, "samplers,",
                SCHEDULERS.length, "schedulers");
  } catch (e) { console.warn("[StubeliusLTX] sampler list fetch failed, using defaults", e); }
}

function defaultSlots() {
  return [{ enable: true, seed: 42, sampler: "euler", scheduler: "linear_quadratic", steps: 12, cfg: 1.0 },
          { enable: false, seed: 1000045, sampler: "euler", scheduler: "linear_quadratic", steps: 12, cfg: 1.0 },
          { enable: false, seed: 2000048, sampler: "euler", scheduler: "linear_quadratic", steps: 12, cfg: 1.0 },
          { enable: false, seed: 3000051, sampler: "euler", scheduler: "linear_quadratic", steps: 12, cfg: 1.0 }];
}
function readSlots(widget) {
  const def = defaultSlots();
  let v;
  try { v = JSON.parse(widget && widget.value); } catch (e) { v = null; }
  if (!Array.isArray(v)) return def;
  // coerce each entry to a well-formed slot, filling from defaults
  const out = [];
  for (let i = 0; i < 4; i++) {
    const s = (v[i] && typeof v[i] === "object") ? v[i] : {};
    out.push({
      enable: typeof s.enable === "boolean" ? s.enable : (i === 0),
      seed: Number.isFinite(+s.seed) ? +s.seed : def[i].seed,
      sampler: typeof s.sampler === "string" ? s.sampler : "euler",
      scheduler: typeof s.scheduler === "string" ? s.scheduler : "linear_quadratic",
      steps: Number.isFinite(+s.steps) ? +s.steps : 12,
      cfg: Number.isFinite(+s.cfg) ? +s.cfg : 1.0,
    });
  }
  return out;
}

function writeSlots(node, widget, slots) {
  const v = JSON.stringify(slots);
  widget.value = v;                                     // live widget
  const idx = (node.widgets || []).indexOf(widget);      // positional store
  if (Array.isArray(node.widgets_values) && idx >= 0) node.widgets_values[idx] = v;
  if (node.widgets_values_named && typeof node.widgets_values_named === "object") {
    node.widgets_values_named[widget.name] = v;          // named store
  }
  if (widget.callback) widget.callback(v, app.canvas, node);
  app.graph?.change?.();
}

function el(tag, style, parent) {
  const e = document.createElement(tag);
  Object.assign(e.style, style || {});
  if (parent) parent.appendChild(e);
  return e;
}

function _repopulate(sel, list, current) {
  while (sel.firstChild) sel.removeChild(sel.firstChild);
  list.forEach(s => { const o = document.createElement("option"); o.value = o.textContent = s; sel.appendChild(o); });
  sel.value = list.includes(current) ? current : list[0];
}
function buildPanel(node, widget) {
  const root = el("div", {
    display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px",
    background: INK, border: `1px solid ${DIM}`, borderRadius: "8px",
    padding: "6px", fontFamily: "monospace", fontSize: "11px", color: TXT,
  });
  const slots = readSlots(widget);
  const sync = () => writeSlots(node, widget, slots);

  slots.forEach((slot, i) => {
    const cell = el("div", {
      background: BODY, border: `1px solid ${slot.enable ? GOLD : "#3a3018"}`,
      borderRadius: "6px", padding: "5px", display: "flex",
      flexDirection: "column", gap: "3px",
    }, root);

    const head = el("div", { display: "flex", justifyContent: "space-between", alignItems: "center" }, cell);
    const title = el("span", { color: GOLD, fontWeight: "700" }, head);
    title.textContent = `SLOT ${i + 1}`;
    const en = el("input", { accentColor: GOLD }, head);
    en.type = "checkbox"; en.checked = !!slot.enable;
    en.onchange = () => { slot.enable = en.checked; cell.style.borderColor = en.checked ? GOLD : "#3a3018"; sync(); };

    const row = (label, input) => {
      const r = el("div", { display: "flex", justifyContent: "space-between", gap: "4px", alignItems: "center" }, cell);
      const l = el("span", { color: DIM, minWidth: "38px" }, r); l.textContent = label;
      r.appendChild(input); return input;
    };
    const inputStyle = { background: "#0a0804", color: TXT, border: `1px solid ${DIM}`, borderRadius: "3px", width: "100%", fontSize: "11px" };

    const seed = el("input", inputStyle); seed.type = "number"; seed.value = slot.seed ?? 42;
    seed.onchange = () => { slot.seed = parseInt(seed.value) || 0; sync(); };
    row("seed", seed);

    const smp = el("select", inputStyle); smp.dataset.kind = "smp";
    (Array.isArray(SAMPLERS) ? SAMPLERS : []).forEach(s => { const o = document.createElement("option"); o.value = o.textContent = s; smp.appendChild(o); });
    smp.value = slot.sampler || "euler";
    smp.onchange = () => { slot.sampler = smp.value; sync(); };
    row("smplr", smp);

    const sch = el("select", inputStyle); sch.dataset.kind = "sch";
    (Array.isArray(SCHEDULERS) ? SCHEDULERS : []).forEach(s => { const o = document.createElement("option"); o.value = o.textContent = s; sch.appendChild(o); });
    sch.value = slot.scheduler || "linear_quadratic";
    sch.onchange = () => { slot.scheduler = sch.value; sync(); };
    row("sched", sch);

    const stepsCfg = el("div", { display: "flex", gap: "4px" }, cell);
    const steps = el("input", { ...inputStyle, width: "50%" }, stepsCfg);
    steps.type = "number"; steps.value = slot.steps ?? 12; steps.title = "steps";
    steps.onchange = () => { slot.steps = parseInt(steps.value) || 12; sync(); };
    const cfg = el("input", { ...inputStyle, width: "50%" }, stepsCfg);
    cfg.type = "number"; cfg.step = "0.1"; cfg.value = slot.cfg ?? 1.0; cfg.title = "cfg";
    cfg.onchange = () => { slot.cfg = parseFloat(cfg.value) || 1.0; sync(); };
  });
  if (!LISTS_LOADED) loadListsFromRegistry();
  if (LISTS_LOADED) {
    root.querySelectorAll("select[data-kind='smp']").forEach((sel, i) =>
      _repopulate(sel, SAMPLERS, slots[i]?.sampler || "euler"));
    root.querySelectorAll("select[data-kind='sch']").forEach((sel, i) =>
      _repopulate(sel, SCHEDULERS, slots[i]?.scheduler || "linear_quadratic"));
  }
  if (!LISTS_LOADED) {
    fetchLists().then(() => {
      if (!LISTS_LOADED) return;
      root.querySelectorAll("select[data-kind='smp']").forEach((sel, i) =>
        _repopulate(sel, SAMPLERS, slots[i]?.sampler || "euler"));
      root.querySelectorAll("select[data-kind='sch']").forEach((sel, i) =>
        _repopulate(sel, SCHEDULERS, slots[i]?.scheduler || "linear_quadratic"));
    });
  }
  return root;
}

app.registerExtension({
  name: "stubelius.ltx.huntpanel",
  async setup() { loadListsFromRegistry(); if (!LISTS_LOADED) await fetchLists(); },
  nodeCreated(node) {
    if (node.comfyClass !== "StubeliusLTXSeedHunt") return;
    const widget = (node.widgets || []).find(w => w.name === "slots_json");
    if (!widget) return;

    let healed;
    try {
      healed = readSlots(widget);
      writeSlots(node, widget, healed);
    } catch (e) {
      console.error("[StubeliusLTX] readSlots/writeSlots threw:", e && e.stack || e);
      healed = defaultSlots();
    }

    let panel;
    try {
      panel = buildPanel(node, widget);
    } catch (e) {
      console.error("[StubeliusLTX] buildPanel threw:", e && e.stack || e);
      return;   // leave the raw slots_json widget usable as fallback
    }

    // hide the raw textbox only AFTER the panel is safely built
    try { widget.type = "hidden"; widget.computeSize = () => [0, -4]; } catch (e) {}

    try {
      const PANEL_H = 470;
      const dw = node.addDOMWidget("stub_hunt_panel", "div", panel, { serialize: false });
      if (dw) {
        dw.computeSize = (w) => [w || node.size[0], PANEL_H];
        if (dw.options) dw.options.getHeight = () => PANEL_H;
      }
      panel.style.minHeight = (PANEL_H - 10) + "px";
      panel.style.maxHeight = (PANEL_H - 10) + "px";
      panel.style.overflowY = "auto";
      const cs = node.computeSize();
      node.setSize([Math.max(430, node.size[0]), Math.max(cs[1], node.size[1])]);
    } catch (e) {
      console.error("[StubeliusLTX] addDOMWidget threw:", e && e.stack || e);
      // fallback: some frontends expose a 3-arg signature
      try { node.addDOMWidget("stub_hunt_panel", panel, { serialize: false }); }
      catch (e2) { console.error("[StubeliusLTX] addDOMWidget 3-arg also threw:", e2 && e2.stack || e2);
                   try { widget.type = "text"; } catch(e3){}  // un-hide raw widget
                   return; }
    }
    try { node.size[0] = Math.max(node.size[0], 430); node.setDirtyCanvas(true, true); } catch (e) {}
  },
});
