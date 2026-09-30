import { app } from "../../scripts/app.js";

// Per-workflow theme. A "Stubelius LTX Theme" node inside a workflow decides how THAT workflow looks
// while its tab is open: node colours, a stage stripe on each title bar, group colours, the canvas
// background and the Director / Live Preview panels. Workflows without the node are untouched;
// "ComfyUI default" clears the theme from this workflow and shows your normal ComfyUI look.
//
// Another pack may run a theme of its own in the same page. This one keeps its own style element
// and node markers, clears itself at once when another workflow opens and applies itself
// last, so it never undoes another pack's look and always wins in its own workflow.

const THEMES = {
  "Studio slate":       { cv: "#1c1d21", nb: "#232429", tb: "#2b2d33", tx: "#e7e8eb", mu: "#8b909c", inp: "#16171b", bd: "#34363e", ac: "#2dd4bf", on: "#0b1413", s: ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24"] },
  "Stuubzzz neon":      { cv: "#141117", nb: "#1b1720", tb: "#241e2a", tx: "#f3eef6", mu: "#a095ab", inp: "#110e14", bd: "#3a2f42", ac: "#ff3da8", on: "#1a0610", s: ["#ff3da8", "#ff8a5c", "#ffd64a", "#7cf0c8"] },
  "Film stock":         { cv: "#171411", nb: "#1f1a15", tb: "#29221b", tx: "#efe6d8", mu: "#a3937e", inp: "#14110d", bd: "#3b3127", ac: "#e8a33d", on: "#1a1206", s: ["#e8a33d", "#c8553d", "#9bb07a", "#d9c7a3"] },
  "Paper light":        { cv: "#e7e6e1", nb: "#f8f7f3", tb: "#eeede7", tx: "#26262a", mu: "#6f6d68", inp: "#ffffff", bd: "#d6d4cc", ac: "#2f6fdd", on: "#ffffff", s: ["#2f6fdd", "#7a4fd6", "#0f8a6e", "#c27a12"] },
  "Midnight blueprint": { cv: "#0c1524", nb: "#101d31", tb: "#15263f", tx: "#dbe7ff", mu: "#7f93b5", inp: "#0a1220", bd: "#233857", ac: "#4f9dff", on: "#06101f", s: ["#4f9dff", "#58e1ff", "#8ef0b6", "#ffc86b"] },
  "ComfyUI default":    null,
};
const STYLE_ID = "stubelius-ltx-workflow-theme";
// Stage of each dashboard node: 0 setup, 1 models, 2 director / seeds (the default), 3 finish.
const STAGES = {
  StubeliusLTXSetup: 0, StubeliusLTXSeedSamplers: 0, StubeliusLTXTheme: 0, MarkdownNote: 0,
  StubeliusLTXModels: 1,
  StubeliusLTXFinish: 3,
};
let savedCanvasBg;            // the canvas colour before this theme touched it
let ownsCanvasBg = false;     // the canvas colour is ours until we put savedCanvasBg back
let active = null;            // name of the theme currently applied, or null

function stageOf(node) {
  const t = node.comfyClass || node.type || "";
  const title = (node.title || "").toLowerCase();
  if (t in STAGES) return STAGES[t];
  if (title.startsWith("final") || title.startsWith("last frame")) return 3;
  return 2;
}

function css(p) {
  return `
.stb-ltx-live-preview, .stb-ltx-live-preview > div { color: ${p.mu} !important; }
.stb-ltx-live-preview-stage { background: ${p.inp} !important; border: 1px solid ${p.bd}; box-sizing: border-box; }
.prcs-controls-group, .prcs-gap-menu, .prcs-settings-menu, .prcs-segmented-control, .prcs-autocomplete-menu,
.prcs-motion-info, .prcs-audio-info, .prcs-prompt-wrapper { background: ${p.nb} !important; border-color: ${p.bd} !important; color: ${p.tx} !important; }
.prcs-btn, .prcs-gap-menu-btn, .prcs-icon-btn, .prcs-msel, .prcs-ref-option-select, .prcs-settings-select, .prcs-settings-toggle-btn,
.prcs-number-control, .prcs-number-btn, .prcs-strength-input, .prcs-character-desc, .prcs-autocomplete-item, .prcs-character-slot,
.prcs-character-preview-wrapper, .prcs-canvas { background: ${p.inp} !important; color: ${p.tx} !important; border-color: ${p.bd} !important; }
.prcs-btn:hover:not(:disabled), .prcs-gap-menu-btn:hover, .prcs-icon-btn:hover, .prcs-msel:hover, .prcs-msel.prcs-msel-open,
.prcs-ref-option-select:hover, .prcs-settings-toggle-btn:hover, .prcs-number-btn:hover, .prcs-character-slot:hover,
.prcs-character-slot.drag-over, .prcs-autocomplete-item:hover { border-color: ${p.ac} !important; color: ${p.ac} !important; }
.prcs-btn.toggle-on, .prcs-icon-btn.active, .prcs-segment.active, .prcs-gap-menu-btn.prcs-msel-selected,
.prcs-autocomplete-item.active, .prcs-character-validate-btn:hover { background: ${p.ac} !important; color: ${p.on} !important; border-color: ${p.ac} !important; }
.prcs-prompt-area, .prcs-timecode, .prcs-strength-label, .prcs-settings-input, .prcs-motion-info span, .prcs-audio-info span { color: ${p.tx} !important; }
.prcs-prompt-label, .prcs-settings-title, .prcs-settings-label, .prcs-segment, .prcs-segment-bounds, .prcs-character-label,
.prcs-character-placeholder, .prcs-msel-caret, .prcs-settings-close-btn { color: ${p.mu} !important; }
.prcs-settings-title, .prcs-settings-divider { border-color: ${p.bd} !important; }
.prcs-prompt-wrapper.focus-active, .prcs-prompt-area:focus, .prcs-character-desc:focus { border-color: ${p.ac} !important; }
.prcs-ref-icon, .prcs-autocomplete-item span { color: ${p.ac} !important; }
.prcs-strength-slider, .prcs-seek-bar, .prcs-zoom-slider { accent-color: ${p.ac} !important; }
[data-node-type^="StubeliusLTX"], [data-node-type^="LTXDirector"] { background: ${p.nb} !important; color: ${p.tx} !important; border-color: ${p.bd} !important; }
[data-node-type^="StubeliusLTX"] header, [data-node-type^="LTXDirector"] header,
[data-node-type^="StubeliusLTX"] .node-title, [data-node-type^="LTXDirector"] .node-title { background: ${p.tb} !important; color: ${p.tx} !important; }
`;
}

function drawStripe(node) {
  if (node.__stbLtxStripe) return;
  node.__stbLtxStripe = true;
  const orig = node.onDrawForeground;
  node.onDrawForeground = function (ctx) {
    const r = orig?.apply(this, arguments);
    if (active && THEMES[active] && this.__stbLtxColor && !this.flags?.collapsed) {
      const h = window.LiteGraph?.NODE_TITLE_HEIGHT || 30;
      ctx.save();
      ctx.fillStyle = this.__stbLtxColor;
      ctx.fillRect(0, -h, 4, h);
      ctx.restore();
    }
    return r;
  };
}

function clearOurColours(graph) {
  // color / bgcolor are accessors on LGraphNode: `delete` leaves them set, undefined clears them.
  for (const n of graph?._nodes || []) {
    if (n.__stbLtxColor || n.__stbLtxStripe) { n.color = undefined; n.bgcolor = undefined; n.__stbLtxColor = null; }
  }
}

function themeNode(graph) {
  return (graph?._nodes || graph?.nodes || []).find((n) => (n.comfyClass || n.type) === "StubeliusLTXTheme");
}

function apply() {
  const graph = app.graph;
  const tn = themeNode(graph);
  const name = tn?.widgets?.find((w) => w.name === "theme")?.value ?? null;
  const p = name ? THEMES[name] : null;
  const canvas = app.canvas;

  let style = document.getElementById(STYLE_ID);
  if (!p) {
    // No theme node (another workflow): never touch its nodes. "ComfyUI default" chosen in THIS
    // workflow: clear the colours our theme wrote.
    active = name;
    style?.remove();
    if (canvas && ownsCanvasBg) canvas.clear_background_color = savedCanvasBg;
    ownsCanvasBg = false;
    if (name) clearOurColours(graph);
    canvas?.setDirty(true, true);
    return;
  }

  active = name;
  if (!style) { style = document.createElement("style"); style.id = STYLE_ID; }
  style.textContent = css(p);
  document.head.appendChild(style);
  if (canvas) {
    if (!ownsCanvasBg) savedCanvasBg = canvas.clear_background_color;
    canvas.clear_background_color = p.cv;
    ownsCanvasBg = true;
  }

  for (const n of graph?._nodes || []) {
    n.color = p.tb;
    n.bgcolor = p.nb;
    n.__stbLtxColor = p.s[stageOf(n)];
    drawStripe(n);
  }
  const stageByGroup = { "1": 0, "2": 2, "3": 2, "4": 3 };
  for (const g of graph?._groups || graph?.groups || []) {
    const k = String(g.title || "").trim()[0];
    if (k in stageByGroup) g.color = p.s[stageByGroup[k]];
  }
  canvas?.setDirty(true, true);
}

// Clearing runs at once; applying waits until the other extensions' deferred work (another pack's
// theme on setTimeout 0) has run, so it lands last.
function refresh() {
  if (themeNode(app.graph)) setTimeout(() => setTimeout(apply, 0), 0);
  else apply();
}
const later = () => setTimeout(refresh, 0);   // for calls made while a graph is still loading

app.registerExtension({
  name: "StubeliusLTX.WorkflowTheme",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "StubeliusLTXTheme") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const w = this.widgets?.find((x) => x.name === "theme");
      if (w) {
        const orig = w.callback;
        w.callback = (...args) => { const res = orig?.apply(w, args); refresh(); return res; };
      }
      later();
      return r;
    };
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      const r = onRemoved?.apply(this, arguments);
      const graph = app.graph;
      setTimeout(() => {           // theme node deleted from this workflow: undo its colours here only
        if (!themeNode(graph)) clearOurColours(graph);
        apply();
      }, 0);
      return r;
    };
  },
  nodeCreated() { if (active) later(); },
  afterConfigureGraph() { refresh(); },
});
