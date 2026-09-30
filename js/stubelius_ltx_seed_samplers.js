import { app } from "../../scripts/app.js";

// Stubelius LTX Seed Samplers: a sampler and scheduler per seed. The slots past the seed count of
// the Setup wired into it are greyed out (a disabled widget draws dimmed and takes no input), so the
// box always shows which seeds will render. Seeds ignores the greyed slots anyway.

const BOX = "StubeliusLTXSeedSamplers";
const SETUP = "StubeliusLTXSetup";

function seedCount(box) {
  let src = null;
  try { src = box.getInputNode?.(0) ?? null; } catch (e) { src = null; }
  if (!src?.widgets?.some((w) => w.name === "seeds")) {
    src = (box.graph?._nodes || []).find((n) => n.comfyClass === SETUP) ?? null;
  }
  const w = src?.widgets?.find((x) => x.name === "seeds");
  return w ? Math.max(1, Math.min(4, Number(w.value) || 1)) : 4;   // no Setup: leave all four open
}

function refresh(box) {
  const count = seedCount(box);
  let changed = false;
  for (const w of box.widgets || []) {
    const m = /^seed_(\d)_/.exec(w.name);
    if (!m) continue;
    const off = Number(m[1]) > count;
    if (!!w.disabled !== off) { w.disabled = off; changed = true; }
  }
  if (changed) box.setDirtyCanvas?.(true, true);
}

function refreshAll() {
  for (const n of app.graph?._nodes || []) if (n.comfyClass === BOX) refresh(n);
}

const soon = (fn) => setTimeout(fn, 0);

function after(proto, method, fn) {
  const orig = proto[method];
  proto[method] = function () {
    const r = orig?.apply(this, arguments);
    fn.call(this);
    return r;
  };
}

app.registerExtension({
  name: "StubeliusLTX.SeedSamplers",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === BOX) {
      const p = nodeType.prototype;
      for (const m of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) after(p, m, function () { soon(() => refresh(this)); });
      // classic canvas: also catches changes that fire no callback (undo, paste); only redraws on a change
      after(p, "onDrawForeground", function () { refresh(this); });
    }
    if (nodeData.name === SETUP) {
      after(nodeType.prototype, "onNodeCreated", function () {
        const w = this.widgets?.find((x) => x.name === "seeds");
        if (!w) return;
        const orig = w.callback;
        w.callback = function () {
          const r = orig?.apply(this, arguments);
          soon(refreshAll);
          return r;
        };
      });
    }
  },
  afterConfigureGraph() { soon(refreshAll); },
});
