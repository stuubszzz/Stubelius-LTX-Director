import { app } from "../../scripts/app.js";

// Stubelius LTX Finish: a note on the node saying what the last run did. WINNER 0 holds after the
// seeds and saves no final video; the note says so and what to do next. After a finish it names the
// seed and the output size. The text comes from the node's run ("stubelius_note"); it stays until the
// next run and isn't saved with the workflow. The note is as tall as its text at the node's width.

const FINISH = "StubeliusLTXFinish";
const MIN_H = 30;
const MARGIN = 4;            // ComfyUI's gap around a DOM widget (its default is 10)
const ACCENT = "var(--p-primary-color, #64b5f6)";
const DIM = "var(--border-color, #4e4e4e)";

function after(proto, method, fn) {
  const orig = proto[method];
  proto[method] = function () {
    const r = orig?.apply(this, arguments);
    fn.apply(this, arguments);
    return r;
  };
}

function grow(node) {
  try {
    node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
    node.setDirtyCanvas?.(true, true);
  } catch (e) { /* layout only */ }
}

app.registerExtension({
  name: "StubeliusLTX.FinishNote",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== FINISH) return;
    const p = nodeType.prototype;
    after(p, "onNodeCreated", function () {
      const node = this;
      const box = document.createElement("div");
      Object.assign(box.style, {
        display: "none", boxSizing: "border-box", padding: "5px 8px", borderRadius: "6px",
        border: `1px solid ${DIM}`, color: "var(--input-text, #ddd)", font: "12px/1.35 sans-serif",
        whiteSpace: "pre-wrap", overflow: "hidden",
      });
      let height = 0;
      const measure = () => {
        // only once the note is on screen: before ComfyUI lays the widget out it reads as 0 px tall
        if (box.style.display === "none" || !box.getClientRects().length || !box.scrollHeight) return;
        const need = Math.max(MIN_H, box.scrollHeight + 2 + 2 * MARGIN);   // text + border + margins
        if (Math.abs(need - height) > 1) { height = need; grow(node); }
      };
      try {
        const widget = node.addDOMWidget("finish_note", "div", box, {
          serialize: false, hideInPanel: true, margin: MARGIN,
          getMinHeight: () => height, getMaxHeight: () => height,
        });
        widget.serialize = false;             // keeps it out of the saved workflow's widgets_values too
        node._stubeliusNote = { box, measure, setHeight: (h) => { height = h; } };
        if (typeof ResizeObserver === "function") new ResizeObserver(measure).observe(box);   // node resized
      } catch (e) {
        console.warn("[StubeliusLTX] Finish note unavailable:", e);
      }
    });
    after(p, "onExecuted", function (output) {
      const note = this._stubeliusNote;
      const text = output?.stubelius_note?.[0];
      if (!note || typeof text !== "string") return;
      note.box.textContent = text;
      note.box.style.borderColor = output?.stubelius_hold?.[0] ? ACCENT : DIM;
      note.box.style.display = "block";
      note.setHeight(MIN_H);                  // until the text is laid out and measured
      grow(this);
      requestAnimationFrame(note.measure);
    });
  },
});
