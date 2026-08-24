// Stubelius Gold — black/gold paint for the LTX-2.5 Director pack nodes.
// Scoped to this pack; coexists with the global Stubelius Gold theme.
import { app } from "../../scripts/app.js";

const GOLD = "#FFD700";
const BLACK = "#0d0a04";
const TITLE = "#141006";

const PACK_NODES = new Set([
  "LTXDirectorCS25",
  "LTXDirectorGuideCS25",
  "LTXDirectorCropGuidesCS25",
  "LTXChunkWriterCS25",
  "LTXChunkAssemblerCS25",
  "CleanLatentSliceCS25",
  "PromptRelayEncodeTimeline",
]);

function paint(node) {
  if (!PACK_NODES.has(node.comfyClass || "")) return;
  node.color = TITLE;
  node.bgcolor = BLACK;
}

app.registerExtension({
  name: "stubelius.director25.gold",
  nodeCreated(node) {
    paint(node);
    const cls = node.comfyClass || "";
    if (!PACK_NODES.has(cls)) return;
    // Pulsing gold edge on the two flagship nodes only (canvas perf).
    if (cls !== "LTXDirectorCS25" && cls !== "LTXDirectorGuideCS25") return;
    const orig = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
      if (!this.flags?.collapsed) {
        ctx.save();
        const t = window.LiteGraph.NODE_TITLE_HEIGHT;
        const pulse = 0.55 + 0.45 * Math.sin(performance.now() / 700);
        ctx.shadowColor = GOLD;
        ctx.shadowBlur = 7 + 9 * pulse;
        ctx.strokeStyle = GOLD;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(-0.5, -t - 0.5, this.size[0] + 1, this.size[1] + t + 1, 8);
        ctx.stroke();
        ctx.restore();
        this.setDirtyCanvas(true, false);
      }
      orig?.apply(this, arguments);
    };
  },
  loadedGraphNode(node) { paint(node); },
});
