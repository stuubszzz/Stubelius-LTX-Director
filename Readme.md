
# Stubelius Ultimate LTX 2.5

The LTX 2.5 Director timeline node plus **Stubelius Ultimate LTX**: one clean LTX 2.5 workflow.
Render 1–4 full seeds with sound, pick the winner, and only the winner gets refined at full size,
frame-interpolated and upscaled. Built on
[WhatDreamsCost's LTX Director](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI).

This repo used to be called **Stubelius-LTX-Director**. Old links and existing installs keep working.

This is the **LTX 2.5** build. If you are on LTX 2.3, use
[WhatDreamsCost-CSGlide](https://github.com/CGlide/WhatDreamsCost-CSGlide) instead.

<div align="center">
<img width="665" height="669" alt="Capture d&#39;écran 2026-08-12 160229" src="https://github.com/user-attachments/assets/dfb8ebbd-9956-4142-b07b-af7d505a06a7" />

<img width="795" height="804" alt="Capture d&#39;écran 2026-08-12 145713" src="https://github.com/user-attachments/assets/899b237c-8761-40ff-a293-a6551087e631" />
---
</div>

## Stubelius Ultimate LTX

The workflow is in [`example_workflows/Stubelius_Ultimate_LTX.json`](example_workflows/Stubelius_Ultimate_LTX.json).
Once the pack is installed it also appears in ComfyUI under **Workflow → Browse Templates → Stubelius-Ultimate-LTX2.5**.
The HOW TO USE note inside the workflow explains every node.

| Node | Job |
|---|---|
| **Stubelius LTX Setup** | How the seeds are rendered: 1–4 seeds, seed, steps, cfg, first-pass scale |
| **Stubelius LTX Seed Samplers** | A sampler and scheduler for each seed; the slots past Setup's seed count are greyed out |
| **Stubelius LTX Models** | Checkpoint (safetensors or GGUF), distill LoRA, two global LoRAs, IC-LoRA, MSR LoRA (with its first-pass strength), text encoder, VAEs, latent upscaler, attention (Sage or Comfy Kitchen), memory options, decode tile size, live preview |
| **Stubelius LTX Output** | Refine strength / steps / sampler, the refine's MSR LoRA and reference strengths, keep or regenerate the seed's audio, final resolution (native up to 4K), final fps through RIFE, RTX VSR or DLSS5 + Color Lock |
| **LTX Director CS (2.5)** | The timeline: prompts, images, audio and motion tracks, size, frame rate, duration |
| **Stubelius LTX Seeds** | Renders the seeds at the first-pass scale, with sound, one preview each |
| **Stubelius LTX Finish** | WINNER 1–4 (0 = hold after the seeds). Full-size seeds (first-pass scale 1.0, the default) are finished as rendered; smaller seeds get a latent x2 upscale + refine. Then RIFE and the upscaler. Changing it re-runs only the finish, from cache |
| **Stubelius LTX Live Preview** | Watch Seeds and Finish while they sample |
| **Stubelius LTX RIFE to FPS**, **Stubelius LTX Color Lock** | Frame rate conversion that keeps hard cuts clean; restores the original colours after DLSS5 |
| **Stubelius LTX Theme** | Colour theme for this workflow only |

**Seed hunt:** set seeds = 3 and WINNER = 0, queue, watch Seed 1–3, set WINNER on Finish and queue
again. Only the finish runs; the seeds come from cache.

Every node and script in this pack has its own name, so it installs next to Stubelius Ultimate H3
(MiniMax) without clashes.

### Other node packs

Load the workflow and use Manager → **Install Missing Custom Nodes**, or install them yourself:

| Pack | Needed for |
|---|---|
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | video outputs |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | live preview, Sage attention and the low-VRAM options |
| [Nvidia RTX nodes](https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI) | RTX VSR upscaling (NVIDIA RTX GPU) |
| [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) | final fps above the Director's (RIFE) |
| [ComfyUI-DLSS5-Enhancer](https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer) | DLSS5 upscaler |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | GGUF models |

### Models

Official files from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5):

| Folder | File |
|---|---|
| `models/diffusion_models` | [`ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors) |
| `models/text_encoders` | [`gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors) |
| `models/vae` | [`ltx-2.5-video-vae-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors), [`ltx-2.5-audio-vae-bf16.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors) |
| `models/latent_upscale_models` | [`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors) |
| `models/vae` | optional live preview: a tiny LTX VAE such as `taeltx2_3.safetensors`, picked as "live preview" on the Models node |
| `models/loras` | optional, for the Director's Licon MSR reference option: Licon's LTX 2.5 MSR LoRA (`LTX-2.5-Licon-MSR-V1.safetensors`), picked as "msr lora" on the Models node. The LTX 2.3 MSR LoRAs don't work on 2.5 |

2.5 needs the Gemma 4 text encoder and will not load a 2.3 one. RIFE downloads its checkpoint the
first time a final fps above the Director's is used.

Developed on an RTX 5090 (32 GB). With less VRAM, turn on the Models node's memory options.

The video decode sizes itself to the free VRAM before it starts. When a clip doesn't fit at the
decode tile size, it steps the tile down and still decodes every frame in one pass; only below
256 px does it decode in crossfaded chunks. It doesn't wait for an out-of-memory error: on
Windows, NVIDIA's driver lets an allocation past the card's VRAM spill into system RAM
instead, so that error never comes. The spilled decode crawls and can reset the driver.

To get a clean out-of-memory error from the rest of ComfyUI too, open NVIDIA Control Panel →
Manage 3D Settings → Program Settings. Pick ComfyUI's `python.exe` and set **CUDA - Sysmem
Fallback Policy** to **Prefer No Sysmem Fallback**.

## The Director

### What works

- Timeline with image, text, audio and video segments
- Prompt zones and prompt relay
- Image anchors and end frames (right click a segment → "pin to last frame")
- Multiple keyframes
- Chunk render for long videos, with audio
- Packed timelines (save a timeline with its assets embedded)

### Reference images (Licon MSR)

The reference features are on in this build (`REFERENCE_FEATURES` in `ltx_director.js` and
`ltx_director.py`). **Licon MSR** runs on the Stubelius MSR25 engine made for LTX 2.5: each
character or ingredient image becomes a slot-embedded reference, as in
[liconstudio/ComfyUI-LTX2.5-MSR](https://github.com/liconstudio/ComfyUI-LTX2.5-MSR).

In Stubelius Ultimate LTX, choose Licon MSR as the Director's reference option and pick the
LTX 2.5 MSR LoRA on the Models node. It is used in both passes:

- **Seeds (first pass):** the Models node's MSR LoRA strength, and the Director's reference strength.
- **Refine (second pass):** the Output node's refine MSR LoRA strength and refine MSR reference.
  About 0.4 keeps the detail without the references repainting the opening. 0 means the
  Director's value.

Ghost Mask and the `@ref` sheets were made for LTX 2.3's behaviour.

### Older nodes

**Stubelius LTX Seed Hunt** and **Stubelius LTX Refine (Pass 2)** are still registered, so
workflows built with them keep loading. The Ultimate LTX nodes replace them.

## Install

Clone into your ComfyUI custom nodes folder:

```
cd ComfyUI/custom_nodes
git clone https://github.com/stuubszzz/Stubelius-Ultimate-LTX2.5.git
```

Then restart ComfyUI. To update, `git pull` in that folder (or Manager → Update All) and restart.

**Do not run this alongside the 2.3 pack.** Both register the same node names, and
ComfyUI will silently load only one of them. Keep one or the other in `custom_nodes`.

## Credits

Based on [WhatDreamsCost's LTX Director](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)
— the original node this is forked from.

LTX models by [Lightricks](https://github.com/Lightricks).

Development assistance from Claude (Anthropic).
