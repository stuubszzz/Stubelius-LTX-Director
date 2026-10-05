"""Stubelius Ultimate LTX: Setup, Seed Samplers, Models, Output, Seeds and Finish around the LTX 2.5 Director.

Setup   decides HOW the seeds are rendered: how many (1-4), the seed, steps, cfg and the first-pass
        scale. At 1.0 (the default) the seeds render at the Director's full size and the winner is
        finished as rendered; below 1 they render smaller and Finish upscales + refines the winner.
Seed Samplers  the sampler and scheduler of each seed (the slots past Setup's count are greyed out).
Models  loads the checkpoint, text encoder, VAEs and latent upscaler and applies the LoRAs and the
        speed/memory patches. Its model / clip / vae / audio_vae feed the Director; the bundle
        (models) feeds Seeds and Finish. Loaded files are kept per name, so changing a LoRA or a
        switch doesn't read the checkpoint from disk again. The MSR LoRA (and its first-pass
        strength) sits here too; the Director guides load it in each pass.
Output  decides what comes OUT: how far the full-size refine may move the winner (and the MSR
        strengths of that second pass), the final resolution, frame rate and upscaler. It feeds
        only Finish, so changing it after a seed hunt re-runs only the finish.
Seeds   renders 1-4 full seeds with sound at the first-pass scale (the guides are built once) and
        hands them to Finish as one bundle (takes).
Finish  turns the WINNER into the final video: as rendered when the seeds are full size, else a latent
        upscale and refine to full size (the seed's audio locked); then RIFE to the final frame rate
        and RTX VSR or DLSS5 + Color Lock to the final resolution. The refine is memoised, so frame
        rate / upscaler changes don't repeat it.
Speed/memory patches, the live preview and the RTX VSR / DLSS5 / GGUF nodes are called by class
name at runtime and skipped (or reported clearly) when their pack isn't installed.
"""
import itertools
import logging
import os
import types

import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import torch

from .stubelius_ltx import _decode, _free_vram, _guide, _holds_run_models, _sample_av, _unwrap, call_node

log = logging.getLogger(__name__)

SEED_STEP = 1000003     # seed k = seed + (k - 1) * SEED_STEP, as the old Seed Hunt slots
# The official LTX 2.5 first pass: euler_ancestral; linear_quadratic at 8 steps gives exactly its
# distilled sigmas (1.0 ... 0.975, 0.909, 0.725, 0.422, 0). Used when the Seed Samplers box is bypassed.
DEFAULT_SAMPLER, DEFAULT_SCHEDULER = "euler_ancestral", "linear_quadratic"
UPSCALERS = ["RTX VSR", "DLSS5 + Color Lock"]
AUDIO_MODES = ["keep the seed's audio", "regenerate in the refine"]
DLSS5_FACTORS = {1.5: "1.5x (Quality)", 1.724: "1.724x (Balanced)", 2.0: "2x (Performance)",
                 3.0: "3x (Ultra Performance)"}
# ComfyUI-DLSS5-Enhancer's install_runtime.py asks for release tag "3.0", which doesn't exist (404).
DLSS5_RUNTIME_URL = ("https://github.com/Merserk/dlss5-visual-enhancer/releases/download/v3.0/"
                     "DLSS.5.Visual.Enhancer.v3.0.zip")
VSR_MAX_SCALE = 4.0     # RTX VSR per pass; more runs in two passes
# Final resolution: the value is the exact short side in pixels (None = keep the refined size).
RESOLUTIONS = {
    "native (no upscale)": None,
    "720p": 720,
    "1080p": 1080,
    "2K (1440p)": 1440,
    "4K (2160p)": 2160,
}
PREVIEW_MAX_RES = 640   # live preview frames are scaled to this long side (keeps full-size refines quick)


def _cls(name):
    from nodes import NODE_CLASS_MAPPINGS
    return NODE_CLASS_MAPPINGS.get(name)


def _patch(model, name, label, **kwargs):
    """Apply an optional model patch node; skip with a warning if its pack is missing."""
    if _cls(name) is None:
        log.warning("[StubeliusLTXModels] %s skipped: node '%s' not installed", label, name)
        return model
    return call_node(name, model=model, **kwargs)[0]


def _first_match(options, *needles):
    for needle in needles:
        for o in options:
            if needle in o.lower():
                return o
    return options[0] if options else ""


def _files(folder):
    try:
        return folder_paths.get_filename_list(folder)
    except Exception:
        return []


def _gguf(folder):
    return [f for f in _files(folder) if f.lower().endswith(".gguf")]


def _msr_default(loras):
    """An LTX 2.5 MSR LoRA when one is installed (Licon's LTX-2.5-Licon-MSR-V2 or V1). The 2.3 MSR LoRAs
    lack the slot-embedding tensors the 2.5 engine needs, so they are never picked."""
    for name in loras:
        low = os.path.basename(name).lower()
        if "msr" in low and ("2.5" in low or "2_5" in low or "25" in low.replace("2.5", "")):
            return name
    return "none"


def _resolve_lora(name):
    for candidate in (name, name.replace("\\", "/"), name.replace("/", "\\")):
        path = folder_paths.get_full_path("loras", candidate)
        if path:
            return path
    return None


def refine_sigmas(strength, steps):
    """Sigmas for the full-size refine: `steps` even steps from `strength` down to 0. The start is
    the sigma itself, not a scheduler 'denoise' (denoise 0.42 on linear_quadratic starts at 0.93)."""
    s = max(0.05, min(1.0, float(strength)))
    n = max(1, int(steps))
    return torch.FloatTensor([s * (1 - i / n) for i in range(n + 1)])


# ---------------------------------------------------------------- Setup

class StubeliusLTXSetup:
    """How the seeds are rendered. Every setting here is the real value used; each seed's sampler
    and scheduler are on the Seed Samplers box."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seeds": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip":
                    "How many full videos (with sound) to render. Pick one with WINNER on the Finish node "
                    "(WINNER 0 = hold after the seeds)."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip":
                    "Seed 1. Seeds 2-4 add 1000003 each. Keep it fixed between the seed hunt and the finish."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100, "tooltip":
                    "8 for the distilled model (the Stubelius remix has the distilled LoRA baked in)."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip":
                    "1.0 for the distilled model (the negative is skipped, twice as fast). Raise only with a "
                    "non-distilled checkpoint."}),
                "first_pass_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip":
                    "1.0 = the seeds render at the Director's full size and the winner is finished as rendered: "
                    "real HD detail. Below 1 = quicker, softer seeds; Finish then takes the winner back up with the "
                    "latent upscaler and the refine (x2 upscaler -> 0.5), which can't draw the detail a half-size "
                    "render never had."}),
            }
        }

    RETURN_TYPES = ("LTX_SETUP",)
    RETURN_NAMES = ("setup",)
    FUNCTION = "run"
    CATEGORY = "StubeliusLTX"

    def run(self, seeds, seed, steps, cfg, first_pass_scale):
        s = dict(seed_count=max(1, min(4, int(seeds))), seed=int(seed), steps=int(steps), cfg=float(cfg),
                 scale=float(first_pass_scale))
        log.info("[StubeliusLTXSetup] %s", s)
        return (s,)


# ---------------------------------------------------------------- Seed Samplers

class StubeliusLTXSeedSamplers:
    """The sampler and scheduler of each seed. Sits between Setup and Seeds; the slots past Setup's
    seed count are greyed out (js/stubelius_ltx_seed_samplers.js) and not used."""

    @classmethod
    def INPUT_TYPES(cls):
        samplers = list(comfy.samplers.KSampler.SAMPLERS)
        schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
        scheduler = DEFAULT_SCHEDULER if DEFAULT_SCHEDULER in schedulers else "simple"
        req = {"setup": ("LTX_SETUP", {"tooltip": "From Setup."})}
        for k in range(1, 5):
            req[f"seed_{k}_sampler"] = (samplers, {"default": DEFAULT_SAMPLER, "tooltip":
                f"Sampler for seed {k}. euler_ancestral is the official LTX 2.5 first pass."})
            req[f"seed_{k}_scheduler"] = (schedulers, {"default": scheduler, "tooltip":
                f"Scheduler for seed {k}. linear_quadratic at 8 steps is exactly the official distilled sigmas."})
        return {"required": req}

    RETURN_TYPES = ("LTX_SETUP",)
    RETURN_NAMES = ("setup",)
    FUNCTION = "run"
    CATEGORY = "StubeliusLTX"

    def run(self, setup, **slots):
        per_seed = [(slots[f"seed_{k}_sampler"], slots[f"seed_{k}_scheduler"]) for k in range(1, 5)]
        log.info("[StubeliusLTXSeedSamplers] %s", "  ".join(
            f"seed {k}: {s}/{c}" for k, (s, c) in enumerate(per_seed[:setup["seed_count"]], 1)))
        return (dict(setup, samplers=per_seed),)


# ---------------------------------------------------------------- Models

_BASE, _PATCHED, _AUX = {}, {}, {}
# the Models settings that change the patched model (the rest are files loaded on the side)
MODEL_KEYS = ("diffusion_model", "distill_lora", "distill_lora_strength", "extra_lora_1", "extra_lora_1_strength",
              "extra_lora_2", "extra_lora_2_strength", "attention", "memory_efficient_attention",
              "chunk_feed_forward")
ATTENTIONS = ["sage attention", "comfy kitchen attention", "comfyui default"]


def _model_key(cfg):
    return tuple((k, cfg[k]) for k in MODEL_KEYS)


class LTXModels:
    """Model bundle for Seeds and Finish. Everything loads once per file name and is cached."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.key = tuple(sorted(cfg.items()))
        self.model_key = _model_key(cfg)

    def _base(self):
        name = self.cfg["diffusion_model"]
        if name not in _BASE:
            log.info("[StubeliusLTXModels] loading %s", name)
            if name.lower().endswith(".gguf"):
                if _cls("UnetLoaderGGUF") is None:
                    raise RuntimeError("[StubeliusLTXModels] GGUF model selected but ComfyUI-GGUF is not installed")
                _BASE[name] = call_node("UnetLoaderGGUF", unet_name=name)[0]
            else:
                _BASE[name] = call_node("UNETLoader", unet_name=name, weight_dtype="default")[0]
        return _BASE[name]

    def model(self):
        """The checkpoint with the LoRAs and speed/memory patches applied (what the Director gets)."""
        if self.model_key in _PATCHED:
            return _PATCHED[self.model_key]
        c = self.cfg
        m = self._base()
        loras = [(c["distill_lora"], c["distill_lora_strength"]),
                 (c["extra_lora_1"], c["extra_lora_1_strength"]),
                 (c["extra_lora_2"], c["extra_lora_2_strength"])]
        for name, strength in loras:
            if name and name != "none" and strength != 0:
                path = _resolve_lora(name)
                if path is None:
                    raise FileNotFoundError(f"[StubeliusLTXModels] LoRA '{name}' not found in models/loras")
                m, _ = comfy.sd.load_lora_for_models(m, None, comfy.utils.load_torch_file(path, safe_load=True),
                                                     strength, 0)
        # Sage (KJ) and Comfy Kitchen (core) both set the model's attention override, so one or the other
        if c["attention"] == "sage attention":
            m = _patch(m, "PathchSageAttentionKJ", "sage attention", sage_attention="auto", allow_compile=False)
        elif c["attention"] == "comfy kitchen attention":
            import comfy.ldm.modules.attention as attention
            if attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE:
                m = _patch(m, "ModelAttentionBackend", "comfy kitchen attention", attention="comfy kitchen attention")
            else:
                log.warning("[StubeliusLTXModels] comfy kitchen attention skipped: this comfy-kitchen build has no "
                            "INT8 attention (Nvidia / AMD only); ComfyUI's default attention is used")
        if c["memory_efficient_attention"]:
            m = _patch(m, "LTX2MemoryEfficientSageAttentionPatch", "memory-efficient attention", triton_kernels=True)
        if c["chunk_feed_forward"]:
            m = _patch(m, "LTXVChunkFeedForward", "chunk feed-forward", chunks=2, dim_threshold=4096)
        _PATCHED[self.model_key] = m
        return m

    @property
    def ic_lora(self):
        """(name, strength) for the Director guides; "None" = off, as LTXDirectorGuideCS25 expects."""
        name = self.cfg["ic_lora"]
        return ("None", 0.0) if name == "none" else (name, float(self.cfg["ic_lora_strength"]))

    @property
    def msr_lora(self):
        """The MSR LoRA name as LTXDirectorGuideCS25 expects it ("None" = no LoRA). The guide loads it
        only when the Director hands over MSR references."""
        name = self.cfg["msr_lora"]
        return "None" if name == "none" else name

    @property
    def msr(self):
        """(msr_lora_name, msr_lora_strength) for the seeds' guide (the first pass)."""
        return self.msr_lora, float(self.cfg["msr_lora_strength"])

    def _aux(self, kind, name, loader, **kwargs):
        if (kind, name) not in _AUX:
            _AUX[(kind, name)] = call_node(loader, **kwargs)[0]
        return _AUX[(kind, name)]

    def clip(self):
        name = self.cfg["text_encoder"]
        loader = "CLIPLoaderGGUF" if name.lower().endswith(".gguf") else "CLIPLoader"
        return self._aux("clip", name, loader, clip_name=name, type="ltxv")

    def vae(self):
        return self._aux("vae", self.cfg["video_vae"], "VAELoader", vae_name=self.cfg["video_vae"])

    def audio_vae(self):
        return self._aux("vae", self.cfg["audio_vae"], "VAELoader", vae_name=self.cfg["audio_vae"])

    def upscaler(self):
        name = self.cfg["latent_upscaler"]
        return self._aux("upscaler", name, "LatentUpscaleModelLoader", model_name=name)

    def tiny_vae(self):
        name = self.cfg["live_preview"]
        return None if name == "off" else self._aux("vae", name, "VAELoader", vae_name=name)

    @property
    def tile_size(self):
        return int(self.cfg["decode_tile_size"])

    def preview(self, model, node_id, fps):
        """Attach the live sampling preview (KJ Model Preview Override with the tiny LTX VAE) for the
        node that samples. Cheap (clone + wrapper), so it's applied per run instead of being cached."""
        cls = _cls("ModelPreviewOverrideKJ")
        tiny = self.tiny_vae() if cls is not None else None
        if model is None or tiny is None:
            return model
        saved = getattr(cls, "hidden", None)
        cls.hidden = types.SimpleNamespace(unique_id=node_id)   # normally set by ComfyUI's executor
        try:
            return _unwrap(cls.execute(model=model, max_resolution=PREVIEW_MAX_RES, jpeg_quality=80,
                                       suppress_default_preview=True, preview_frames=257,
                                       preview_fps=max(1, round(fps)), vae=tiny, tiny_vae="none"))[0]
        finally:
            cls.hidden = saved


def _prune(cfg):
    """Drop cached models this config no longer references, so RAM follows the current setup."""
    for k in [k for k in _BASE if k != cfg["diffusion_model"]]:
        _BASE.pop(k)
    key = _model_key(cfg)
    for k in [k for k in _PATCHED if k != key]:
        _PATCHED.pop(k)
    keep = {("clip", cfg["text_encoder"]), ("vae", cfg["video_vae"]), ("vae", cfg["audio_vae"]),
            ("vae", cfg["live_preview"]), ("upscaler", cfg["latent_upscaler"])}
    for k in [k for k in _AUX if k not in keep]:
        _AUX.pop(k)


class StubeliusLTXModels:
    @classmethod
    def INPUT_TYPES(cls):
        unets = _files("diffusion_models") + _gguf("unet_gguf")
        tes = _files("text_encoders") + _gguf("clip_gguf")
        vaes = _files("vae")
        loras = ["none"] + _files("loras")
        upscalers = _files("latent_upscale_models") or ["none_found"]
        tiny = ["off"] + [v for v in vaes if os.path.basename(v).lower().startswith("taeltx")]
        return {
            "required": {
                # newest first: the latest remix beta (int8 before bf16), or distilled before dev
                "diffusion_model": (unets, {"default": _first_match(sorted(unets, key=str.lower, reverse=True),
                                                                    "stubelius_remix", "ltx-2.5", "ltx2.5"),
                    "tooltip": "LTX 2.5 checkpoint, safetensors or GGUF."}),
                "distill_lora": (loras, {"default": "none", "tooltip":
                    "Distilled LoRA, only for a non-distilled checkpoint (8 steps, cfg 1). The Stubelius "
                    "remix has it baked in: leave it at none."}),
                "distill_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "extra_lora_1": (loras, {"default": "none", "tooltip": "Applies to the seeds and the refine."}),
                "extra_lora_1_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "extra_lora_2": (loras, {"default": "none"}),
                "extra_lora_2_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "ic_lora": (loras, {"default": "none", "tooltip":
                    "IC-LoRA for the Director's motion / video guides (pose, depth, ...). none = off."}),
                "ic_lora_strength": ("FLOAT", {"default": 0.6, "min": -2.0, "max": 2.0, "step": 0.05}),
                "msr_lora": (loras, {"default": _msr_default(loras), "tooltip":
                    "LTX 2.5 MSR LoRA (Licon LTX-2.5-Licon-MSR-V2 or V1), its own slot next to the IC-LoRA. Used in "
                    "both passes whenever the Director's reference option is Licon MSR; ignored otherwise."}),
                "msr_lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05, "tooltip":
                    "MSR LoRA strength on the seeds (first pass). The refine's is on the Output node, so "
                    "changing it after a seed hunt re-runs only the finish."}),
                "text_encoder": (tes, {"default": _first_match(tes, "gemma4-12b-with-proj-ltx-2.5", "ltx-2.5", "gemma4")}),
                "video_vae": (vaes, {"default": _first_match(vaes, "ltx-2.5-video-vae")}),
                "audio_vae": (vaes, {"default": _first_match(vaes, "ltx-2.5-audio-vae")}),
                "latent_upscaler": (upscalers, {"default": _first_match(upscalers, "ltx-2.5-latent-spatial-upscaler-x2"),
                    "tooltip": "Takes the winner from the seed size to full size before the refine. "
                               "Match Setup's first-pass scale (x2 -> 0.5)."}),
                "attention": (ATTENTIONS, {"default": "sage attention", "tooltip":
                    "sage attention: SageAttention through KJNodes (auto kernel). comfy kitchen attention: "
                    "ComfyUI's own quantized INT8 attention (comfy-kitchen, Nvidia / AMD). comfyui default: no "
                    "patch, whatever ComfyUI was started with. They replace each other, so pick one."}),
                "memory_efficient_attention": ("BOOLEAN", {"default": False, "tooltip":
                    "KJ's memory-efficient LTX-2 attention (saves VRAM). It swaps each block's self-attention for "
                    "its own Sage kernel, whichever attention is picked above."}),
                "chunk_feed_forward": ("BOOLEAN", {"default": False, "tooltip": "Feed-forward in chunks (saves VRAM)."}),
                "decode_tile_size": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64, "tooltip":
                    "Largest spatial tile of the video decode, overlap a quarter of it (each clip is decoded "
                    "in one temporal pass). Bigger = fewer seams: 768 is Lightricks' spatial-only setting, "
                    "about 17 GB for 5 s at 1280x704; 512 about 8 GB. When a clip doesn't fit in the free VRAM "
                    "at this tile, the decode steps the tile down by itself (below 256: crossfaded chunks) "
                    "instead of spilling into system RAM."}),
                "live_preview": (tiny, {"default": _first_match(tiny, "taeltx2_5", "taeltx"), "tooltip":
                    "Tiny LTX VAE (models/vae) for the LIVE PREVIEW panel while Seeds and Finish sample. "
                    "off = ComfyUI's default preview on the sampling node."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "LTX_MODELS")
    RETURN_NAMES = ("model", "clip", "vae", "audio_vae", "models")
    FUNCTION = "run"
    CATEGORY = "StubeliusLTX"

    def run(self, **cfg):
        _prune(cfg)
        m = LTXModels(cfg)
        return (m.model(), m.clip(), m.vae(), m.audio_vae(), m)


# ---------------------------------------------------------------- Output

class StubeliusLTXOutput:
    """What comes out: chosen up front, used only by Finish. It never feeds Seeds, so changing it
    after a seed hunt re-runs only the finish."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refine_strength": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.01, "tooltip":
                    "The refine settings only apply when Setup's first-pass scale is below 1; full-size seeds are "
                    "finished as rendered. How far the refine of the upscaled winner may move it: the noise level "
                    "(sigma) it restarts from. 0.3-0.45 keeps the seed's look. Higher (Lightricks' two-stage "
                    "pipeline restarts at 0.91) changes the look but adds little detail: a half-size seed stays "
                    "soft either way."}),
                "refine_steps": ("INT", {"default": 4, "min": 1, "max": 12, "tooltip":
                    "Steps from refine_strength down to 0, evenly spaced."}),
                "refine_sampler": (list(comfy.samplers.KSampler.SAMPLERS), {"default": "euler", "tooltip":
                    "euler, as Lightricks' two-stage pipeline: ancestral noise only in the first pass (the "
                    "seeds), none in the refine."}),
                "refine_msr_lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "MSR LoRA strength in the refine (second pass); the LoRA itself and its first-pass "
                    "strength are on the Models node. Only used when the Director's reference option is Licon MSR."}),
                "refine_msr_reference": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "How hard the MSR reference images pull in the refine. ~0.4 holds the detail without the "
                    "references repainting the opening (stops second-pass mist / ghosting). 0 = the Director's "
                    "reference strength, which the seeds always use."}),
                "audio": (AUDIO_MODES, {"default": AUDIO_MODES[0], "tooltip":
                    "keep: the refine leaves the seed's sound exactly as you heard it."}),
                "final_resolution": (list(RESOLUTIONS), {"default": "1080p", "tooltip":
                    "Short side of the final video (portrait too). native = the Director's size after the refine. "
                    "4K holds about 100 MB of RAM per frame, so keep 4K clips short, especially at 48/60 fps."}),
                "final_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 1.0, "tooltip":
                    "0 = the Director's frame rate, untouched. Higher (48, 50, 60) = RIFE draws the in-between "
                    "frames; length and sync kept."}),
                "upscaler": (UPSCALERS, {"default": "RTX VSR", "tooltip":
                    "Used when the final resolution needs an upscale. DLSS5 adds far more detail; Color Lock "
                    "then restores the original colours. Needs ComfyUI-DLSS5-Enhancer."}),
            }
        }

    RETURN_TYPES = ("LTX_OUTPUT",)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "StubeliusLTX"

    def run(self, refine_strength, refine_steps, refine_sampler, refine_msr_lora_strength, refine_msr_reference,
            audio, final_resolution, final_fps, upscaler):
        o = dict(strength=float(refine_strength), steps=int(refine_steps), sampler=refine_sampler,
                 msr_lora_strength=float(refine_msr_lora_strength), msr_reference=float(refine_msr_reference),
                 audio_lock=(audio == AUDIO_MODES[0]), resolution=final_resolution, fps=float(final_fps),
                 upscaler=upscaler)
        log.info("[StubeliusLTXOutput] %s", o)
        return (o,)


# ---------------------------------------------------------------- Seeds

_TOKENS = itertools.count(1)


class StubeliusLTXSeeds:
    """Renders 1-4 full seeds with sound at the first-pass scale. The guides are built once (they
    don't depend on the seed); only the sampling varies per seed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "setup": ("LTX_SETUP",),
                "models": ("LTX_MODELS",),
                "model": ("MODEL", {"tooltip": "From the Director (model)."}),
                "positive": ("CONDITIONING", {"tooltip": "From the Director."}),
                "negative": ("CONDITIONING", {"tooltip": "From the Director."}),
                "video_latent": ("LATENT", {"tooltip": "From the Director."}),
                "audio_latent": ("LATENT", {"tooltip": "From the Director."}),
                "guide_data": ("GUIDE_DATA", {"tooltip": "From the Director."}),
                "frame_rate": ("FLOAT", {"forceInput": True, "tooltip": "From the Director."}),
            },
            "optional": {"motion_guide_data": ("MOTION_GUIDE_DATA", {"tooltip": "From the Director."})},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "AUDIO", "IMAGE", "AUDIO", "IMAGE", "AUDIO", "LTX_TAKES")
    RETURN_NAMES = ("seed_1", "seed_1_audio", "seed_2", "seed_2_audio", "seed_3", "seed_3_audio",
                    "seed_4", "seed_4_audio", "takes")
    FUNCTION = "run"
    CATEGORY = "StubeliusLTX"

    @_holds_run_models(lambda self, setup, models, model, *a, **k: model)
    def run(self, setup, models, model, positive, negative, video_latent, audio_latent, guide_data,
            frame_rate, motion_guide_data=None, unique_id=None):
        from comfy_execution.graph import ExecutionBlocker
        vae, audio_vae = models.vae(), models.audio_vae()
        # the conditioning the old graph built by hand: neutral negative, frame rate on both
        negative = call_node("ConditioningZeroOut", conditioning=negative)[0]
        positive, negative = call_node("LTXVConditioning", positive=positive, negative=negative,
                                       frame_rate=float(frame_rate))[:2]
        # msr_strength 0 = the Director's reference strength: the full pull the first pass wants
        pos_g, neg_g, lat_g, model_g, _ = _guide(positive, negative, vae, video_latent, guide_data,
                                                 motion_guide_data, model, *models.ic_lora, setup["scale"], 1.0, 0.0,
                                                 *models.msr)
        model_g = models.preview(model_g, unique_id, frame_rate)

        n = setup["seed_count"]
        # each seed's sampler/scheduler from the Seed Samplers box; the official pair when it's bypassed
        per_seed = setup.get("samplers") or [(DEFAULT_SAMPLER, DEFAULT_SCHEDULER)] * 4
        outs, latents, audios, seeds, decoded = [], [], [], [], []
        for k in range(n):
            seed = (setup["seed"] + k * SEED_STEP) % (1 << 64)
            sampler, scheduler = per_seed[k]
            log.info("[StubeliusLTXSeeds] seed %d/%d: %d  %s/%s  %d steps  cfg %.2f", k + 1, n, seed,
                     sampler, scheduler, setup["steps"], setup["cfg"])
            sigmas = call_node("BasicScheduler", model=model_g, scheduler=scheduler,
                               steps=setup["steps"], denoise=1.0)[0]
            video, audio = _sample_av(model_g, pos_g, neg_g, setup["cfg"], sampler, sigmas, seed,
                                      lat_g, audio_latent)
            images, audio_out, cropped = _decode(vae, audio_vae, pos_g, neg_g, video, audio,
                                                 models.tile_size, models.tile_size // 4)
            outs += [images, audio_out]
            latents.append(cropped["samples"])
            audios.append(audio["samples"])
            seeds.append(seed)
            decoded.append((images, audio_out))   # the same tensors as the previews: no extra memory
        outs += [ExecutionBlocker(None)] * (8 - len(outs))   # unused seeds: their previews are skipped

        takes = dict(token=next(_TOKENS), count=n, latents=latents, audio=audios, seeds=seeds,
                     samplers=per_seed[:n], cfg=setup["cfg"], model=model, positive=positive, negative=negative,
                     guide_data=guide_data, motion_guide_data=motion_guide_data, fps=float(frame_rate),
                     scale=setup["scale"], decoded=decoded)
        return (*outs, takes)


# ---------------------------------------------------------------- Finish

def _short(images):
    return min(images.shape[1], images.shape[2])


def _target_size(images, short_side):
    """(width, height) with the short side at `short_side` and the shape kept, both multiples
    of 8 (what RTX VSR outputs; video encoders like it too)."""
    h, w = images.shape[1], images.shape[2]
    k = short_side / min(h, w)
    return max(8, round(w * k / 8) * 8), max(8, round(h * k / 8) * 8)


def _fit(images, short_side, inplace=False, batch=8):
    """Bring an IMAGE batch to _target_size; no-op when it already fits. A frame at most 2% too
    big (DLSS5 gives 1920x1088 for 1080p) is centre-cropped rather than resampled; otherwise
    bicubic, antialiased. inplace=True (a private tensor only, e.g. Finish's own DLSS5 output)
    writes a smaller result back into the same memory, so no second full-size copy is made:
    frame i lands at or before where frame i was, never over a frame not yet read."""
    n, h, w, c = images.shape
    width, height = _target_size(images, short_side)
    if (w, h) == (width, height):
        return images
    shrink = width <= w and height <= h
    crop = shrink and width >= 0.98 * w and height >= 0.98 * h
    y0, x0 = (h - height) // 2, (w - width) // 2
    size = height * width * c
    reuse = inplace and shrink and images.is_contiguous() and images.device.type == "cpu"
    if reuse:
        flat = images.view(-1)
    else:
        out = torch.empty((n, height, width, c), dtype=images.dtype)
    import comfy.model_management
    device = comfy.model_management.get_torch_device()
    for i in range(0, n, batch):
        if crop:
            part = images[i:i + batch, y0:y0 + height, x0:x0 + width].clone()
        else:
            x = images[i:i + batch].to(device).movedim(-1, 1)
            x = torch.nn.functional.interpolate(x, size=(height, width), mode="bicubic", antialias=True,
                                                align_corners=False)
            part = x.clamp(0, 1).movedim(1, -1).to("cpu", images.dtype).contiguous()
        if reuse:
            flat[i * size:(i + part.shape[0]) * size].copy_(part.view(-1))
        else:
            out[i:i + part.shape[0]] = part
    return flat[:n * size].view(n, height, width, c) if reuse else out


def _vsr(images, short_side):
    """RTX VSR to the exact size; past its 4x per pass in two passes."""
    if short_side / _short(images) > VSR_MAX_SCALE:
        images = _vsr(images, max(8, round(short_side / 2 / 8) * 8))
    width, height = _target_size(images, short_side)
    return call_node("RTXVideoSuperResolution", images=images,
                     resize_type={"resize_type": "target dimensions", "width": width, "height": height},
                     quality="ULTRA")[0]


def _dlss5(images, need):
    """One DLSS5 pass at the smallest factor that reaches `need` (its largest, 3x, when more is
    needed), so the exact fit afterwards is a slight downscale."""
    factor = next((f for f in DLSS5_FACTORS if f >= need - 0.005), max(DLSS5_FACTORS))
    settings = call_node("DLSS5Settings", upscaling_mode=DLSS5_FACTORS[factor], nr_preset="Default",
                         nr_style="Default", nr_intensity=1.0, local_tone_strength=1.0,
                         local_structure_strength=1.5, skin_structure_strength=2.0, automatic_mask=True,
                         dlss_model_preset="M", motion="auto", scene_change_threshold=0.24, warmup_frames=0,
                         runtime_dir="")[0]
    return call_node("DLSS5EnhanceImages", images=images, settings=settings, verify_neural_rendering=True)[0]


def _ram_check(images, target, source_fps, fps):
    """The finished frames are held uncompressed (width x height x 12 bytes each). Say so up front
    when they alone won't fit in RAM, instead of letting the job crawl through the pagefile."""
    import psutil
    frames = round(images.shape[0] * max(1.0, fps / source_fps))
    width, height = _target_size(images, target) if target else (images.shape[2], images.shape[1])
    need, total = frames * width * height * 12, psutil.virtual_memory().total
    if need > 0.6 * total:
        log.warning("[StubeliusLTXFinish] %d frames at %dx%d need about %.0f GB of RAM on this %.0f GB "
                    "machine: expect a heavy slowdown. A shorter clip, a lower final FPS or a smaller "
                    "resolution avoids it.", frames, width, height, need / 1e9, total / 1e9)


_REFINE_MEMO = []   # [(key, (images, audio))], newest last, two kept


class StubeliusLTXFinish:
    """The engine: everything it does comes from the Output node. WINNER is the only choice left
    after a seed hunt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "takes": ("LTX_TAKES",),
                "models": ("LTX_MODELS",),
                "output": ("LTX_OUTPUT",),
                "winner": ("INT", {"default": 1, "min": 0, "max": 4, "tooltip":
                    "Which seed to finish. 0 = hold: render the seeds and their previews only, then "
                    "set 1-4 and re-queue (the seeds come from cache, only the finish runs)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "fps")
    FUNCTION = "run"
    CATEGORY = "StubeliusLTX"

    def run(self, takes, models, output, winner, unique_id=None):
        if int(winner) == 0:
            from comfy_execution.graph import ExecutionBlocker
            log.info("[StubeliusLTXFinish] WINNER 0 = hold: %d seed(s) rendered, finish skipped. "
                     "Set WINNER 1-%d and re-queue.", takes["count"], takes["count"])
            return (ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None))
        w = max(1, min(int(winner), takes["count"]))
        if w != winner:
            log.warning("[StubeliusLTXFinish] WINNER %d but only %d seed(s) rendered, using %d",
                        winner, takes["count"], w)
        o = output
        full_size = takes.get("scale", 0.5) >= 0.999
        if full_size:
            # rendered at the Director's size already: a latent upscale + refine would only blur it
            images, audio = takes["decoded"][w - 1]
            log.info("[StubeliusLTXFinish] seeds are full size: winner %d finished as rendered (no latent upscale "
                     "or refine)", w)
        else:
            images, audio = self._refine(takes, models, w, o, unique_id)
        target = RESOLUTIONS.get(o["resolution"])
        source_fps = takes["fps"]
        _ram_check(images, target, source_fps, max(source_fps, o["fps"]))
        if target and _short(images) > target:
            images = _fit(images, target)   # smaller than the render: downscale before RIFE (less work)

        fps = source_fps
        if o["fps"] > source_fps + 1e-6:
            from .stubelius_rife_fps import StubeliusLTXRIFEToFPS
            images, fps = StubeliusLTXRIFEToFPS().run(images, source_fps, o["fps"], "rife47.pth", True, True, 8)

        upscaled = bool(target and target / _short(images) > 1.02)
        if upscaled:
            images = self._upscale(images, target, o["upscaler"])
        if target:
            images = _fit(images, target)   # exact size (DLSS5 only scales by fixed factors)
        sampler, scheduler = takes.get("samplers", [("?", "?")] * w)[w - 1]
        refine = "as rendered" if full_size else f"refine {o['strength']:.2f} x{o['steps']}"
        log.info("[StubeliusLTXFinish] winner %d (seed %d, %s/%s), %s, %s -> %dx%d @ %.3g fps, upscaler %s",
                 w, takes["seeds"][w - 1], sampler, scheduler, refine, o["resolution"], images.shape[2],
                 images.shape[1], fps, o["upscaler"] if upscaled else "none")
        return (images, audio, float(fps))

    @staticmethod
    @_holds_run_models(lambda takes, *a, **k: takes["model"])
    def _refine(takes, models, w, o, node_id=None):
        """Latent upscale + guides at full size + the tail re-noise, the seed's audio locked. The MSR
        LoRA (when the Director is in Licon MSR mode) comes back with the second-pass strengths."""
        strength, steps, sampler, audio_lock = o["strength"], o["steps"], o["sampler"], o["audio_lock"]
        key = (takes["token"], w, strength, steps, sampler, audio_lock, o["msr_lora_strength"], o["msr_reference"],
               models.key)
        for k, v in _REFINE_MEMO:
            if k == key:
                log.info("[StubeliusLTXFinish] refine reused from memory (winner %d)", w)
                return v
        vae, audio_vae = models.vae(), models.audio_vae()
        upsampled = call_node("LTXVLatentUpsampler", samples={"samples": takes["latents"][w - 1]},
                              upscale_model=models.upscaler(), vae=vae)[0]
        _free_vram("post-upscale")
        pos, neg, lat, model, _ = _guide(takes["positive"], takes["negative"], vae, upsampled, takes["guide_data"],
                                         takes["motion_guide_data"], takes["model"], *models.ic_lora, 1.0, 1.0,
                                         o["msr_reference"], models.msr_lora, o["msr_lora_strength"])
        model = models.preview(model, node_id, takes["fps"])
        sigmas = refine_sigmas(strength, steps)
        msr = (f", MSR LoRA {o['msr_lora_strength']:g} reference {o['msr_reference']:g} (if the Director is in "
               f"Licon MSR mode)" if models.msr_lora != "None" else "")
        log.info("[StubeliusLTXFinish] refining winner %d: sigmas %s, %s cfg %.2f, audio %s%s", w,
                 [round(float(s), 4) for s in sigmas], sampler, takes["cfg"],
                 "locked" if audio_lock else "re-rendered", msr)
        video, audio = _sample_av(model, pos, neg, takes["cfg"], sampler, sigmas, takes["seeds"][w - 1],
                                  lat, {"samples": takes["audio"][w - 1]}, audio_lock=audio_lock)
        _free_vram("pre-decode")
        images, audio_out, _ = _decode(vae, audio_vae, pos, neg, video, audio, models.tile_size, models.tile_size // 4)
        _REFINE_MEMO.append((key, (images, audio_out)))
        del _REFINE_MEMO[:-2]
        return images, audio_out

    @staticmethod
    def _upscale(images, short_side, upscaler):
        if upscaler == "RTX VSR":
            if _cls("RTXVideoSuperResolution") is None:
                raise RuntimeError("[StubeliusLTXFinish] 'RTX VSR' needs the Nvidia RTX nodes installed "
                                   "(Comfy-Org/Nvidia_RTX_Nodes_ComfyUI). Pick 'DLSS5 + Color Lock' or install them.")
            return _vsr(images, short_side)
        if _cls("DLSS5EnhanceImages") is None:
            raise RuntimeError("[StubeliusLTXFinish] 'DLSS5 + Color Lock' needs ComfyUI-DLSS5-Enhancer installed. "
                               "Pick 'RTX VSR' or install it.")
        try:
            enhanced = _dlss5(images, short_side / _short(images))
        except RuntimeError as e:
            if "install_runtime.py" not in str(e):
                raise
            raise RuntimeError(f"{e}\n\n[StubeliusLTXFinish] install_runtime.py's own download is broken (404); "
                               f"give it the v3.0 release:\n    install_runtime.py --url {DLSS5_RUNTIME_URL}\n"
                               "then restart ComfyUI. The later Visual Enhancer releases don't have the files "
                               "the node runs on.") from e
        rest = short_side / _short(enhanced)
        if rest > 1.02:
            # past DLSS5's 3x: RTX VSR takes it the rest of the way, or a second DLSS5 pass when VSR
            # isn't installed
            enhanced = _vsr(enhanced, short_side) if _cls("RTXVideoSuperResolution") else _dlss5(enhanced, rest)
        from .stubelius_color_lock import StubeliusLTXColorLock
        # The upscaled frames are ours alone (unless DLSS5 handed back its input): Color Lock and
        # the exact fit then work in place, so the frames exist once, not three times.
        private = enhanced.untyped_storage().data_ptr() != images.untyped_storage().data_ptr()
        enhanced = StubeliusLTXColorLock().run(enhanced, images, 1.0, 16, inplace=private)[0]
        return _fit(enhanced, short_side, inplace=private)


# ---------------------------------------------------------------- theme and live preview
# Display-only nodes, drawn by the pack's JavaScript. Their names are this pack's own, so it installs
# next to other packs that ship a theme or preview node of their own.

THEMES = ["Studio slate", "Stuubzzz neon", "Film stock", "Paper light", "Midnight blueprint", "ComfyUI default"]


class StubeliusLTXTheme:
    """Look of THIS workflow only (js/stubelius_ltx_theme.js). Saved in the workflow; never runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"theme": (THEMES, {"default": "Studio slate", "tooltip":
            "Applies to this workflow only, while its tab is open. Other workflows keep their look."})}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "StubeliusLTX"

    def run(self, theme):
        return ()


class StubeliusLTXLivePreview:
    """Display-only panel for the live sampling preview (see js/stubelius_ltx_live_preview.js).
    No inputs or outputs; it never runs as part of the prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "StubeliusLTX"

    def run(self):
        return ()


NODE_CLASS_MAPPINGS = {
    "StubeliusLTXSetup": StubeliusLTXSetup,
    "StubeliusLTXSeedSamplers": StubeliusLTXSeedSamplers,
    "StubeliusLTXModels": StubeliusLTXModels,
    "StubeliusLTXOutput": StubeliusLTXOutput,
    "StubeliusLTXSeeds": StubeliusLTXSeeds,
    "StubeliusLTXFinish": StubeliusLTXFinish,
    "StubeliusLTXTheme": StubeliusLTXTheme,
    "StubeliusLTXLivePreview": StubeliusLTXLivePreview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StubeliusLTXSetup": "Stubelius LTX Setup",
    "StubeliusLTXSeedSamplers": "Stubelius LTX Seed Samplers",
    "StubeliusLTXModels": "Stubelius LTX Models",
    "StubeliusLTXOutput": "Stubelius LTX Output",
    "StubeliusLTXSeeds": "Stubelius LTX Seeds",
    "StubeliusLTXFinish": "Stubelius LTX Finish",
    "StubeliusLTXTheme": "Stubelius LTX Theme (this workflow)",
    "StubeliusLTXLivePreview": "Stubelius LTX Live Preview",
}
