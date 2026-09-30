"""Stubelius LTX — the older Seed Hunt + Refine dashboards for the LTX 2.5 Director (CS),
and the helpers they share with the Ultimate LTX nodes.

The Ultimate LTX nodes (stubelius_ltx_pipeline.py) replace these two dashboards. Both stay
registered, so workflows built with them keep loading:

  * StubeliusLTXSeedHunt — up to 4 full first-pass candidates, each with its own
    seed / sampler / scheduler / steps / cfg. Guides are built ONCE (they don't
    depend on seed) and only the sampling varies per slot. Each candidate latent
    carries its settings + its audio latent, so the Refine auto-syncs.

  * StubeliusLTXRefine — pick a candidate (lazy inputs: un-run slots can never
    block execution), latent-upsample it, rebuild guides at full scale, and run
    the CS-style tail re-noise pass. Audio lock keeps the exact audio you
    auditioned; sigmas default to the CS manual schedule (0.85 → 0).

The pipeline takes from here: call_node, the guide and sampling helpers, the video decode
that sizes itself to the free VRAM, and the hold on a cancelled run's model clones.

All third-party node calls are resolved from NODE_CLASS_MAPPINGS at execution
time and kwargs are filtered against each node's live INPUT_TYPES, so version
drift in the CS / LTXVideo packs cannot break the calls.
"""

import functools
import gc
import json
import logging

import torch

log = logging.getLogger(__name__)

STUB_SETTINGS = "_stubelius_ltx_settings"
STUB_AUDIO = "_stubelius_ltx_audio"
CS_DEFAULT_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _mappings():
    from nodes import NODE_CLASS_MAPPINGS
    return NODE_CLASS_MAPPINGS


def _sentinel_latent():
    """Inert placeholder emitted on the candidate output of DISABLED hunt slots.
    ExecutionBlocker on a wired input prunes the consumer node before execute()
    is ever called (confirmed against the ComfyUI 0.33 executor) - lazy inputs
    can't save it because the blocker is produced in the same pass. A tiny real
    latent with a marker flows harmlessly instead; the Refine ignores un-picked
    slots and blocks cleanly only if the PICKED slot is a sentinel."""
    import torch
    return {"samples": torch.zeros(1, 1, 1, 1, 1), "_stub_disabled": True}


def _blocker():
    from comfy_execution.graph import ExecutionBlocker
    return ExecutionBlocker(None)


def _unwrap(res):
    args = getattr(res, "args", None)
    if args is not None:
        return args
    if isinstance(res, dict) and "result" in res:
        return res["result"]
    return res


def call_node(name, **kw):
    """Call a registered node (v1 or v3), filtering kwargs to its live schema."""
    cls = _mappings().get(name)
    if cls is None:
        raise RuntimeError(f"[StubeliusLTX] required node '{name}' is not installed/loaded.")
    try:
        spec = cls.INPUT_TYPES()
        allowed = set()
        for sec in ("required", "optional"):
            allowed |= set(spec.get(sec, {}) or {})
        kw = {k: v for k, v in kw.items() if k in allowed}
    except Exception:
        pass
    fn = getattr(cls, "FUNCTION", None)
    if fn and fn != "EXECUTE_NORMALIZED":
        return _unwrap(getattr(cls(), fn)(**kw))
    return _unwrap(cls.execute(**kw))


def _coerce_int(v, d):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return int(d)


def _coerce_float(v, d):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(d)


def _coerce_bool(v, d):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "none"):
            return bool(d)
        return s not in ("0", "false", "no", "off")
    return bool(d)


def _parse_sigmas(text):
    vals = [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError(f"[StubeliusLTX] need at least 2 sigma values, got: {text!r}")
    return torch.FloatTensor(vals)


def _samplers():
    import comfy.samplers
    return comfy.samplers.KSampler.SAMPLERS


def _schedulers():
    import comfy.samplers
    return comfy.samplers.KSampler.SCHEDULERS


def _guide(positive, negative, vae, latent, guide_data, motion_guide_data, model,
           ic_lora_name, ic_lora_strength, scale_by, image_attention_strength, msr_strength,
           msr_lora_name="None", msr_lora_strength=1.0):
    return call_node(
        "LTXDirectorGuideCS25",
        positive=positive, negative=negative, vae=vae, latent=latent,
        guide_data=guide_data, motion_guide_data=motion_guide_data, model=model,
        ic_lora_name=ic_lora_name, ic_lora_strength=ic_lora_strength,
        scale_by=scale_by, upscale_method="bicubic",
        image_attention_strength=image_attention_strength,
        crop="center", auto_snap_ic_grid=True, msr_strength=msr_strength,
        msr_lora_name=msr_lora_name, msr_lora_strength=msr_lora_strength,
    )


def _sample_av(model, positive, negative, cfg, sampler_name, sigmas, seed,
               video_latent, audio_latent, audio_lock=False):
    guider = call_node("CFGGuider", model=model, positive=positive, negative=negative, cfg=cfg)[0]
    sampler = call_node("KSamplerSelect", sampler_name=sampler_name)[0]
    noise = call_node("RandomNoise", noise_seed=seed)[0]
    combined = call_node("LTXVConcatAVLatent", video_latent=video_latent, audio_latent=audio_latent)[0]
    if audio_lock:
        try:
            import comfy.nested_tensor
            # keep the guide's video mask (reference keyframes stay clean); freeze only the audio
            v_mask = video_latent.get("noise_mask")
            if v_mask is None:
                v_mask = torch.ones_like(video_latent["samples"])
            a = audio_latent["samples"]
            combined = dict(combined)
            combined["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (v_mask, torch.zeros_like(a)))
            log.info("[StubeliusLTX] audio lock: candidate audio frozen through pass 2.")
        except Exception as e:  # noqa: BLE001 - lock is best-effort, never fatal
            log.warning("[StubeliusLTX] audio lock unavailable (%s) - audio will re-render.", e)
    out = call_node("SamplerCustomAdvanced", noise=noise, guider=guider,
                    sampler=sampler, sigmas=sigmas, latent_image=combined)[0]
    video, audio = call_node("LTXVSeparateAVLatent", av_latent=out, latent=out, samples=out)[:2]
    return video, audio


_HELD = []   # model clones of the last cancelled or failed run, see _hold_run_models()


def _hold_run_models(*roots):
    """Keep a cancelled or failed run's model clones alive until the next run starts.

    ComfyUI hands the exception back up its executor and keeps it in a reference cycle, and the
    run's frames go with it. The clones made during the run (the guide's IC-LoRA and MSR LoRA
    clones, the per-run live-preview clone) are then freed by the garbage collector in one pass.
    ComfyUI follows a loaded model from a freed clone to its parent, and loses it when both go in
    the same pass: "memory leak with model LTXAV" and a full garbage collect on every later model
    load, until a restart.

    Nothing is freed here: ComfyUI still cleans up after the exception, and freeing a run's
    memory before that can take ComfyUI down. This only keeps each loaded clone's chain back to
    `roots` (the run's input models) alive."""
    import comfy.model_management as mm
    roots = [r for r in roots if r is not None]
    for loaded in list(mm.current_loaded_models):
        chain, patcher = [], loaded.model
        while patcher is not None:
            chain.append(patcher)
            if any(patcher is r for r in roots):
                _HELD.extend(chain)
                break
            patcher = getattr(patcher, "parent", None)


def _release_held_models():
    """Drop what _hold_run_models() kept, at the start of the next run. Collect first, while it
    is still held, so the cancelled run's frames are gone and the clones then go one at a time,
    each handing ComfyUI's tracking on to its parent."""
    if _HELD:
        gc.collect()
        _HELD.clear()


def _holds_run_models(root):
    """For a node method that samples with per-run clones of a model: release what the last
    cancelled run left, and hold this run's clones if it is cancelled or fails. `root` picks
    the run's input model out of the method's arguments."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _release_held_models()
            try:
                return fn(*args, **kwargs)
            except BaseException:
                _hold_run_models(root(*args, **kwargs))
                raise
        return wrapper
    return deco


def _free_vram(reason=""):
    """Aggressively release VRAM between heavy stages - the 2x upscale + tiled decode
    otherwise spikes because the hunt's decoded candidates and pass-2 intermediates
    stay resident simultaneously."""
    try:
        import gc
        import torch
        import comfy.model_management as mm
        gc.collect()
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        log.info("[StubeliusLTX] freed VRAM%s.", f" ({reason})" if reason else "")
    except Exception as e:  # noqa: BLE001
        log.warning("[StubeliusLTX] VRAM free failed: %s", e)


WHOLE_CLIP = 4096       # VAEDecodeTiled's largest temporal_size: every frame in one pass
MIN_DECODE_TILE = 256   # px; smaller spatial tiles are only tried in temporal chunks
DECODE_TILE_STEP = 64   # px, the Models node's decode_tile_size step
MIN_CHUNK = 3           # latent frames: the 2-frame crossfade overlap plus one
GB = 1024 ** 3


def _is_oom(e):
    import comfy.model_management as mm
    is_oom = getattr(mm, "is_oom", None)   # also catches the AcceleratorError form of newer torch
    return is_oom(e) if is_oom else isinstance(e, mm.OOM_EXCEPTION)


def _decode_need(vae, shape, frames, tile):
    """ComfyUI's own estimate of one tile's decode memory: VAEDecodeTiled with this tile and
    temporal_size, on a latent of `shape`. ComfyUI sizes every decode's model unloading with it;
    for the LTX 2.5 VAE it runs about 10% above the measured peak."""
    s, tc = vae.spacial_compression_decode(), vae.temporal_compression_decode()
    _, c, t, h, w = shape
    n = max(1, tile // s)
    return vae.memory_used_decode((1, c, min(t, max(2, frames // tc)), min(h, n), min(w, n)), vae.vae_dtype)


def _decode_budget(vae, shape, tile_size):
    """Bytes of VRAM the decode may use, or None when the device doesn't report it (CPU,
    DirectML): what's free once ComfyUI has made room for the whole-clip decode (it unloads or
    evicts other models, as for any decode), minus the VRAM reserved for other apps."""
    import comfy.model_management as mm
    device = getattr(vae, "device", None)
    if getattr(device, "type", "cpu") == "cpu" or mm.is_directml_enabled():
        return None
    try:
        need = _decode_need(vae, shape, WHOLE_CLIP, tile_size)
        mm.load_models_gpu([vae.patcher], memory_required=need, force_full_load=getattr(vae, "disable_offload", False))
        budget = vae.patcher.get_free_memory(device) - mm.extra_reserved_memory()
    except Exception as e:  # noqa: BLE001 - no estimate: the plain one-pass decode, as before
        log.warning("[StubeliusLTX] free VRAM unknown (%s): decoding every frame in one pass.", e)
        return None
    out = getattr(vae, "output_device", None)
    if getattr(out, "type", "cpu") != "cpu":
        # --gpu-only: the tiler assembles the whole video in VRAM (float32 RGB + a weight channel)
        _, _, t, h, w = shape
        s, tc = vae.spacial_compression_decode(), vae.temporal_compression_decode()
        budget -= 16 * ((t - 1) * tc + 1) * h * s * w * s
    return budget


def _decode_plan(vae, shape, tile_size, budget):
    """(tile_size, temporal_size) for VAEDecodeTiled. Every frame in one temporal pass (no seams
    in time) at the largest spatial tile, from tile_size down to MIN_DECODE_TILE, whose estimate
    fits the budget. Past that, the longest crossfaded temporal chunks that fit at the smallest
    tile, with the tile grown back while it still fits."""
    if budget is None:
        return tile_size, WHOLE_CLIP
    tiles = list(range(tile_size, MIN_DECODE_TILE - 1, -DECODE_TILE_STEP)) or [tile_size]
    for tile in tiles:
        if _decode_need(vae, shape, WHOLE_CLIP, tile) <= budget:
            return tile, WHOLE_CLIP
    tile, tc = tiles[-1], vae.temporal_compression_decode()
    chunk = min(shape[2], WHOLE_CLIP // tc)
    while chunk > MIN_CHUNK and _decode_need(vae, shape, chunk * tc, tile) > budget:
        chunk -= 1
    while tile + DECODE_TILE_STEP <= tile_size and _decode_need(vae, shape, chunk * tc, tile + DECODE_TILE_STEP) <= budget:
        tile += DECODE_TILE_STEP
    return tile, chunk * tc


def _log_plan(vae, shape, tile_size, budget, tile, frames):
    if budget is None:
        return
    tc = vae.temporal_compression_decode()
    n = (shape[2] - 1) * tc + 1
    need, full = _decode_need(vae, shape, frames, tile) / GB, _decode_need(vae, shape, WHOLE_CLIP, tile_size) / GB
    if (tile, frames) == (tile_size, WHOLE_CLIP):
        log.info("[StubeliusLTX] decode: %d frames in one pass at %d px tiles, ~%.1f GB of %.1f GB free VRAM.",
                 n, tile, need, budget / GB)
    elif frames == WHOLE_CLIP:
        log.info("[StubeliusLTX] decode: %d frames at %d px tiles would need ~%.1f GB, %.1f GB VRAM is free - "
                 "one pass at %d px tiles instead (~%.1f GB).", n, tile_size, full, budget / GB, tile, need)
    else:
        log.warning("[StubeliusLTX] decode: %d frames don't fit %.1f GB of free VRAM in one pass even at %d px tiles - "
                    "%d-frame chunks with a crossfade at %d px tiles instead (~%.1f GB).", n, budget / GB,
                    min(tile_size, MIN_DECODE_TILE), frames - tc + 1, tile, need)
    if need > budget / GB:
        log.warning("[StubeliusLTX] decode: even the smallest decode needs ~%.1f GB and only %.1f GB VRAM is free - "
                    "it may spill into system RAM and crawl. Close other apps using the GPU.", need, budget / GB)


def _decode_video(vae, samples, tile_size, tile_overlap):
    """Spatially tiled, temporally whole. The LTX 2.5 video VAE decodes with a diffusion
    decoder, and ComfyUI's generic tiler renders each temporal chunk as a standalone clip
    (own noise, chunk ends treated as clip ends) - with the old 8-frame overlap that was a
    1-frame blend and showed as a detail pop every 56 frames. Chunks with a 16-frame
    overlap (9-frame crossfade, the official template setting) are only the last resort.

    Sized before it starts, from ComfyUI's memory estimate and the VRAM that's actually free,
    rather than by catching an out-of-memory error: on Windows the NVIDIA driver's default
    sysmem fallback lets allocations past the card's VRAM spill into system RAM instead of
    failing, so the error never comes - the decode crawls, and a long clip can reset the
    driver and take ComfyUI down with it. The out-of-memory retry stays for estimates that
    fall short."""
    lat = samples["samples"]
    if getattr(lat, "is_nested", False):
        lat = lat.unbind()[0]
    if lat.ndim != 5 or not vae.temporal_compression_decode():
        return call_node("VAEDecodeTiled", vae=vae, samples=samples, tile_size=tile_size,
                         overlap=tile_overlap, temporal_size=WHOLE_CLIP, temporal_overlap=16)[0]
    budget = _decode_budget(vae, lat.shape, tile_size)
    tile, frames = _decode_plan(vae, lat.shape, tile_size, budget)
    _log_plan(vae, lat.shape, tile_size, budget, tile, frames)
    for attempt in range(3):
        try:
            return call_node("VAEDecodeTiled", vae=vae, samples=samples, tile_size=tile,
                             overlap=min(tile_overlap, tile // 4), temporal_size=frames, temporal_overlap=16)[0]
        except Exception as e:  # noqa: BLE001
            if attempt == 2 or not _is_oom(e):
                raise
        # outside the except block, so the failed attempt's tensors can be freed first
        _free_vram("decode OOM retry")
        failed = tile
        tile, frames = _decode_plan(vae, lat.shape, tile_size, _decode_need(vae, lat.shape, frames, tile) / 2)
        chunks = "" if frames == WHOLE_CLIP else f", {frames - vae.temporal_compression_decode() + 1}-frame chunks"
        log.warning("[StubeliusLTX] decode ran out of VRAM at %d px tiles - retrying with half the memory: %d px "
                    "tiles%s.", failed, tile, chunks)


def _decode(vae, audio_vae, positive, negative, video_latent, audio_latent,
            tile_size, tile_overlap):
    pos_c, neg_c, cropped = call_node("LTXDirectorCropGuidesCS25",
                                      positive=positive, negative=negative, latent=video_latent)[:3]
    images = _decode_video(vae, cropped, tile_size, tile_overlap)
    audio = call_node("LTXVAudioVAEDecode", vae=audio_vae, audio_vae=audio_vae,
                      samples=audio_latent, latent=audio_latent)[0]
    return images, audio, cropped


# --------------------------------------------------------------------------- #
# Seed Hunt dashboard
# --------------------------------------------------------------------------- #

class StubeliusLTXSeedHunt:
    """Full first-pass seed hunt: up to 4 candidates, per-slot sampling settings."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        loras = folder_paths.get_filename_list("loras") or ["none_found"]
        req = {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "audio_vae": ("VAE",),
            "latent": ("LATENT", {"tooltip": "Video latent from the Director (out: latent)."}),
            "audio_latent": ("LATENT", {"tooltip": "Audio latent from the Director (out: audio latent)."}),
            "guide_data": ("GUIDE_DATA",),
            "ic_lora_name": (["None"] + loras, {"default": "None"}),
            "ic_lora_strength": ("FLOAT", {"default": 0.6, "min": -100.0, "max": 100.0, "step": 0.01}),
            "scale_by": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 2.0, "step": 0.01,
                                   "tooltip": "First-pass scale. 0.5 recommended for IC-LoRA."}),
            "image_attention_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "msr_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "msr_lora_name": (["None"] + loras, {"default": "None", "tooltip": "LTX-2.5 MSR LoRA (Licon V1). Required when the Director ref option is Licon MSR."}),
            "msr_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            "tile_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 32}),
            "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
        }
        req["slots_json"] = ("STRING", {"multiline": True, "default": '[{"enable": true, "seed": 42, "sampler": "euler", "scheduler": "linear_quadratic", "steps": 12, "cfg": 1.0}, {"enable": false, "seed": 1000045, "sampler": "euler", "scheduler": "linear_quadratic", "steps": 12, "cfg": 1.0}, {"enable": false, "seed": 2000048, "sampler": "euler", "scheduler": "linear_quadratic", "steps": 12, "cfg": 1.0}, {"enable": false, "seed": 3000051, "sampler": "euler", "scheduler": "linear_quadratic", "steps": 12, "cfg": 1.0}]',
            "tooltip": "Per-slot hunt config (JSON) - normally edited via the 2x2 panel above. "
            "Hand-editable fallback: list of up to 4 objects with enable/seed/sampler/scheduler/steps/cfg."})
        return {"required": req, "optional": {"motion_guide_data": ("MOTION_GUIDE_DATA",)}}

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT", "IMAGE", "AUDIO", "LATENT",
                    "IMAGE", "AUDIO", "LATENT", "IMAGE", "AUDIO", "LATENT", "STRING")
    RETURN_NAMES = ("images_1", "audio_1", "candidate_1", "images_2", "audio_2", "candidate_2",
                    "images_3", "audio_3", "candidate_3", "images_4", "audio_4", "candidate_4", "info")
    FUNCTION = "hunt"
    CATEGORY = "StubeliusLTX"

    @_holds_run_models(lambda self, model, *a, **k: model)
    def hunt(self, model, positive, negative, vae, audio_vae, latent, audio_latent,
             guide_data, ic_lora_name, ic_lora_strength, scale_by,
             image_attention_strength, msr_strength, msr_lora_name, msr_lora_strength,
             tile_size, tile_overlap, slots_json="", motion_guide_data=None, **_legacy):

        try:
            cfgs = json.loads(slots_json) if str(slots_json).strip() else []
            assert isinstance(cfgs, list)
        except Exception:
            log.warning("[StubeliusLTX] slots_json invalid - falling back to slot 1 defaults.")
            cfgs = []
        while len(cfgs) < 4:
            cfgs.append({"enable": len(cfgs) == 0})

        # guides are seed-independent: build once, share across all slots
        pos_g, neg_g, lat_g, model_g, _ = _guide(
            positive, negative, vae, latent, guide_data, motion_guide_data, model,
            ic_lora_name, _coerce_float(ic_lora_strength, 0.6), _coerce_float(scale_by, 0.5),
            _coerce_float(image_attention_strength, 1.0), _coerce_float(msr_strength, 0.0),
            msr_lora_name, _coerce_float(msr_lora_strength, 1.0))

        outs, ran = [], []
        for s in (1, 2, 3, 4):
            c = cfgs[s - 1] if isinstance(cfgs[s - 1], dict) else {}
            if not _coerce_bool(c.get("enable"), s == 1):
                outs.extend([_blocker(), _blocker(), _sentinel_latent()])
                continue
            seed = _coerce_int(c.get("seed"), 42 + (s - 1) * 1000003)
            sampler_name = str(c.get("sampler") or "euler")
            scheduler = str(c.get("scheduler") or "simple")
            steps = _coerce_int(c.get("steps"), 12)
            cfg = _coerce_float(c.get("cfg"), 1.0)
            log.info("[StubeliusLTX] hunt slot %d: seed=%d %s/%s steps=%d cfg=%.2f",
                     s, seed, sampler_name, scheduler, steps, cfg)
            sigmas = call_node("BasicScheduler", model=model_g, scheduler=scheduler,
                               steps=steps, denoise=1.0)[0]
            video, audio = _sample_av(model_g, pos_g, neg_g, cfg, sampler_name,
                                      sigmas, seed, lat_g, audio_latent)
            images, audio_out, cropped = _decode(vae, audio_vae, pos_g, neg_g,
                                                 video, audio, tile_size, tile_overlap)
            cand = dict(cropped)
            cand[STUB_AUDIO] = audio["samples"]
            cand[STUB_SETTINGS] = {
                "seed": seed, "sampler_name": sampler_name, "scheduler": scheduler,
                "steps": steps, "cfg": cfg, "ic_lora_name": ic_lora_name,
                "ic_lora_strength": _coerce_float(ic_lora_strength, 0.6),
                "image_attention_strength": _coerce_float(image_attention_strength, 1.0),
            }
            outs.extend([images, audio_out, cand])
            ran.append(s)

        info = "Stubelius LTX hunt: slots " + (", ".join(map(str, ran)) or "none") + " rendered."
        return (*outs, info)


# --------------------------------------------------------------------------- #
# Refine dashboard
# --------------------------------------------------------------------------- #

class StubeliusLTXRefine:
    """Pass 2: latent-upsample the picked candidate, rebuild guides at full
    scale, tail re-noise (CS manual sigmas by default), audio locked."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        loras = folder_paths.get_filename_list("loras") or ["none_found"]
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "guide_data": ("GUIDE_DATA",),
                "candidate_1": ("LATENT", {"lazy": True}),
                "candidate": ("INT", {"default": 0, "min": 0, "max": 4,
                              "tooltip": "0 = WAIT (hunt only, refine blocks). Set 1-4 after reviewing candidates, then re-queue."}),
                "sync_from_hunt": ("BOOLEAN", {"default": True}),
                "audio_mode": (["keep candidate audio (locked)", "regenerate in pass 2"],
                               {"default": "keep candidate audio (locked)"}),
                "sigma_mode": (["manual sigmas (CS)", "scheduler"], {"default": "manual sigmas (CS)"}),
                "manual_sigmas": ("STRING", {"default": CS_DEFAULT_SIGMAS}),
                "polish_steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "refine_denoise": ("FLOAT", {"default": 0.42, "min": 0.05, "max": 1.0, "step": 0.01}),
                "sampler_name": (_samplers(), {"default": "euler"}),
                "scheduler": (_schedulers(), {"default": "linear_quadratic" if "linear_quadratic" in _schedulers() else "simple"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "ic_lora_name": (["None"] + loras, {"default": "None"}),
                "ic_lora_strength": ("FLOAT", {"default": 0.6, "min": -100.0, "max": 100.0, "step": 0.01}),
                "image_attention_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "msr_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                                 "tooltip": "0.4 recommended on the refine stage (per CS guidance)."}),
                "msr_lora_name": (["None"] + loras, {"default": "None"}),
                "msr_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 32}),
                "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
            },
            "optional": {
                "candidate_2": ("LATENT", {"lazy": True}),
                "candidate_3": ("LATENT", {"lazy": True}),
                "candidate_4": ("LATENT", {"lazy": True}),
                "motion_guide_data": ("MOTION_GUIDE_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT", "STRING")
    RETURN_NAMES = ("images", "audio", "latent", "info")
    FUNCTION = "refine"
    CATEGORY = "StubeliusLTX"

    def check_lazy_status(self, candidate=1, **kwargs):
        c = _coerce_int(candidate, 0)
        if 1 <= c <= 4 and kwargs.get(f"candidate_{c}") is None:
            return [f"candidate_{c}"]
        return []  # candidate 0/invalid: request nothing, node blocks in refine()

    @classmethod
    def VALIDATE_INPUTS(cls, polish_steps=None, refine_denoise=None, seed=None, cfg=None,
                        candidate=None, sync_from_hunt=None, audio_mode=None,
                        sigma_mode=None, manual_sigmas=None):
        return True

    @_holds_run_models(lambda self, model, *a, **k: model)
    def refine(self, model, positive, negative, vae, audio_vae, upscale_model, guide_data,
               candidate_1=None, candidate=1, sync_from_hunt=True,
               audio_mode="keep candidate audio (locked)",
               sigma_mode="manual sigmas (CS)", manual_sigmas=CS_DEFAULT_SIGMAS,
               polish_steps=4, refine_denoise=0.42, sampler_name="euler",
               scheduler="simple", cfg=1.0, seed=42,
               ic_lora_name="None", ic_lora_strength=0.6,
               image_attention_strength=1.0, msr_strength=0.4,
               msr_lora_name="None", msr_lora_strength=1.0,
               tile_size=512, tile_overlap=64,
               candidate_2=None, candidate_3=None, candidate_4=None,
               motion_guide_data=None):

        candidate = _coerce_int(candidate, 0)
        if candidate < 1:
            log.info("[StubeliusLTX] Refine on WAIT (candidate=0): hunt only. Set candidate 1-4 and re-queue.")
            from comfy_execution.graph import ExecutionBlocker
            b = ExecutionBlocker(None)
            return (b, b, b, "Refine waiting - pick a candidate (1-4) and re-queue.")
        polish_steps = _coerce_int(polish_steps, 4)
        refine_denoise = _coerce_float(refine_denoise, 0.42)
        cfg = _coerce_float(cfg, 1.0)
        seed = _coerce_int(seed, 42)
        sync_from_hunt = _coerce_bool(sync_from_hunt, True)
        if not str(audio_mode or "").strip():
            audio_mode = "keep candidate audio (locked)"

        cands = {1: candidate_1, 2: candidate_2, 3: candidate_3, 4: candidate_4}
        sel = cands.get(candidate)
        if isinstance(sel, dict) and sel.get("_stub_disabled"):
            log.warning("[StubeliusLTX] Candidate %d was not hunted (slot disabled) - "
                        "enable that slot and re-run the hunt, or pick a hunted slot.", candidate)
            sel = None
        if sel is None:
            raise ValueError(f"[StubeliusLTX] candidate {candidate} is not connected / was not hunted.")

        synced = sel.get(STUB_SETTINGS) if isinstance(sel, dict) else None
        if sync_from_hunt and synced:
            seed = _coerce_int(synced.get("seed", seed), seed)
            sampler_name = synced.get("sampler_name", sampler_name)
            cfg = _coerce_float(synced.get("cfg", cfg), cfg)
            if str(ic_lora_name) in ("", "None"):
                ic_lora_name = synced.get("ic_lora_name", ic_lora_name)
                ic_lora_strength = _coerce_float(synced.get("ic_lora_strength", ic_lora_strength), ic_lora_strength)
            log.info("[StubeliusLTX] sync: candidate %d settings pulled (seed=%d, %s, cfg=%.2f)",
                     candidate, seed, sampler_name, cfg)
        elif sync_from_hunt:
            log.info("[StubeliusLTX] sync: candidate carries no embedded settings - using widget values.")

        audio_samples = sel.get(STUB_AUDIO) if isinstance(sel, dict) else None
        if audio_samples is None:
            raise ValueError("[StubeliusLTX] candidate carries no audio latent - re-run the hunt with the current pack version.")
        audio_latent = {"samples": audio_samples}

        upsampled = call_node("LTXVLatentUpsampler", samples={"samples": sel["samples"]},
                              latent={"samples": sel["samples"]},
                              upscale_model=upscale_model, vae=vae)[0]
        # the picked candidate's low-res latent is no longer needed once upscaled;
        # free before the guide rebuild + tiled decode, which are the real VRAM peak.
        try:
            sel_samples = sel["samples"]
            del sel_samples
        except Exception:
            pass
        _free_vram("post-upscale")

        pos2, neg2, lat2, model2, _ = _guide(
            positive, negative, vae, upsampled, guide_data, motion_guide_data, model,
            ic_lora_name, _coerce_float(ic_lora_strength, 0.6), 1.0,
            _coerce_float(image_attention_strength, 1.0), _coerce_float(msr_strength, 0.4),
            msr_lora_name, _coerce_float(msr_lora_strength, 1.0))

        if str(sigma_mode).startswith("manual"):
            sigmas = _parse_sigmas(manual_sigmas)
        else:
            sigmas = call_node("BasicScheduler", model=model2, scheduler=scheduler,
                               steps=polish_steps, denoise=refine_denoise)[0]

        audio_lock = str(audio_mode).startswith("keep")
        video2, audio2 = _sample_av(model2, pos2, neg2, cfg, sampler_name,
                                    sigmas, seed, lat2, audio_latent, audio_lock=audio_lock)

        _free_vram("pre-decode")
        images, audio_out, cropped = _decode(vae, audio_vae, pos2, neg2,
                                             video2, audio2, tile_size, tile_overlap)
        n_sig = int(sigmas.shape[0]) - 1 if hasattr(sigmas, "shape") else "?"
        info = (f"StubeliusLTX refined candidate {candidate} | {n_sig} tail steps | "
                f"{sampler_name} cfg {cfg} | audio: {audio_mode} | msr {msr_strength}")
        log.info("[StubeliusLTX] %s", info)
        return (images, audio_out, cropped, info)


NODE_CLASS_MAPPINGS = {
    "StubeliusLTXSeedHunt": StubeliusLTXSeedHunt,
    "StubeliusLTXRefine": StubeliusLTXRefine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StubeliusLTXSeedHunt": "⭐ Stubelius LTX — Seed Hunt",
    "StubeliusLTXRefine": "⭐ Stubelius LTX — Refine (Pass 2)",
}
