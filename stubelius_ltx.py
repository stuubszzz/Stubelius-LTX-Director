"""Stubelius LTX — Seed Hunt + Refine dashboards for the LTX 2.5 Director (CS).

Replaces the manual two-pass node spaghetti with two dashboard nodes:

  * StubeliusLTXSeedHunt — up to 4 full first-pass candidates, each with its own
    seed / sampler / scheduler / steps / cfg. Guides are built ONCE (they don't
    depend on seed) and only the sampling varies per slot. Each candidate latent
    carries its settings + its audio latent, so the Refine auto-syncs.

  * StubeliusLTXRefine — pick a candidate (lazy inputs: un-run slots can never
    block execution), latent-upsample it, rebuild guides at full scale, and run
    the CS-style tail re-noise pass. Audio lock keeps the exact audio you
    auditioned; sigmas default to the CS manual schedule (0.85 → 0).

All third-party node calls are resolved from NODE_CLASS_MAPPINGS at execution
time and kwargs are filtered against each node's live INPUT_TYPES, so version
drift in the CS / LTXVideo packs cannot break the calls.
"""

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
            v = video_latent["samples"]
            a = audio_latent["samples"]
            combined = dict(combined)
            combined["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (torch.ones_like(v), torch.zeros_like(a)))
            log.info("[StubeliusLTX] audio lock: candidate audio frozen through pass 2.")
        except Exception as e:  # noqa: BLE001 - lock is best-effort, never fatal
            log.warning("[StubeliusLTX] audio lock unavailable (%s) - audio will re-render.", e)
    out = call_node("SamplerCustomAdvanced", noise=noise, guider=guider,
                    sampler=sampler, sigmas=sigmas, latent_image=combined)[0]
    video, audio = call_node("LTXVSeparateAVLatent", av_latent=out, latent=out, samples=out)[:2]
    return video, audio


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


def _decode(vae, audio_vae, positive, negative, video_latent, audio_latent,
            tile_size, tile_overlap):
    pos_c, neg_c, cropped = call_node("LTXDirectorCropGuidesCS25",
                                      positive=positive, negative=negative, latent=video_latent)[:3]
    images = call_node("VAEDecodeTiled", vae=vae, samples=cropped,
                       tile_size=tile_size, overlap=tile_overlap,
                       temporal_size=64, temporal_overlap=8)[0]
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
            "tooltip": "Per-slot hunt config (JSON) - normally edited via the 2x2 gold panel above. "
            "Hand-editable fallback: list of up to 4 objects with enable/seed/sampler/scheduler/steps/cfg."})
        return {"required": req, "optional": {"motion_guide_data": ("MOTION_GUIDE_DATA",)}}

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT", "IMAGE", "AUDIO", "LATENT",
                    "IMAGE", "AUDIO", "LATENT", "IMAGE", "AUDIO", "LATENT", "STRING")
    RETURN_NAMES = ("images_1", "audio_1", "candidate_1", "images_2", "audio_2", "candidate_2",
                    "images_3", "audio_3", "candidate_3", "images_4", "audio_4", "candidate_4", "info")
    FUNCTION = "hunt"
    CATEGORY = "StubeliusLTX"

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
                outs.extend([_blocker(), _blocker(), _blocker()])
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
