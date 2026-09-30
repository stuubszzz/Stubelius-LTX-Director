"""Stubelius MSR25 engine — LTX-2.5 Licon MSR injection for the Director guide.

Ports the slot-embedding reference convention from liconstudio/ComfyUI-LTX2.5-MSR
(learned Fourier-MLP slot embeddings + consecutive negative temporal offsets,
reference tokens appended as keyframes) into the Director pipeline. Replaces the
LTX-2.3 slideshow-prefix engine, which 2.5 was never trained on.
"""

import logging

import torch

import comfy.sd
import comfy.utils
import comfy_extras.nodes_lt as nodes_lt
import folder_paths
import node_helpers

log = logging.getLogger("LTXDirector")

_SLOT_PREFIXES = (
    "diffusion_model.reference_slot_embedding.",
    "reference_slot_embedding.",
)
_REQUIRED_SLOT_KEYS = {
    "frequencies",
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
}


def _extract_slot_state(lora):
    state, normal = {}, {}
    for key, value in lora.items():
        for prefix in _SLOT_PREFIXES:
            if key.startswith(prefix):
                state[key[len(prefix):]] = value.detach().cpu()
                break
        else:
            normal[key] = value
    return normal, state


def _slot_embedding(slot_id, state, device, dtype):
    frequencies = state["frequencies"].to(device=device, dtype=torch.float32)
    slot_value = torch.tensor(float(slot_id), device=device, dtype=torch.float32)
    scaled = slot_value / 16.0
    phases = scaled * frequencies
    features = torch.cat((scaled.reshape(1), torch.sin(phases), torch.cos(phases)))
    w0 = state["net.0.weight"].to(device=device, dtype=torch.float32)
    b0 = state["net.0.bias"].to(device=device, dtype=torch.float32)
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(features, w0, b0))
    w2 = state["net.2.weight"].to(device=device, dtype=torch.float32)
    b2 = state["net.2.bias"].to(device=device, dtype=torch.float32)
    return torch.nn.functional.linear(hidden, w2, b2).to(dtype=dtype)


def load_msr25_lora(model, lora_name, strength_model):
    """Load an LTX-2.5 MSR LoRA (model-only) and return (model, slot_state, downscale)."""
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
    metadata = metadata or {}
    normal_lora, slot_state = _extract_slot_state(lora)
    missing = sorted(_REQUIRED_SLOT_KEYS.difference(slot_state))
    if not slot_state or missing:
        raise ValueError(
            f"'{lora_name}' is not an LTX-2.5 MSR checkpoint "
            f"(missing reference_slot_embedding tensors{': ' + ', '.join(missing) if missing else ''}). "
            "Use the Licon LTX-2.5 MSR LoRA here; union control stays in ic_lora_name."
        )
    if metadata.get("reference_token_order", "prepend") != "prepend":
        raise ValueError("Unsupported reference_token_order; expected 'prepend'.")
    offsets = metadata.get("reference_slot_time_offsets", "pic1_based_negative_time")
    if offsets != "pic1_based_negative_time":
        raise ValueError("Unsupported reference_slot_time_offsets; expected 'pic1_based_negative_time'.")
    if model is not None and strength_model != 0:
        model, _ = comfy.sd.load_lora_for_models(model, None, normal_lora, strength_model, 0, lora_metadata=metadata)
    downscale = max(1, round(float(metadata.get("reference_downscale_factor", 1))))
    log.info("[Stubelius MSR25] Loaded %s (slot tensors=%d, downscale=%d)", lora_name, len(slot_state), downscale)
    return model, slot_state, downscale


def _append_attention_entry(conditioning, pre_filter_count, latent_shape, strength):
    existing = None
    for _, values in conditioning:
        if "guide_attention_entries" in values:
            existing = values["guide_attention_entries"]
            break
    existing = existing or []
    entry = {
        "pre_filter_count": pre_filter_count,
        "strength": strength,
        "pixel_mask": None,
        "latent_shape": latent_shape,
    }
    return node_helpers.conditioning_set_values(
        conditioning, {"guide_attention_entries": [*existing, entry]}
    )


def inject_msr25(positive, negative, vae, latent_image, noise_mask, slot_state,
                 references, strength, reference_frames, downscale=1):
    """Append MSR references as slot-embedded keyframes at negative temporal positions.

    references: list of (label, IMAGE[1,H,W,C], is_background). Returns
    (positive, negative, latent_image, noise_mask, appended_latent_frames).
    """
    if latent_image.ndim != 5 or latent_image.shape[1] != 128:
        raise ValueError("MSR25 needs a video-only LTX latent [B, 128, F, H, W] (before AV concat).")
    if latent_image.shape[0] != 1:
        raise ValueError("MSR25 inference requires batch_size=1.")
    reference_frames = int(reference_frames)
    if reference_frames not in (25, 33):
        reference_frames = 33
    num_slots = len(references)
    if not 1 <= num_slots <= 5:
        raise ValueError(f"MSR25 requires 1-5 references, got {num_slots}.")

    scale_factors = vae.downscale_index_formula
    _, _, _, latent_height, latent_width = latent_image.shape
    if latent_height % downscale or latent_width % downscale:
        raise ValueError(
            f"Latent grid {latent_width}x{latent_height} not divisible by reference_downscale_factor={downscale}."
        )

    frames_before = int(latent_image.shape[2])
    log.info(
        "[Stubelius MSR25] Inject start: references=%d, reference_frames=%d, downscale=%d, target_latent=%s",
        num_slots, reference_frames, downscale, tuple(latent_image.shape),
    )

    for slot_index, (label, image, is_background) in enumerate(references):
        slot_id = slot_index + 1
        if image.shape[0] != 1:
            image = image[:1]
        repeated = image.repeat(reference_frames, 1, 1, 1)
        _, guide_latent = nodes_lt.LTXVAddGuide.encode(
            vae, latent_width, latent_height, repeated, scale_factors, downscale
        )
        embedding = _slot_embedding(slot_id, slot_state, guide_latent.device, guide_latent.dtype)
        if embedding.numel() != guide_latent.shape[1]:
            raise ValueError(
                f"Slot embedding dim {embedding.numel()} != latent channels {guide_latent.shape[1]}."
            )
        guide_latent = guide_latent + embedding.view(1, -1, 1, 1, 1)
        original_shape = list(guide_latent.shape[2:])

        guide_mask = None
        if downscale > 1:
            guide_latent, guide_mask = nodes_lt.LTXVAddGuide.dilate_latent(guide_latent, downscale)

        frame_offset = -(num_slots - slot_index)
        positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
            positive, negative, frame_offset, latent_image, noise_mask, guide_latent,
            strength, scale_factors, guide_mask=guide_mask,
            latent_downscale_factor=downscale, causal_fix=True,
        )
        token_count = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
        positive = _append_attention_entry(positive, token_count, original_shape, strength)
        negative = _append_attention_entry(negative, token_count, original_shape, strength)
        log.info(
            "[Stubelius MSR25] %s: slot_id=%d, time_offset=%d, guide_latent=%s",
            label, slot_id, frame_offset, tuple(guide_latent.shape),
        )

    appended = int(latent_image.shape[2]) - frames_before
    log.info(
        "[Stubelius MSR25] Inject complete: added=%d refs (%d latent frames), output_latent=%s",
        num_slots, appended, tuple(latent_image.shape),
    )
    return positive, negative, latent_image, noise_mask, appended
