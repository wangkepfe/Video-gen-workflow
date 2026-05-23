"""LTX-Video 13B 0.9.8 distilled (fp8) backend.

Uses the all-in-one Lightricks checkpoint (model + VAE bundled) plus a
separate T5XXL fp8 text encoder. The distilled model targets ~8 sampling
steps at CFG=1.0, so a clip generates in seconds rather than minutes.

References (ComfyUI built-ins):
  - LTXVImgToVideo / EmptyLTXVLatentVideo  (comfy_extras/nodes_lt.py)
  - LTXVConditioning (frame_rate)
  - ModelSamplingLTXV (token-aware sigma shift)
  - LTXVScheduler (sigmas)
"""
import math
import os

import torch

import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_sampling
import comfy.model_management
import folder_paths
import node_helpers

from .base import Field, MissingFiles
from ._util import load_image_as_tensor, save_webm


NAME = "ltx"
DISPLAY_NAME = "LTX-Video 13B 0.9.8 distilled (fp8)"
DESCRIPTION = "Lightricks LTX-Video 13B distilled. Fast: 8 steps @ CFG 1.0, single-pass — ~30s/clip on RTX 4090."

# Distilled LTX models react badly to long, contradictory negative prompts.
# A short list works better; the default mirrors Lightricks' recommendation.
DEFAULT_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, "
    "motion smear, motion artifacts, fused fingers, bad anatomy, "
    "weird hand, ugly"
)

DEFAULT_MAX_SHIFT = 2.05
DEFAULT_BASE_SHIFT = 0.95
DEFAULT_TERMINAL = 0.1


def _comfy_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ComfyUI_windows_portable", "ComfyUI",
    )


def _paths():
    models = os.path.join(_comfy_dir(), "models")
    return {
        # The 0.9.8 distilled fp8 file bundles model + VAE; loaded via
        # CheckpointLoaderSimple-style guess_config.
        "ckpt": os.path.join(models, "checkpoints", "ltxv-13b-0.9.8-distilled-fp8.safetensors"),
        "t5":   os.path.join(models, "text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors"),
    }


def required_files():
    p = _paths()
    return [
        ("LTX-Video 13B 0.9.8 distilled fp8", p["ckpt"]),
        ("T5XXL fp8 text encoder",             p["t5"]),
    ]


FIELDS = [
    # LTX latent dims: width/height must be multiples of 32, length must be 8n+1.
    Field("ratio", "Aspect ratio", "ratio_buttons", default=None, presets=[
        {"w": 1216, "h": 704, "label": "16:9 · 1216×704"},
        {"w": 704,  "h": 1216, "label": "9:16 · 704×1216"},
        {"w": 960,  "h": 960,  "label": "1:1 · 960×960"},
        {"w": 768,  "h": 512,  "label": "3:2 · 768×512 (fast)"},
    ]),
    Field("width",  "Width",  "int", default=1216, step=32, min=64,  max=2048,
          help="Must be a multiple of 32."),
    Field("height", "Height", "int", default=704,  step=32, min=64,  max=2048,
          help="Must be a multiple of 32."),
    Field("frames", "Frames", "int", default=121,  step=8,  min=9,   max=257,
          help="Must be 8·n + 1 (rounded automatically). 121 ≈ 5s at 24fps."),
    Field("fps",    "FPS",    "float", default=24.0, step=1, min=1, max=60),
    Field("steps",  "Steps",  "int",   default=8,    step=1, min=1, max=50,
          help="Distilled model is tuned for 8 steps."),
    Field("cfg",    "CFG",    "float", default=1.0, step=0.1, min=1.0, max=10.0,
          help="Distilled model uses CFG=1.0. Raise only if you understand the trade-off."),
    Field("sampler", "Sampler", "select", default="euler",
          options=["euler", "euler_ancestral", "dpmpp_2m"], advanced=True),
    Field("max_shift",  "Max shift",  "float", default=DEFAULT_MAX_SHIFT,  step=0.05, min=0, max=10, advanced=True),
    Field("base_shift", "Base shift", "float", default=DEFAULT_BASE_SHIFT, step=0.05, min=0, max=10, advanced=True),
    Field("terminal",   "Terminal sigma", "float", default=DEFAULT_TERMINAL, step=0.01, min=0, max=0.99, advanced=True,
          help="Lower terminal = stronger final denoise. Stretches the sigma schedule."),
    Field("seed",  "Seed", "text", default="", placeholder="random",
          help="Blank = random."),
]


# ── model loading ──────────────────────────────────────────────────────────

def load_models():
    paths = _paths()
    missing = [(label, p) for label, p in required_files() if not os.path.isfile(p)]
    if missing:
        msg = "Missing LTX-Video model files. Run download_models.bat to fetch:\n"
        for label, p in missing:
            msg += f"  - {label}\n      {p}\n"
        raise MissingFiles(msg)

    print("Loading LTX-Video 13B distilled fp8 (model + bundled VAE)...")
    # output_clip=False because the LTX checkpoint doesn't carry the text encoder;
    # we load T5XXL separately below.
    out = comfy.sd.load_checkpoint_guess_config(
        paths["ckpt"],
        output_vae=True,
        output_clip=False,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    model, _, vae = out[0], out[1], out[2]

    print("Loading T5XXL fp8 text encoder...")
    clip = comfy.sd.load_clip(
        ckpt_paths=[paths["t5"]],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.LTXV,
    )
    print("LTX-Video ready.\n")
    return {"model": model, "clip": clip, "vae": vae}


# ── helpers replicating the LTX nodes ──────────────────────────────────────

def _patch_model_sampling(model, tokens, max_shift, base_shift):
    """Replicates ModelSamplingLTXV: token-count-aware shift on the flux base."""
    x1, x2 = 1024, 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    shift = tokens * mm + b

    m = model.clone()

    class _Sampling(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
        pass

    ms = _Sampling(m.model.model_config)
    ms.set_parameters(shift=shift)
    m.add_object_patch("model_sampling", ms)
    return m


def _build_ltxv_sigmas(steps, tokens, max_shift, base_shift, terminal, stretch=True):
    """Replicates LTXVScheduler."""
    x1, x2 = 1024, 4096
    mm = (max_shift - base_shift) / (x2 - x1)
    b = base_shift - mm * x1
    sigma_shift = tokens * mm + b

    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    sigmas = torch.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
        torch.zeros_like(sigmas),
    )
    if stretch:
        non_zero_mask = sigmas != 0
        non_zero = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero
        scale = one_minus_z[-1] / (1.0 - terminal)
        sigmas[non_zero_mask] = 1.0 - (one_minus_z / scale)
    return sigmas


def _snap_length(length):
    """LTX latents need length = 8·n + 1; snap and clamp to a sane minimum."""
    length = max(9, int(length))
    return ((length - 1) // 8) * 8 + 1


def _snap_dim(d):
    return max(64, (int(d) // 32) * 32)


# ── inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(models, image_path, prompt, negative, settings, output_path):
    model_in = models["model"]
    clip = models["clip"]
    vae = models["vae"]

    width  = _snap_dim(settings["width"])
    height = _snap_dim(settings["height"])
    length = _snap_length(settings["frames"])
    steps  = int(settings["steps"])
    cfg    = float(settings["cfg"])
    fps    = float(settings["fps"])
    seed   = int(settings["seed"])
    sampler_name = settings.get("sampler", "euler")
    max_shift  = float(settings.get("max_shift",  DEFAULT_MAX_SHIFT))
    base_shift = float(settings.get("base_shift", DEFAULT_BASE_SHIFT))
    terminal   = float(settings.get("terminal",   DEFAULT_TERMINAL))

    if (width, height, length) != (int(settings["width"]), int(settings["height"]), int(settings["frames"])):
        print(f"  [LTX] snapped dims → {width}x{height}, {length} frames "
              f"(LTX requires 32-multiples and length = 8·n+1)")

    print(f"  Image: {image_path}")
    print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    neg_cond = clip.encode_from_tokens_scheduled(clip.tokenize(negative))

    print(f"  Building LTX latent ({width}x{height}, {length} frames)...")
    img_tensor = load_image_as_tensor(image_path)
    pixels = comfy.utils.common_upscale(
        img_tensor.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    t = vae.encode(pixels[:, :, :, :3])

    device = comfy.model_management.intermediate_device()
    latent_image = torch.zeros(
        [1, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=device
    )
    latent_image[:, :, :t.shape[2]] = t

    # noise_mask: tokens where mask == 0 are kept (image-conditioned region).
    noise_mask = torch.ones((1, 1, latent_image.shape[2], 1, 1),
                            dtype=torch.float32, device=latent_image.device)
    noise_mask[:, :, :t.shape[2]] = 0.0  # strength 1.0

    # LTXVConditioning: stamp frame_rate onto the conditioning tuples.
    positive = node_helpers.conditioning_set_values(positive, {"frame_rate": fps})
    neg_cond = node_helpers.conditioning_set_values(neg_cond, {"frame_rate": fps})

    # ModelSamplingLTXV + LTXVScheduler — both need the latent token count.
    tokens = math.prod(latent_image.shape[2:])
    model = _patch_model_sampling(model_in, tokens, max_shift, base_shift)
    sigmas = _build_ltxv_sigmas(steps, tokens, max_shift, base_shift, terminal)

    print(f"  Sampling: {steps} steps, CFG {cfg}, sampler {sampler_name}, seed {seed}...")
    noise = comfy.sample.prepare_noise(latent_image, seed)
    sampler = comfy.samplers.sampler_object(sampler_name)
    samples = comfy.sample.sample_custom(
        model, noise, cfg, sampler, sigmas,
        positive, neg_cond, latent_image,
        noise_mask=noise_mask, seed=seed,
    )

    print(f"  Decoding and saving ({fps} fps)...")
    video_frames = vae.decode(samples)
    if len(video_frames.shape) == 5:
        video_frames = video_frames.reshape(
            -1, video_frames.shape[-3], video_frames.shape[-2], video_frames.shape[-1]
        )

    save_webm(video_frames, output_path, fps)
    print(f"  Saved: {output_path} ({video_frames.shape[0]} frames, "
          f"{video_frames.shape[0]/fps:.1f}s)\n")
