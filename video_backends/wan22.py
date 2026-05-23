"""Wan2.2 14B I2V (MoE high-noise + low-noise expert pair) backend.

Ported from the original generate_video.py. Behaviour is intended to be
bit-exact with the previous implementation when defaults are kept.
"""
import os

import torch

import comfy.sd
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy_extras.nodes_model_advanced as nma
import folder_paths
import node_helpers

from .base import Field, MissingFiles
from ._util import load_image_as_tensor, save_webm


NAME = "wan22"
DISPLAY_NAME = "Wan2.2 14B I2V (MoE)"
DESCRIPTION = "Two-expert (high/low noise) Wan2.2 14B image-to-video. Best quality, ~6-8 min/clip on RTX 4090."

DEFAULT_NEGATIVE = (
    "static, blurry, low quality, worst quality, JPEG artifacts, ugly, "
    "deformed, extra limbs, fused fingers, still frame, cluttered background, "
    "overexposed, underexposed"
)

DEFAULT_SHIFT = 5.0


def _comfy_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ComfyUI_windows_portable", "ComfyUI",
    )


def _paths():
    models = os.path.join(_comfy_dir(), "models")
    return {
        "unet_high": os.path.join(models, "diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"),
        "unet_low":  os.path.join(models, "diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
        "clip":      os.path.join(models, "text_encoders",    "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        "vae":       os.path.join(models, "vae",              "wan_2.1_vae.safetensors"),
    }


def required_files():
    p = _paths()
    return [
        ("Wan2.2 14B I2V high-noise expert", p["unet_high"]),
        ("Wan2.2 14B I2V low-noise expert",  p["unet_low"]),
        ("UMT5-XXL text encoder",            p["clip"]),
        ("Wan 2.1 VAE (16-channel)",         p["vae"]),
    ]


FIELDS = [
    Field("ratio", "Aspect ratio", "ratio_buttons", default=None, presets=[
        {"w": 1280, "h": 720,  "label": "16:9 · 1280×720"},
        {"w": 720,  "h": 1280, "label": "9:16 · 720×1280"},
        {"w": 1024, "h": 1024, "label": "1:1 · 1024×1024"},
    ]),
    Field("width",         "Width",         "int", default=1280, step=16, min=64, max=4096),
    Field("height",        "Height",        "int", default=720,  step=16, min=64, max=4096),
    Field("frames",        "Frames",        "int", default=120,  step=1,  min=1,  max=1024,
          help="Number of frames. 120 ≈ 5s at 24fps."),
    Field("fps",           "FPS",           "float", default=24.0, step=1, min=1, max=120),
    Field("steps",         "Steps",         "int", default=30,   step=1,  min=1,  max=200,
          help="Total diffusion steps, split half/half between the two experts."),
    Field("cfg",           "CFG",           "float", default=3.5, step=0.5, min=0, max=20),
    Field("boundary_step", "Boundary step", "text",  default="", placeholder="auto (= steps/2)",
          help="Step at which to switch from high-noise to low-noise expert.", advanced=True),
    Field("sampler",       "Sampler",       "select", default="euler",
          options=["euler", "uni_pc", "euler_ancestral", "dpmpp_2m"], advanced=True),
    Field("scheduler",     "Scheduler",     "select", default="simple",
          options=["simple", "normal", "karras", "exponential"], advanced=True),
    Field("shift",         "Sampling shift", "float", default=DEFAULT_SHIFT, step=0.1, min=0, max=20,
          advanced=True),
    Field("seed",          "Seed",          "text",  default="", placeholder="random",
          help="Blank = random."),
]


# ── model loading ──────────────────────────────────────────────────────────

def _patch_shift(model, shift):
    if shift and shift > 0:
        return nma.ModelSamplingSD3().patch(model, shift=shift)[0]
    return model


def load_models(shift=DEFAULT_SHIFT):
    paths = _paths()
    missing = [(label, p) for label, p in required_files() if not os.path.isfile(p)]
    if missing:
        msg = "Missing Wan2.2 model files. Run download_models.bat to fetch:\n"
        for label, p in missing:
            msg += f"  - {label}\n      {p}\n"
        raise MissingFiles(msg)

    print("Loading Wan2.2 14B high-noise expert...")
    model_high = comfy.sd.load_diffusion_model(paths["unet_high"])
    print("Loading Wan2.2 14B low-noise expert...")
    model_low = comfy.sd.load_diffusion_model(paths["unet_low"])

    if shift and shift > 0:
        print(f"Applying ModelSamplingSD3 shift={shift} to both experts...")
        model_high = _patch_shift(model_high, shift)
        model_low = _patch_shift(model_low, shift)

    print("Loading UMT5-XXL text encoder...")
    clip = comfy.sd.load_clip(
        ckpt_paths=[paths["clip"]],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.WAN,
    )
    print("Loading Wan 2.1 VAE (16-ch)...")
    vae_sd, vae_meta = comfy.utils.load_torch_file(paths["vae"], return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_meta)
    print("Wan2.2 ready.\n")
    return {"model_high": model_high, "model_low": model_low, "clip": clip, "vae": vae,
            "_shift_applied": shift}


# ── inference ──────────────────────────────────────────────────────────────

def _build_i2v_latent(vae, start_image, width, height, length):
    """Replicates ComfyUI's WanImageToVideo node."""
    latent_length = ((length - 1) // 4) + 1
    device = comfy.model_management.intermediate_device()

    latent = torch.zeros([1, 16, latent_length, height // 8, width // 8], device=device)

    start_image = comfy.utils.common_upscale(
        start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)

    image = torch.ones(
        (length, height, width, start_image.shape[-1]),
        device=start_image.device, dtype=start_image.dtype,
    ) * 0.5
    image[:start_image.shape[0]] = start_image

    concat_latent_image = vae.encode(image[:, :, :, :3])
    mask = torch.ones(
        (1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
        device=start_image.device, dtype=start_image.dtype,
    )
    mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

    return {"samples": latent, "concat_latent_image": concat_latent_image, "concat_mask": mask}


@torch.no_grad()
def generate(models, image_path, prompt, negative, settings, output_path):
    model_high = models["model_high"]
    model_low = models["model_low"]
    clip = models["clip"]
    vae = models["vae"]

    width  = int(settings["width"])
    height = int(settings["height"])
    frames = int(settings["frames"])
    steps  = int(settings["steps"])
    cfg    = float(settings["cfg"])
    fps    = float(settings["fps"])
    seed   = int(settings["seed"])
    sampler_name = settings.get("sampler", "euler")
    scheduler    = settings.get("scheduler", "simple")

    boundary_step = settings.get("boundary_step")
    if boundary_step in (None, "", "auto"):
        boundary_step = steps // 2
    boundary_step = max(1, min(steps - 1, int(boundary_step)))

    print(f"  Image: {image_path}")
    print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    neg_cond = clip.encode_from_tokens_scheduled(clip.tokenize(negative))

    print(f"  Building I2V latent ({width}x{height}, {frames} frames)...")
    img_tensor = load_image_as_tensor(image_path)

    latent_dict = _build_i2v_latent(vae, img_tensor, width, height, frames)
    latent_image = latent_dict["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model_high, latent_image)

    cond_extras = {
        "concat_latent_image": latent_dict["concat_latent_image"],
        "concat_mask": latent_dict["concat_mask"],
    }
    positive = node_helpers.conditioning_set_values(positive, cond_extras)
    neg_cond = node_helpers.conditioning_set_values(neg_cond, cond_extras)

    noise = comfy.sample.prepare_noise(latent_image, seed)

    print(f"  [Pass 1/2] high-noise expert  steps 0-{boundary_step}/{steps}  "
          f"({sampler_name}/{scheduler}, CFG {cfg}, seed {seed})...")
    samples_partial = comfy.sample.sample(
        model_high, noise, steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=neg_cond,
        latent_image=latent_image, denoise=1.0,
        start_step=0, last_step=boundary_step, force_full_denoise=False,
        seed=seed,
    )

    print(f"  [Pass 2/2] low-noise expert   steps {boundary_step}-{steps}/{steps}...")
    zero_noise = torch.zeros_like(noise)
    samples = comfy.sample.sample(
        model_low, zero_noise, steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=neg_cond,
        latent_image=samples_partial, denoise=1.0,
        disable_noise=True,
        start_step=boundary_step, last_step=steps, force_full_denoise=True,
        seed=seed,
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
