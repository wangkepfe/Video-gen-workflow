import sys
import os
import argparse
import time
import random
import ctypes
from fractions import Fraction

_LOW_PRIORITY = "--low-priority" in sys.argv

if _LOW_PRIORITY:
    try:
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

COMFY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI_windows_portable", "ComfyUI")
sys.path.insert(0, COMFY_DIR)

import torch
import numpy as np
import av
from PIL import Image

if _LOW_PRIORITY:
    for _dll in ("cudart64_12.dll", "cudart64_110.dll"):
        try:
            _cudart = ctypes.WinDLL(_dll)
            _cudart.cudaSetDeviceFlags(ctypes.c_uint(0x04))
            break
        except (OSError, Exception):
            continue

import comfy.sd
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.latent_formats
import folder_paths
import comfy_extras.nodes_model_advanced as nma

MODELS_DIR = os.path.join(COMFY_DIR, "models")
UNET_HIGH_PATH = os.path.join(MODELS_DIR, "diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
UNET_LOW_PATH = os.path.join(MODELS_DIR, "diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
CLIP_PATH = os.path.join(MODELS_DIR, "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
VAE_PATH = os.path.join(MODELS_DIR, "vae", "wan2.2_vae.safetensors")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "videos")

DEFAULT_SHIFT = 5.0


def _patch_shift(model, shift):
    if shift and shift > 0:
        return nma.ModelSamplingSD3().patch(model, shift=shift)[0]
    return model


def load_models(shift=DEFAULT_SHIFT):
    if not os.path.isfile(UNET_HIGH_PATH):
        raise FileNotFoundError(f"Missing high-noise expert: {UNET_HIGH_PATH}\nRun download_models.bat first.")
    if not os.path.isfile(UNET_LOW_PATH):
        raise FileNotFoundError(f"Missing low-noise expert: {UNET_LOW_PATH}\nRun download_models.bat first.")

    print("Loading Wan2.2 14B high-noise expert...")
    model_high = comfy.sd.load_diffusion_model(UNET_HIGH_PATH)
    print("Loading Wan2.2 14B low-noise expert...")
    model_low = comfy.sd.load_diffusion_model(UNET_LOW_PATH)

    if shift and shift > 0:
        print(f"Applying ModelSamplingSD3 shift={shift} to both experts...")
        model_high = _patch_shift(model_high, shift)
        model_low = _patch_shift(model_low, shift)

    print("Loading UMT5-XXL text encoder...")
    clip = comfy.sd.load_clip(
        ckpt_paths=[CLIP_PATH],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.WAN,
    )
    print("Loading Wan2.2 VAE...")
    vae_sd, vae_meta = comfy.utils.load_torch_file(VAE_PATH, return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_meta)
    print("Models loaded. Ready.\n")
    return model_high, model_low, clip, vae


def read_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def make_output_path():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"wan22_{ts}.webm")


def build_i2v_latent(vae, start_image, width, height, length):
    """Replicates Wan22ImageToVideoLatent logic."""
    latent_length = ((length - 1) // 4) + 1
    device = comfy.model_management.intermediate_device()

    latent = torch.zeros([1, 48, latent_length, height // 16, width // 16], device=device)
    mask = torch.ones([1, 1, latent_length, latent.shape[-2], latent.shape[-1]], device=device)

    start_image = comfy.utils.common_upscale(
        start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)

    latent_temp = vae.encode(start_image)
    latent[:, :, :latent_temp.shape[-3]] = latent_temp
    mask[:, :, :latent_temp.shape[-3]] *= 0.0

    latent_format = comfy.latent_formats.Wan22()
    latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)

    return {"samples": latent, "noise_mask": mask}


def save_webm(frames, output_path, fps):
    """Save NHWC float [0,1] tensor as VP9 WEBM."""
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libvpx-vp9", rate=Fraction(round(fps * 1000), 1000))
    stream.width = frames.shape[-2]
    stream.height = frames.shape[-3]
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = 0
    stream.options = {"crf": "16"}

    for frame_tensor in frames:
        frame_np = torch.clamp(frame_tensor[..., :3] * 255, 0, 255).to(dtype=torch.uint8, device="cpu").numpy()
        frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    container.mux(stream.encode())
    container.close()


@torch.inference_mode()
def generate_video(model_high, model_low, clip, vae, image_path, prompt, negative_prompt,
                   width, height, frames, steps, cfg, sampler_name, scheduler, seed, fps, output_path,
                   boundary_step=None):
    """Two-pass MoE sampling: high-noise expert for first half, low-noise for second."""
    print(f"  Image: {image_path}")
    print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    if boundary_step is None:
        boundary_step = steps // 2
    boundary_step = max(1, min(steps - 1, int(boundary_step)))

    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

    print(f"  Building I2V latent ({width}x{height}, {frames} frames)...")
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

    latent_dict = build_i2v_latent(vae, img_tensor, width, height, frames)
    latent_image = latent_dict["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model_high, latent_image)

    noise = comfy.sample.prepare_noise(latent_image, seed)
    noise_mask = latent_dict.get("noise_mask")

    print(f"  [Pass 1/2] high-noise expert  steps 0-{boundary_step}/{steps}  ({sampler_name}/{scheduler}, CFG {cfg}, seed {seed})...")
    samples_partial = comfy.sample.sample(
        model_high, noise, steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=negative,
        latent_image=latent_image, denoise=1.0,
        start_step=0, last_step=boundary_step, force_full_denoise=False,
        noise_mask=noise_mask, seed=seed,
    )

    print(f"  [Pass 2/2] low-noise expert   steps {boundary_step}-{steps}/{steps}...")
    zero_noise = torch.zeros_like(noise)
    samples = comfy.sample.sample(
        model_low, zero_noise, steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=negative,
        latent_image=samples_partial, denoise=1.0,
        disable_noise=True,
        start_step=boundary_step, last_step=steps, force_full_denoise=True,
        noise_mask=noise_mask, seed=seed,
    )

    print(f"  Decoding and saving ({fps} fps)...")
    video_frames = vae.decode(samples)
    if len(video_frames.shape) == 5:
        video_frames = video_frames.reshape(-1, video_frames.shape[-3], video_frames.shape[-2], video_frames.shape[-1])

    save_webm(video_frames, output_path, fps)
    print(f"  Saved: {output_path} ({video_frames.shape[0]} frames, {video_frames.shape[0]/fps:.1f}s)\n")


DEFAULT_NEGATIVE = "static, blurry, low quality, worst quality, JPEG artifacts, ugly, deformed, extra limbs, fused fingers, still frame, cluttered background, overexposed, underexposed"


def run_interactive(models, args):
    model_high, model_low, clip, vae = models
    print("=" * 60)
    print("  Wan2.2 14B I2V (MoE) - Interactive Mode")
    print(f"  Resolution: {args.width}x{args.height}  Frames: {args.frames}")
    print(f"  Steps: {args.steps}  CFG: {args.cfg}  Sampler: {args.sampler}/{args.scheduler}  Shift: {args.shift}")
    print("=" * 60)
    print("Enter: image_path  prompt_file_or_text  [negative_file_or_text]")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            line = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split(maxsplit=2)
        if len(parts) < 2:
            print("  Usage: image_path  prompt_file  [negative_file]\n")
            continue

        image_path = parts[0]
        prompt_input = parts[1]
        negative_input = parts[2] if len(parts) > 2 else None

        if not os.path.isfile(image_path):
            print(f"  Image not found: {image_path}\n")
            continue

        prompt = read_prompt(prompt_input) if os.path.isfile(prompt_input) else prompt_input
        if not prompt:
            print("  Prompt is empty.\n")
            continue

        neg = read_prompt(negative_input) if negative_input and os.path.isfile(negative_input) else (negative_input or args.negative)

        seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
        output_path = make_output_path()

        try:
            generate_video(model_high, model_low, clip, vae, image_path, prompt, neg,
                           args.width, args.height, args.frames, args.steps,
                           args.cfg, args.sampler, args.scheduler, seed, args.fps, output_path)
        except Exception as e:
            print(f"  Error: {e}\n")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def find_prompt_file(image_name_no_ext, folder_path, prompt_dir=None):
    candidates = []
    if prompt_dir:
        candidates.append(os.path.join(prompt_dir, f"{image_name_no_ext}.txt"))
    candidates.append(os.path.join(folder_path, f"{image_name_no_ext}.txt"))

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def process_folder(models, folder_path, prompt, negative, args):
    model_high, model_low, clip, vae = models
    files = sorted(
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        print(f"[ERROR] No images found in: {folder_path}")
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(folder_path))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "videos", f"batch_{folder_name}")
    os.makedirs(out_dir, exist_ok=True)

    prompt_dir = getattr(args, "prompt_dir", None)
    if prompt_dir:
        print(f"Prompt directory: {prompt_dir}")

    print(f"Processing {len(files)} images from: {folder_path}")
    print(f"Output folder: {out_dir}\n")

    for i, fname in enumerate(files, 1):
        image_path = os.path.join(folder_path, fname)
        name_no_ext = os.path.splitext(fname)[0]
        output_path = os.path.join(out_dir, f"{name_no_ext}.webm")

        img_prompt = prompt
        prompt_src = "(global)"
        prompt_file = find_prompt_file(name_no_ext, folder_path, prompt_dir)
        if prompt_file:
            img_prompt = read_prompt(prompt_file)
            prompt_src = prompt_file

        if not img_prompt:
            print(f"  [SKIP] {fname}: no prompt found\n")
            continue

        seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

        print(f"--- [{i}/{len(files)}] {fname}  prompt: {prompt_src} ---")
        try:
            generate_video(model_high, model_low, clip, vae, image_path, img_prompt, negative,
                           args.width, args.height, args.frames, args.steps,
                           args.cfg, args.sampler, args.scheduler, seed, args.fps, output_path)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}\n")
            continue

    print(f"Done. {len(files)} videos generated -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 14B I2V (MoE) - Image to Video")
    parser.add_argument("image", type=str, nargs="?", default=None, help="Input image path or folder of images")
    parser.add_argument("prompt", type=str, nargs="?", default=None, help="Motion prompt or use --prompt-file")
    parser.add_argument("--prompt-file", type=str, default=None, help="Read positive prompt from a text file")
    parser.add_argument("--negative-file", type=str, default=None, help="Read negative prompt from a text file")
    parser.add_argument("--negative", type=str, default=DEFAULT_NEGATIVE, help="Negative prompt text")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive mode: load models once, accept inputs in a loop")
    parser.add_argument("--width", type=int, default=1280, help="Video width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Video height (default: 720)")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames (default: 120, 5s at 24fps)")
    parser.add_argument("--steps", type=int, default=30, help="Total sampling steps (default: 30, split half/half between experts)")
    parser.add_argument("--boundary-step", type=int, default=None, help="Step at which to switch from high-noise to low-noise expert (default: steps/2)")
    parser.add_argument("--cfg", type=float, default=3.5, help="CFG scale (default: 3.5)")
    parser.add_argument("--sampler", type=str, default="euler", help="Sampler (default: euler)")
    parser.add_argument("--scheduler", type=str, default="simple", help="Scheduler (default: simple)")
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT, help=f"ModelSamplingSD3 shift (default: {DEFAULT_SHIFT})")
    parser.add_argument("--seed", type=int, default=None, help="Seed (default: random)")
    parser.add_argument("--fps", type=float, default=24.0, help="Output FPS (default: 24)")
    parser.add_argument("--prompt-dir", type=str, default=None,
                        help="Directory of per-image prompt .txt files (matched by image basename)")
    parser.add_argument("--output", type=str, default=None, help="Output path (single image mode)")
    parser.add_argument("--low-priority", action="store_true",
                        help="Reduce CPU/GPU priority so other applications stay responsive")
    args = parser.parse_args()

    if _LOW_PRIORITY:
        print("[Low priority] CPU/GPU scheduling priority reduced.\n")

    if args.interactive:
        models = load_models(shift=args.shift)
        run_interactive(models, args)
        return

    if args.prompt_file:
        args.prompt = read_prompt(args.prompt_file)
    if args.negative_file:
        args.negative = read_prompt(args.negative_file)

    if not args.image:
        parser.error("provide image + prompt (or --prompt-file), or use -i for interactive mode")

    if os.path.isdir(args.image):
        if not args.prompt and not args.prompt_dir:
            parser.error("folder mode requires a prompt (or --prompt-file / --prompt-dir)")
        models = load_models(shift=args.shift)
        process_folder(models, args.image, args.prompt or "", args.negative, args)
        return

    if not args.prompt:
        parser.error("provide image + prompt (or --prompt-file), or use -i for interactive mode")

    if not os.path.isfile(args.image):
        print(f"[ERROR] Input not found: {args.image}")
        sys.exit(1)

    models = load_models(shift=args.shift)
    model_high, model_low, clip, vae = models

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    if args.output is None:
        args.output = make_output_path()

    generate_video(model_high, model_low, clip, vae, args.image, args.prompt, args.negative,
                   args.width, args.height, args.frames, args.steps,
                   args.cfg, args.sampler, args.scheduler, args.seed, args.fps, args.output,
                   boundary_step=args.boundary_step)


if __name__ == "__main__":
    main()
