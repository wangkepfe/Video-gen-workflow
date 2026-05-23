import sys
import os
import argparse
import time
import random
import ctypes

_LOW_PRIORITY = "--low-priority" in sys.argv

if _LOW_PRIORITY:
    try:
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    os.environ["CUDA_DEVICE_SCHEDULE"] = "YIELD"

COMFY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI_windows_portable", "ComfyUI")
sys.path.insert(0, COMFY_DIR)

import torch
import numpy as np
from PIL import Image

if _LOW_PRIORITY:
    for _dll in ("cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll"):
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
import folder_paths

MODELS_DIR = os.path.join(COMFY_DIR, "models")
CHECKPOINT = os.path.join(MODELS_DIR, "checkpoints", "flux1-schnell-fp8.safetensors")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "images")


def ensure_checkpoint():
    if os.path.isfile(CHECKPOINT):
        return
    print("flux1-schnell-fp8.safetensors not found. Downloading from HuggingFace (~17 GB)...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="Comfy-Org/flux1-schnell",
        filename="flux1-schnell-fp8.safetensors",
        local_dir=os.path.dirname(CHECKPOINT),
    )
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"Download failed. Please manually download flux1-schnell-fp8.safetensors "
            f"and place it in:\n  {os.path.dirname(CHECKPOINT)}"
        )
    print("Download complete.\n")


def load_models():
    ensure_checkpoint()
    print("Loading FLUX.1 Schnell FP8 checkpoint...")
    model, clip, vae = comfy.sd.load_checkpoint_guess_config(
        CHECKPOINT, output_vae=True, output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )[:3]
    print("Models loaded. Ready.\n")
    return model, clip, vae


@torch.inference_mode()
def generate(model, clip, vae, prompt, width, height, steps, sampler_name, scheduler, seed, output_path):
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    latent_image = torch.zeros([1, 4, height // 8, width // 8],
                               device=comfy.model_management.intermediate_device())
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
    noise = comfy.sample.prepare_noise(latent_image, seed)

    print(f"Sampling ({steps} steps, {sampler_name}/{scheduler}, seed {seed})...")
    samples = comfy.sample.sample(
        model, noise, steps, cfg=1.0,
        sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=negative,
        latent_image=latent_image, denoise=1.0, seed=seed,
    )

    images = vae.decode(samples)
    img_np = np.clip(255.0 * images[0].detach().cpu().float().numpy(), 0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(output_path)
    print(f"Saved: {output_path}\n")


def make_output_path(suffix=""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"schnell_{ts}{suffix}.png")


def parse_input(user_input):
    """Parse 'prompt_or_file [xN]' and return (prompt, count)."""
    parts = user_input.rsplit(maxsplit=1)
    count = 1
    if len(parts) == 2 and parts[1].lower().startswith("x") and parts[1][1:].isdigit():
        count = int(parts[1][1:])
        user_input = parts[0]

    if os.path.isfile(user_input):
        with open(user_input, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    else:
        prompt = user_input

    return prompt, count


def run_interactive(model, clip, vae, args):
    print("=" * 60)
    print("  FLUX.1 Schnell FP8 — Interactive Mode")
    print(f"  Resolution: {args.width}x{args.height}  Steps: {args.steps}")
    print(f"  Sampler: {args.sampler}/{args.scheduler}")
    print("=" * 60)
    print("Enter a prompt or file path. Append xN to queue N runs.")
    print("  Examples:")
    print("    prompts\\office.txt")
    print("    prompts\\office.txt x5")
    print("    a beautiful sunset x3")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        prompt, count = parse_input(user_input)
        if not prompt:
            print("  Prompt is empty.\n")
            continue

        if count > 1:
            print(f"  Queued {count} generations.\n")

        for i in range(count):
            seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
            suffix = f"_{i+1}" if count > 1 else ""
            output_path = make_output_path(suffix)

            tag = f"[{i+1}/{count}] " if count > 1 else ""
            print(f"{tag}", end="")
            try:
                generate(model, clip, vae, prompt, args.width, args.height,
                         args.steps, args.sampler, args.scheduler, seed, output_path)
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.1 Schnell FP8 — Text to Image (interactive or single-shot)")
    parser.add_argument("prompt", type=str, nargs="?", default=None,
                        help="Text prompt for single-shot mode (or use --prompt-file)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read prompt from a text file (single-shot mode)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode: load models once, then accept prompts in a loop")
    parser.add_argument("--width", type=int, default=704, help="Image width (default: 704)")
    parser.add_argument("--height", type=int, default=1280, help="Image height (default: 1280)")
    parser.add_argument("--steps", type=int, default=4, help="Sampling steps (default: 4)")
    parser.add_argument("--sampler", type=str, default="euler", help="Sampler (default: euler)")
    parser.add_argument("--scheduler", type=str, default="simple", help="Scheduler (default: simple)")
    parser.add_argument("--seed", type=int, default=None, help="Seed (default: random)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: output/images/schnell_TIMESTAMP.png)")
    parser.add_argument("--low-priority", action="store_true",
                        help="Reduce CPU/GPU priority so other applications stay responsive")
    args = parser.parse_args()

    if _LOW_PRIORITY:
        print("[Low priority] CPU/GPU scheduling priority reduced.\n")

    if args.interactive:
        model, clip, vae = load_models()
        run_interactive(model, clip, vae, args)
        return

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            args.prompt = f.read().strip()
    if not args.prompt:
        parser.error("provide a prompt, --prompt-file, or use --interactive / -i")

    model, clip, vae = load_models()

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    if args.output is None:
        args.output = make_output_path()

    generate(model, clip, vae, args.prompt, args.width, args.height,
             args.steps, args.sampler, args.scheduler, args.seed, args.output)


if __name__ == "__main__":
    main()
