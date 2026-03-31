import sys
import os
import argparse
import time
import random

COMFY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI_windows_portable", "ComfyUI")
sys.path.insert(0, COMFY_DIR)

import torch
import numpy as np
from PIL import Image

import comfy.sd
import comfy.sample
import comfy.utils
import comfy.model_management
import folder_paths

MODELS_DIR = os.path.join(COMFY_DIR, "models")
CHECKPOINT = os.path.join(MODELS_DIR, "checkpoints", "flux1-dev-fp8.safetensors")


def generate(prompt, width, height, steps, sampler_name, scheduler, seed, output_path):
    print(f"[1/5] Loading FLUX.1 Dev FP8 checkpoint...")
    model, clip, vae = comfy.sd.load_checkpoint_guess_config(
        CHECKPOINT, output_vae=True, output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )[:3]

    print(f"[2/5] Encoding prompt...")
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    print(f"[3/5] Preparing latent ({width}x{height})...")
    latent_image = torch.zeros([1, 4, height // 8, width // 8],
                               device=comfy.model_management.intermediate_device())
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    noise = comfy.sample.prepare_noise(latent_image, seed)

    print(f"[4/5] Sampling ({steps} steps, {sampler_name}/{scheduler}, CFG 1.0, seed {seed})...")
    samples = comfy.sample.sample(
        model, noise, steps, cfg=1.0,
        sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=negative,
        latent_image=latent_image, denoise=1.0, seed=seed,
    )

    print(f"[5/5] Decoding and saving...")
    images = vae.decode(samples)
    img_np = np.clip(255.0 * images[0].cpu().numpy(), 0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FLUX.1 Dev FP8 — Text to Image")
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument("--width", type=int, default=704, help="Image width (default: 704)")
    parser.add_argument("--height", type=int, default=1280, help="Image height (default: 1280)")
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps (default: 20)")
    parser.add_argument("--sampler", type=str, default="euler", help="Sampler (default: euler)")
    parser.add_argument("--scheduler", type=str, default="simple", help="Scheduler (default: simple)")
    parser.add_argument("--seed", type=int, default=None, help="Seed (default: random)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: output/images/flux_TIMESTAMP.png)")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    if args.output is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "images")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(out_dir, f"flux_{ts}.png")

    generate(args.prompt, args.width, args.height, args.steps, args.sampler, args.scheduler, args.seed, args.output)


if __name__ == "__main__":
    main()
