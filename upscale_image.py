import sys
import os
import argparse
import time
import subprocess
import tempfile
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
COMFY_DIR = os.path.join(ROOT, "ComfyUI_windows_portable", "ComfyUI")
SEEDVR2_CLI = os.path.join(COMFY_DIR, "custom_nodes", "seedvr2_videoupscaler", "inference_cli.py")
PYTHON_EXE = os.path.join(ROOT, "ComfyUI_windows_portable", "python_embeded", "python.exe")
SEEDVR2_MODEL_DIR = os.path.join(COMFY_DIR, "models", "SEEDVR2")


def upscale(input_path, downscale_factor, resolution, dit_model, color_correction, seed, output_path):
    img = Image.open(input_path)
    orig_w, orig_h = img.size
    new_w = orig_w // downscale_factor
    new_h = orig_h // downscale_factor
    print(f"[1/3] Downscaling {orig_w}x{orig_h} → {new_w}x{new_h} ({downscale_factor}x, lanczos)...")

    downscaled = img.resize((new_w, new_h), Image.LANCZOS)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    downscaled.save(tmp_path)

    print(f"[2/3] SeedVR2 upscaling to resolution {resolution} (target shortest edge)...")
    cmd = [
        PYTHON_EXE, SEEDVR2_CLI,
        tmp_path,
        "--output", output_path,
        "--resolution", str(resolution),
        "--dit_model", dit_model,
        "--color_correction", color_correction,
        "--seed", str(seed),
        "--batch_size", "1",
        "--model_dir", SEEDVR2_MODEL_DIR,
    ]

    result = subprocess.run(cmd, cwd=os.path.dirname(SEEDVR2_CLI))

    os.unlink(tmp_path)

    if result.returncode != 0:
        print(f"[ERROR] SeedVR2 failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"[3/3] Done!")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SeedVR2 Creative Upscale — Downscale + AI Upscale")
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("--downscale", type=int, default=4, help="Downscale factor before upscaling (default: 4)")
    parser.add_argument("--resolution", type=int, default=704, help="Target shortest-edge resolution (default: 704)")
    parser.add_argument("--dit-model", type=str, default="seedvr2_ema_7b_fp16.safetensors",
                        help="DiT model (default: seedvr2_ema_7b_fp16.safetensors)")
    parser.add_argument("--color-correction", type=str, default="lab",
                        choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                        help="Color correction method (default: lab)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: output/images/upscaled_TIMESTAMP.png)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        out_dir = os.path.join(ROOT, "output", "images")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(out_dir, f"upscaled_{ts}.png")

    upscale(args.input, args.downscale, args.resolution, args.dit_model, args.color_correction, args.seed, args.output)


if __name__ == "__main__":
    main()
