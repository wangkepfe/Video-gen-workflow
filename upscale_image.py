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
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def seedvr2_upscale(input_path, output_path, resolution, dit_model, color_correction, seed, low_priority=False):
    cmd = [
        PYTHON_EXE, SEEDVR2_CLI,
        input_path,
        "--output", output_path,
        "--resolution", str(resolution),
        "--dit_model", dit_model,
        "--color_correction", color_correction,
        "--seed", str(seed),
        "--batch_size", "1",
        "--model_dir", SEEDVR2_MODEL_DIR,
    ]
    kwargs = {"cwd": os.path.dirname(SEEDVR2_CLI)}
    if low_priority:
        kwargs["creationflags"] = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"SeedVR2 failed with exit code {result.returncode}")


def upscale_single(input_path, name_no_ext, out_4x_dir, out_final_dir, args):
    """
    Pipeline: input → 2x downscale → SeedVR2 4x upscale → save 4x
                                                          → 2x downscale → save final
    """
    img = Image.open(input_path)
    orig_w, orig_h = img.size

    ds_w, ds_h = orig_w // 2, orig_h // 2
    print(f"  [1/4] Downscale 2x: {orig_w}x{orig_h} → {ds_w}x{ds_h}")
    downscaled = img.resize((ds_w, ds_h), Image.LANCZOS)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    downscaled.save(tmp_path)

    up4x_res = min(ds_w, ds_h) * 4
    path_4x = os.path.join(out_4x_dir, f"{name_no_ext}_4x.png")
    print(f"  [2/4] SeedVR2 4x upscale → {ds_w*4}x{ds_h*4} (target shortest edge {up4x_res})")
    seedvr2_upscale(tmp_path, path_4x, up4x_res, args.dit_model, args.color_correction, args.seed,
                    low_priority=args.low_priority)
    os.unlink(tmp_path)

    print(f"  [3/4] Saved 4x: {path_4x}")

    img_4x = Image.open(path_4x)
    final_w, final_h = img_4x.size[0] // 2, img_4x.size[1] // 2
    print(f"  [4/4] Downscale 2x: {img_4x.size[0]}x{img_4x.size[1]} → {final_w}x{final_h}")
    final = img_4x.resize((final_w, final_h), Image.LANCZOS)
    path_final = os.path.join(out_final_dir, f"{name_no_ext}.png")
    final.save(path_final)
    print(f"  Saved final: {path_final}\n")


def process_single(input_path, args):
    name_no_ext = os.path.splitext(os.path.basename(input_path))[0]
    ts = time.strftime("%Y%m%d_%H%M%S")

    out_4x_dir = os.path.join(ROOT, "output", "images", "upscaled_4x")
    out_final_dir = os.path.join(ROOT, "output", "images", "upscaled_final")
    os.makedirs(out_4x_dir, exist_ok=True)
    os.makedirs(out_final_dir, exist_ok=True)

    upscale_single(input_path, f"{name_no_ext}_{ts}", out_4x_dir, out_final_dir, args)


def process_folder(folder_path, args):
    files = sorted(
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        print(f"[ERROR] No images found in: {folder_path}")
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(folder_path))
    out_4x_dir = os.path.join(ROOT, "output", "images", f"upscaled_4x_{folder_name}")
    out_final_dir = os.path.join(ROOT, "output", "images", f"upscaled_final_{folder_name}")
    os.makedirs(out_4x_dir, exist_ok=True)
    os.makedirs(out_final_dir, exist_ok=True)

    print(f"Processing {len(files)} images from: {folder_path}")
    print(f"  4x upscaled → {out_4x_dir}")
    print(f"  Final (2x down) → {out_final_dir}\n")

    for i, fname in enumerate(files, 1):
        input_path = os.path.join(folder_path, fname)
        name_no_ext = os.path.splitext(fname)[0]

        print(f"--- [{i}/{len(files)}] {fname} ---")
        try:
            upscale_single(input_path, name_no_ext, out_4x_dir, out_final_dir, args)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}\n")
            continue

    print(f"Done. {len(files)} images processed.")
    print(f"  4x upscaled → {out_4x_dir}")
    print(f"  Final (2x down) → {out_final_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="SeedVR2 Creative Upscale: 2x down → 4x AI up → save 4x + 2x down final")
    parser.add_argument("input", type=str, help="Input image path or folder of images")
    parser.add_argument("--dit-model", type=str, default="seedvr2_ema_7b_fp16.safetensors",
                        help="DiT model (default: seedvr2_ema_7b_fp16.safetensors)")
    parser.add_argument("--color-correction", type=str, default="lab",
                        choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                        help="Color correction method (default: lab)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--low-priority", action="store_true",
                        help="Reduce subprocess priority so other applications stay responsive")
    args = parser.parse_args()

    if args.low_priority:
        import ctypes
        try:
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass
        print("[Low priority] Process/subprocess priority reduced.\n")

    if os.path.isdir(args.input):
        process_folder(args.input, args)
    elif os.path.isfile(args.input):
        process_single(args.input, args)
    else:
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
