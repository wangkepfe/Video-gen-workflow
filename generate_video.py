"""CLI front-end for the video backends.

Defaults to Wan2.2; pass --backend ltx to use LTX-Video 13B 0.9.8 distilled.
The web app shares the same backend modules in `video_backends/`.
"""
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

COMFY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ComfyUI_windows_portable", "ComfyUI")
sys.path.insert(0, COMFY_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if _LOW_PRIORITY:
    for _dll in ("cudart64_12.dll", "cudart64_110.dll"):
        try:
            _cudart = ctypes.WinDLL(_dll)
            _cudart.cudaSetDeviceFlags(ctypes.c_uint(0x04))
            break
        except (OSError, Exception):
            continue

import video_backends

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output", "videos")

# Re-exported so older scripts that did `from generate_video import DEFAULT_NEGATIVE`
# don't break. New code should pull this from the chosen backend module.
DEFAULT_NEGATIVE = video_backends.get(video_backends.DEFAULT_BACKEND).DEFAULT_NEGATIVE


def read_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def make_output_path(backend_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"{backend_name}_{ts}.webm")


def _settings_from_args(backend, args):
    """Pull values for the backend's FIELDS from argparse args, falling back
    to each field's declared default. CLI flags use --kebab-case; field ids
    use snake_case (e.g. --boundary-step ↔ boundary_step)."""
    out = {}
    for f in backend.FIELDS:
        if f.type == "ratio_buttons":
            continue
        attr = f.id  # argparse stores dashes as underscores already
        v = getattr(args, attr, None)
        if v is None or v == "":
            v = f.default
        out[f.id] = v
    return out


def _add_backend_args(parser, backend):
    """Add one CLI flag per backend field."""
    seen = set()
    for f in backend.FIELDS:
        if f.type == "ratio_buttons" or f.id in seen:
            continue
        seen.add(f.id)
        flag = "--" + f.id.replace("_", "-")
        kwargs = {"default": None, "help": (f.help or f.label)}
        if f.type == "int":
            kwargs["type"] = int
        elif f.type == "float":
            kwargs["type"] = float
        else:
            kwargs["type"] = str
        if f.type == "select" and f.options:
            kwargs["choices"] = f.options
        parser.add_argument(flag, **kwargs)


def run_interactive(backend, models, args):
    print("=" * 60)
    print(f"  {backend.DISPLAY_NAME} - Interactive Mode")
    print("=" * 60)
    print("Enter: image_path  prompt_file_or_text  [negative_file_or_text]")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    base_settings = _settings_from_args(backend, args)

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

        if negative_input and os.path.isfile(negative_input):
            neg = read_prompt(negative_input)
        else:
            neg = negative_input or args.negative or backend.DEFAULT_NEGATIVE

        settings = dict(base_settings)
        if settings.get("seed") in (None, ""):
            settings["seed"] = random.randint(0, 2**32 - 1)

        output_path = make_output_path(backend.NAME)

        try:
            backend.generate(models, image_path, prompt, neg, settings, output_path)
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


def process_folder(backend, models, folder_path, prompt, negative, args):
    files = sorted(
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        print(f"[ERROR] No images found in: {folder_path}")
        sys.exit(1)

    folder_name = os.path.basename(os.path.normpath(folder_path))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output", "videos",
                           f"batch_{backend.NAME}_{folder_name}")
    os.makedirs(out_dir, exist_ok=True)

    prompt_dir = getattr(args, "prompt_dir", None)
    if prompt_dir:
        print(f"Prompt directory: {prompt_dir}")

    print(f"Backend: {backend.DISPLAY_NAME}")
    print(f"Processing {len(files)} images from: {folder_path}")
    print(f"Output folder: {out_dir}\n")

    base_settings = _settings_from_args(backend, args)

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

        settings = dict(base_settings)
        if settings.get("seed") in (None, ""):
            settings["seed"] = random.randint(0, 2**32 - 1)

        print(f"--- [{i}/{len(files)}] {fname}  prompt: {prompt_src} ---")
        try:
            backend.generate(models, image_path, img_prompt, negative, settings, output_path)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}\n")
            continue

    print(f"Done. {len(files)} videos generated -> {out_dir}")


def main():
    # Two-pass argparse: first parse just --backend so we can register the
    # backend-specific flags before final parsing.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", type=str, default=video_backends.DEFAULT_BACKEND,
                     choices=sorted(video_backends.BACKENDS.keys()))
    pre_args, _ = pre.parse_known_args()
    backend = video_backends.get(pre_args.backend)

    parser = argparse.ArgumentParser(
        description=f"Image-to-Video generator (default backend: "
                    f"{video_backends.DEFAULT_BACKEND})")
    parser.add_argument("--backend", type=str, default=video_backends.DEFAULT_BACKEND,
                        choices=sorted(video_backends.BACKENDS.keys()),
                        help="Which video model to use.")
    parser.add_argument("image", type=str, nargs="?", default=None,
                        help="Input image path or folder of images")
    parser.add_argument("prompt", type=str, nargs="?", default=None,
                        help="Motion prompt or use --prompt-file")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read positive prompt from a text file")
    parser.add_argument("--negative-file", type=str, default=None,
                        help="Read negative prompt from a text file")
    parser.add_argument("--negative", type=str, default=backend.DEFAULT_NEGATIVE,
                        help="Negative prompt text (default: backend's default)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive mode: load models once, accept inputs in a loop")
    parser.add_argument("--prompt-dir", type=str, default=None,
                        help="Directory of per-image prompt .txt files (matched by image basename)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (single image mode)")
    parser.add_argument("--low-priority", action="store_true",
                        help="Reduce CPU/GPU priority so other applications stay responsive")

    _add_backend_args(parser, backend)
    args = parser.parse_args()

    if _LOW_PRIORITY:
        print("[Low priority] CPU/GPU scheduling priority reduced.\n")

    # Resolve prompt/negative from files if specified.
    if args.prompt_file:
        args.prompt = read_prompt(args.prompt_file)
    if args.negative_file:
        args.negative = read_prompt(args.negative_file)

    # Wan2.2 needs the --shift to be applied at model-load time; pull it off
    # the args dict so we can pass it to load_models() when supported.
    load_kwargs = {}
    if backend.NAME == "wan22":
        shift = getattr(args, "shift", None)
        if shift is not None:
            load_kwargs["shift"] = float(shift)

    if args.interactive:
        models = backend.load_models(**load_kwargs)
        run_interactive(backend, models, args)
        return

    if not args.image:
        parser.error("provide image + prompt (or --prompt-file), or use -i for interactive mode")

    if os.path.isdir(args.image):
        if not args.prompt and not args.prompt_dir:
            parser.error("folder mode requires a prompt (or --prompt-file / --prompt-dir)")
        models = backend.load_models(**load_kwargs)
        process_folder(backend, models, args.image, args.prompt or "", args.negative, args)
        return

    if not args.prompt:
        parser.error("provide image + prompt (or --prompt-file), or use -i for interactive mode")

    if not os.path.isfile(args.image):
        print(f"[ERROR] Input not found: {args.image}")
        sys.exit(1)

    models = backend.load_models(**load_kwargs)

    output_path = args.output or make_output_path(backend.NAME)

    settings = _settings_from_args(backend, args)
    if settings.get("seed") in (None, ""):
        settings["seed"] = random.randint(0, 2**32 - 1)

    backend.generate(models, args.image, args.prompt, args.negative, settings, output_path)


if __name__ == "__main__":
    main()
