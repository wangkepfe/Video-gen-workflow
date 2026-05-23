import sys
import os
import argparse
import time
import random
import ctypes
import traceback

# Force UTF-8 on stdout/stderr so tqdm progress bars (█ glyphs etc.) don't
# crash with [Errno 22] Invalid argument on a non-UTF-8 Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

_LOW_PRIORITY = "--low-priority" in sys.argv

if _LOW_PRIORITY:
    try:
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    os.environ["CUDA_DEVICE_SCHEDULE"] = "YIELD"

import torch

if _LOW_PRIORITY:
    for _dll in ("cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll"):
        try:
            _cudart = ctypes.WinDLL(_dll)
            _cudart.cudaSetDeviceFlags(ctypes.c_uint(0x04))
            break
        except (OSError, Exception):
            continue

from diffusers import ZImagePipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "images")

_pipe = None


def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe
    print(f"Loading Z-Image-Turbo from {MODEL_ID}...")
    print("(first run will download ~12 GB from HuggingFace)\n")
    _pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    if _LOW_PRIORITY:
        _pipe.enable_model_cpu_offload()
        try:
            _pipe.enable_attention_slicing(slice_size=1)
        except (AttributeError, NotImplementedError):
            pass
    else:
        _pipe.to("cuda")
    print("Pipeline loaded. Ready.\n")
    return _pipe


def _yield_callback(pipe, step_index, timestep, callback_kwargs):
    """Flush GPU work and pause so DWM / other GUI apps get GPU time."""
    torch.cuda.synchronize()
    time.sleep(0.1)
    return callback_kwargs


@torch.inference_mode()
def generate(pipe, prompt, width, height, steps, seed, output_path):
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"Sampling ({steps - 1} NFEs, seed {seed})...")

    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    if _LOW_PRIORITY:
        kwargs["callback_on_step_end"] = _yield_callback

    image = pipe(**kwargs).images[0]

    image.save(output_path)
    print(f"Saved: {output_path}\n")


def make_output_path(suffix=""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"zimage_{ts}{suffix}.png")


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


def run_interactive(pipe, args):
    print("=" * 60)
    print("  Z-Image-Turbo — Interactive Mode")
    print(f"  Resolution: {args.width}x{args.height}  Steps: {args.steps} ({args.steps - 1} NFEs)")
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
                generate(pipe, prompt, args.width, args.height,
                         args.steps, seed, output_path)
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                print()


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo — Text to Image (interactive or single-shot)")
    parser.add_argument("prompt", type=str, nargs="?", default=None,
                        help="Text prompt for single-shot mode (or use --prompt-file)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read prompt from a text file (single-shot mode)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode: load pipeline once, then accept prompts in a loop")
    parser.add_argument("--width", type=int, default=704, help="Image width (default: 704)")
    parser.add_argument("--height", type=int, default=1280, help="Image height (default: 1280)")
    parser.add_argument("--steps", type=int, default=9,
                        help="Inference steps (default: 9, results in 8 DiT forwards)")
    parser.add_argument("--seed", type=int, default=None, help="Seed (default: random)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: output/images/zimage_TIMESTAMP.png)")
    parser.add_argument("--low-priority", action="store_true",
                        help="Reduce CPU/GPU priority so other applications stay responsive")
    args = parser.parse_args()

    if _LOW_PRIORITY:
        print("[Low priority] CPU/GPU scheduling priority reduced.\n")

    pipe = load_pipeline()

    if args.interactive:
        run_interactive(pipe, args)
        return

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            args.prompt = f.read().strip()
    if not args.prompt:
        parser.error("provide a prompt, --prompt-file, or use --interactive / -i")

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    if args.output is None:
        args.output = make_output_path()

    generate(pipe, args.prompt, args.width, args.height,
             args.steps, args.seed, args.output)


if __name__ == "__main__":
    main()
