import sys
import os
import io
import re
import time
import random
import base64
import ctypes
import threading
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

from flask import Flask, request, jsonify, send_from_directory, render_template
from diffusers import ZImagePipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE_DIR, "output", "images")
PROMPTS_DIR = os.path.join(_BASE_DIR, "saved_prompts")

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))

_pipe = None
_lock = threading.Lock()


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
    torch.cuda.synchronize()
    time.sleep(0.1)
    return callback_kwargs


@torch.inference_mode()
def generate_image(prompt, width=768, height=1344, steps=9, seed=None):
    pipe = load_pipeline()
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"zimage_{ts}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    print(f"Saved: {filepath}")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return b64, filename


@app.route("/")
def index():
    return render_template("zimage.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is empty"}), 400

    width = data.get("width", 768)
    height = data.get("height", 1344)

    with _lock:
        try:
            b64, filename = generate_image(prompt, width=width, height=height)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return jsonify({"image": b64, "filename": filename})


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


def _safe_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
    return name[:120] or "untitled"


@app.route("/prompts/list")
def prompts_list():
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    names = sorted(
        (os.path.splitext(f)[0] for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")),
        key=str.lower,
    )
    return jsonify({"prompts": names})


@app.route("/prompts/save", methods=["POST"])
def prompts_save():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    text = (data.get("text") or "").strip()
    if not name:
        return jsonify({"error": "Name is empty"}), 400
    if not text:
        return jsonify({"error": "Prompt text is empty"}), 400

    os.makedirs(PROMPTS_DIR, exist_ok=True)
    filepath = os.path.join(PROMPTS_DIR, _safe_filename(name) + ".txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return jsonify({"ok": True})


@app.route("/prompts/load", methods=["POST"])
def prompts_load():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is empty"}), 400

    filepath = os.path.join(PROMPTS_DIR, _safe_filename(name) + ".txt")
    if not os.path.isfile(filepath):
        return jsonify({"error": "Prompt not found"}), 404
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return jsonify({"text": text})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--low-priority", action="store_true")
    args = parser.parse_args()

    load_pipeline()

    print(f"\n  Serving on http://0.0.0.0:{args.port}")
    print("  Open this URL on any device on your local network.\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
