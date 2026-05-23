import sys
import os
import time
import random
import uuid
import queue
import socket
import logging
import threading
import ctypes
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LOW_PRIORITY = "--low-priority" in sys.argv

if _LOW_PRIORITY:
    try:
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

# Make ComfyUI importable for the backend modules.
_COMFY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "ComfyUI_windows_portable", "ComfyUI")
sys.path.insert(0, _COMFY_DIR)

if _LOW_PRIORITY:
    for _dll in ("cudart64_12.dll", "cudart64_110.dll"):
        try:
            _cudart = ctypes.WinDLL(_dll)
            _cudart.cudaSetDeviceFlags(ctypes.c_uint(0x04))
            break
        except (OSError, Exception):
            continue

import video_backends
from video_backends import MissingFiles

from flask import Flask, request, jsonify, send_from_directory, render_template, abort

_BASE_DIR = Path(__file__).parent.resolve()
IMAGES_DIR = _BASE_DIR / "output" / "images"
VIDEOS_DIR = _BASE_DIR / "output" / "videos"

app = Flask(__name__, template_folder=str(_BASE_DIR / "templates"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".webm", ".mp4", ".mov", ".gif"}

# Per-backend model cache. Each entry is the opaque value returned by the
# backend's load_models(). Backends are loaded lazily on first use.
_models = {}
_models_lock = threading.Lock()

jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()


def _ensure_models(backend_name):
    """Load (or return cached) models for the given backend."""
    with _models_lock:
        if backend_name not in _models:
            backend = video_backends.get(backend_name)
            _models[backend_name] = backend.load_models()
        return _models[backend_name]


def _safe_under(base: Path, rel: str) -> Path:
    base_resolved = base.resolve()
    p = (base / rel).resolve()
    try:
        p.relative_to(base_resolved)
    except ValueError:
        raise ValueError("path escape")
    return p


@app.route("/")
def index():
    return render_template("video.html")


@app.route("/api/backends")
def api_backends():
    return jsonify({
        "default": video_backends.DEFAULT_BACKEND,
        "backends": video_backends.list_backends(),
    })


@app.route("/api/images")
def list_images():
    if not IMAGES_DIR.exists():
        return jsonify([])
    items = []
    for p in IMAGES_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(IMAGES_DIR).as_posix()
            folder = p.parent.relative_to(IMAGES_DIR).as_posix() or "."
            stat = p.stat()
            items.append({
                "name": p.name,
                "path": rel,
                "folder": folder,
                "url": f"/image/{rel}",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(items)


@app.route("/image/<path:relpath>")
def serve_image(relpath):
    try:
        full = _safe_under(IMAGES_DIR, relpath)
    except ValueError:
        abort(403)
    if not full.is_file():
        abort(404)
    return send_from_directory(full.parent, full.name)


@app.route("/api/videos")
def list_videos():
    if not VIDEOS_DIR.exists():
        return jsonify([])
    items = []
    for p in VIDEOS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            rel = p.relative_to(VIDEOS_DIR).as_posix()
            folder = p.parent.relative_to(VIDEOS_DIR).as_posix() or "."
            stat = p.stat()
            items.append({
                "name": p.name,
                "path": rel,
                "folder": folder,
                "url": f"/video/{rel}",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify(items)


@app.route("/video/<path:relpath>")
def serve_video(relpath):
    try:
        full = _safe_under(VIDEOS_DIR, relpath)
    except ValueError:
        abort(403)
    if not full.is_file():
        abort(404)
    return send_from_directory(full.parent, full.name)


def _coerce_settings(backend_mod, raw):
    """Validate + coerce a settings dict against the backend's FIELDS schema.

    Returns (settings_dict, error_message_or_none). Numeric fields are parsed,
    blank/optional values fall back to the field default, "auto" passes through
    untouched so backends can interpret it.
    """
    settings = {}
    for f in backend_mod.FIELDS:
        if f.type == "ratio_buttons":
            # purely a UI helper, not a real setting
            continue
        v = raw.get(f.id)
        if v in (None, ""):
            v = f.default
        try:
            if f.type == "int":
                if isinstance(v, str) and v.strip().lower() == "auto":
                    pass
                elif v is not None and v != "":
                    v = int(v)
            elif f.type == "float":
                if v is not None and v != "":
                    v = float(v)
        except (TypeError, ValueError):
            return None, f"Invalid value for {f.id!r}: {raw.get(f.id)!r}"
        settings[f.id] = v
    return settings, None


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True)

    backend_name = (data.get("backend") or video_backends.DEFAULT_BACKEND).strip()
    try:
        backend = video_backends.get(backend_name)
    except KeyError as e:
        return jsonify({"error": str(e)}), 400

    prompt = (data.get("prompt") or "").strip()
    image_paths = data.get("images") or []
    negative = (data.get("negative") or "").strip() or backend.DEFAULT_NEGATIVE

    settings, err = _coerce_settings(backend, data)
    if err is not None:
        return jsonify({"error": err}), 400

    if not prompt:
        return jsonify({"error": "Prompt is empty"}), 400
    if not image_paths:
        return jsonify({"error": "No images selected"}), 400

    # Seed handling: blank/None means random, otherwise an int.
    seed_in = settings.get("seed")
    if isinstance(seed_in, str):
        seed_in = seed_in.strip()
    if seed_in in (None, ""):
        base_seed = None
    else:
        try:
            base_seed = int(seed_in)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid seed"}), 400

    batch_id = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = VIDEOS_DIR / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    job_ids = []
    for rel in image_paths:
        try:
            img_path = _safe_under(IMAGES_DIR, rel)
        except ValueError:
            continue
        if not img_path.is_file():
            continue
        stem = img_path.stem
        out_path = batch_dir / f"{stem}.webm"
        n = 1
        while out_path.exists():
            n += 1
            out_path = batch_dir / f"{stem}_{n}.webm"

        job_seed = base_seed if base_seed is not None else random.randint(0, 2**32 - 1)
        job_id = uuid.uuid4().hex[:8]

        # Per-job settings (one independent dict per job so seeds can differ).
        job_settings = dict(settings)
        job_settings["seed"] = job_seed

        job = {
            "id": job_id,
            "backend": backend_name,
            "status": "queued",
            "image": rel,
            "image_url": f"/image/{rel}",
            "prompt": prompt,
            "output": out_path.relative_to(VIDEOS_DIR).as_posix(),
            "video_url": None,
            "error": None,
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "settings": job_settings,
        }
        with jobs_lock:
            jobs[job_id] = job
        job_queue.put((job_id, backend_name, str(img_path), prompt, negative, str(out_path)))
        job_ids.append(job_id)

    if not job_ids:
        return jsonify({"error": "No valid images found"}), 400

    return jsonify({"batch": batch_id, "jobs": job_ids, "backend": backend_name})


@app.route("/api/jobs")
def api_jobs():
    with jobs_lock:
        return jsonify(sorted(jobs.values(), key=lambda j: j["queued_at"], reverse=True))


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            return jsonify({"error": "not found"}), 404
        return jsonify(j)


@app.route("/api/jobs/clear", methods=["POST"])
def api_jobs_clear():
    with jobs_lock:
        for jid in list(jobs.keys()):
            if jobs[jid]["status"] in ("done", "failed"):
                del jobs[jid]
    return jsonify({"ok": True})


def worker():
    while True:
        job_id, backend_name, image_path, prompt, negative, output_path = job_queue.get()
        with jobs_lock:
            j = jobs.get(job_id)
            if j is None:
                job_queue.task_done()
                continue
            j["status"] = "loading_models" if backend_name not in _models else "running"
            j["started_at"] = time.time()
            settings = dict(j["settings"])

        try:
            models = _ensure_models(backend_name)
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "running"
            backend = video_backends.get(backend_name)
            backend.generate(models, image_path, prompt, negative, settings, output_path)
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "done"
                    rel = jobs[job_id]["output"]
                    jobs[job_id]["video_url"] = f"/video/{rel}"
        except MissingFiles as e:
            print(f"[Job {job_id}] Missing files:\n{e}")
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = str(e)
        except Exception as e:
            print(f"[Job {job_id}] Error: {e}")
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = str(e)
        finally:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["finished_at"] = time.time()
            job_queue.task_done()


_worker_thread = threading.Thread(target=worker, daemon=True)
_worker_thread.start()


def _lan_ip():
    """Best-effort detection of the primary LAN IPv4 address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = None
    finally:
        s.close()
    return ip


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--low-priority", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="Show INFO-level logs (per-request HTTP access logs).")
    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

    if _LOW_PRIORITY:
        print("[Low priority] CPU/GPU scheduling priority reduced.\n")

    print(f"  Images:  {IMAGES_DIR}")
    print(f"  Videos:  {VIDEOS_DIR}")
    print(f"  Backends: {', '.join(video_backends.BACKENDS.keys())} "
          f"(default: {video_backends.DEFAULT_BACKEND})")
    print()
    print(f"  This computer:  http://127.0.0.1:{args.port}")
    lan = _lan_ip()
    if lan and args.host in ("0.0.0.0", ""):
        print(f"  Phone (Wi-Fi):  http://{lan}:{args.port}")
        print(f"     -> Make sure Windows Firewall allows inbound TCP {args.port}")
    print()
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
