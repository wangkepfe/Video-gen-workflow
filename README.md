# Video Generation Workflow

Text-to-Image (FLUX.1 Dev FP8) and Image-to-Video (Wan2.2 **14B I2V** MoE) pipeline running locally on an RTX 4090 via ComfyUI, with a Flask web app for easy batch generation from your phone or browser.

## Prerequisites

- **GPU**: NVIDIA RTX 4090 (24 GB VRAM)
- **OS**: Windows 10 or later
- **Disk space**: ~55 GB free (ComfyUI ~2 GB + models ~50 GB + output)
- **Internet**: Required for initial model downloads
- **7-Zip** (recommended): Install from https://www.7-zip.org/ for fast extraction. The setup script falls back to Windows `tar` if 7z is unavailable.

## Quick Start

### 1. Install ComfyUI

```
setup_comfyui.bat
```

Downloads the official ComfyUI portable build for NVIDIA GPUs (~1.9 GB) and extracts it.

### 2. Download Models

```
download_models.bat
```

Downloads the model files (~50 GB total). Resumes interrupted downloads automatically.

| Model | Size | Purpose |
|-------|------|---------|
| `flux1-dev-fp8.safetensors` | 17.2 GB | FLUX.1 Dev text-to-image (fp8) |
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | ~14 GB | Wan2.2 14B I2V high-noise expert |
| `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | ~14 GB | Wan2.2 14B I2V low-noise expert |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 4.9 GB | UMT5-XXL text encoder |
| `wan2.2_vae.safetensors` | ~200 MB | Wan2.2 video VAE |

> The Wan2.2 14B I2V model uses a **Mixture-of-Experts** design: the high-noise expert handles early denoising steps (large noise), then the low-noise expert refines the result. ComfyUI swaps them on the GPU so peak VRAM stays around 20 GB.

### 3. Launch the web app

```
web_video.bat
```

You'll see something like:

```
  This computer:  http://127.0.0.1:5001
  Phone (Wi-Fi):  http://192.168.1.50:5001
     -> Make sure Windows Firewall allows inbound TCP 5001
```

Open the link on your computer or phone (must be on the same Wi-Fi). The first generation triggers model loading (~2-3 min), then subsequent jobs reuse the loaded models.

## Web app features

- **Image picker** — recursive grid of `output/images/` (groups by subfolder)
- **Multi-select** — click thumbnails (Shift+click for range, Ctrl+A for all, per-folder buttons)
- **Prompt + negative prompt** — single prompt applies to whole batch
- **Settings** — width/height/frames/fps/steps/CFG/sampler/scheduler/seed/boundary-step
- **Aspect ratio buttons** — 16:9 / 9:16 / 1:1
- **Job queue** — sticky panel at bottom; live status (queued → loading model → running → done)
- **Video gallery** — recursive scan of `output/videos/`, click any thumbnail to play in modal
- Videos save to `output/videos/batch_YYYYMMDD_HHMMSS/<image_basename>.webm`

## Phone access (LAN)

The web app binds to `0.0.0.0:5001` so any device on your Wi-Fi can connect.

1. **Find your PC's LAN IP** — `web_video.bat` prints it on startup, or run `ipconfig` and look for "IPv4 Address" under your Wi-Fi adapter.
2. **Allow the port through Windows Firewall** (one-time, run PowerShell as Administrator):
   ```powershell
   New-NetFirewallRule -DisplayName "Wan2.2 Web App" -Direction Inbound -Protocol TCP -LocalPort 5001 -Action Allow
   ```
   Or: when you first launch `web_video.bat`, Windows pops up a "Windows Defender Firewall" prompt — check both **Private** and **Public** networks if you want phone access, then click **Allow access**.
3. **Open `http://<your-pc-ip>:5001` on your phone** while on the same Wi-Fi.

## Wan2.2 14B I2V settings

Defaults (good starting point for 5s @ 720p video on RTX 4090):

| Setting | Default | Notes |
|---------|---------|-------|
| Resolution | 1280×720 | model native; portrait works (720×1280) |
| Frames | 120 | = 5s at 24fps |
| FPS | 24 | |
| Steps | 30 | split 15 high-noise + 15 low-noise |
| CFG | 3.5 | 14B I2V uses lower CFG than 5B |
| Sampler | euler | recommended for MoE; `uni_pc` works |
| Scheduler | simple | |
| Shift | 5.0 | ModelSamplingSD3, applied at load time |
| Boundary step | auto | step at which expert switches; default = steps/2 |

Generation time on a 4090: ~9 minutes for a 5-second 720p clip at 30 steps.

## Command-line usage (no web app)

`generate_video.py` still works as a CLI for scripting / batch jobs:

```
generate_video.bat <image> "<prompt>" [flags]
```

Useful flags: `--prompt-file`, `--negative-file`, `--frames`, `--steps`, `--cfg`, `--shift`, `--seed`, `--low-priority`, `-i` (interactive mode keeps models loaded).

Folder mode: pass a directory as `<image>` and it processes every image in it, saving to `output/videos/batch_<folder_name>/`. Optionally pair with `--prompt-dir` for per-image prompts (filename basename match).

## File Structure

```
Video-gen-workflow/
├── setup_comfyui.bat              # Step 1: Install ComfyUI
├── download_models.bat            # Step 2: Download all models
├── web_video.bat                  # Step 3: Launch web app (recommended)
├── run_comfyui.bat                # Or: launch ComfyUI directly
├── generate_video.py              # I2V engine (14B MoE)
├── generate_video.bat             # CLI launcher
├── web_video.py                   # Flask backend
├── web_zimage.py                  # Z-Image text-to-image web app
├── templates/
│   ├── video.html                 # Web app UI
│   └── zimage.html                # Z-Image UI
├── output/
│   ├── images/                    # Source images for I2V
│   └── videos/                    # Generated videos (organized by batch)
└── ComfyUI_windows_portable/      # Created by setup script
    └── ComfyUI/models/
        ├── checkpoints/           # FLUX.1 Dev FP8
        ├── diffusion_models/      # Wan2.2 14B high + low noise
        ├── text_encoders/         # UMT5-XXL
        └── vae/                   # Wan2.2 VAE
```

## Troubleshooting

**"CUDA out of memory"**
- Close other GPU apps (browsers with hardware acceleration, games)
- Reduce frames (try 80 instead of 120) or resolution (832×480)
- The 14B MoE swaps experts on/off the GPU automatically — peak VRAM ≈ 20 GB

**"Missing high-noise expert" error on launch**
- Run `download_models.bat` first. The 14B I2V needs **both** `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` and `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`.

**Phone can't connect**
- Same Wi-Fi network? (5 GHz vs 2.4 GHz on the same SSID is fine; different SSIDs may not route.)
- Firewall: see the "Phone access" section above.
- Test from your computer first at `http://<lan-ip>:5001` (not 127.0.0.1) to confirm the bind works.

**Download interrupted**
- Just run `download_models.bat` again — `curl -C -` resumes automatically.

**ComfyUI shows red nodes / model not found**
- Verify all 5 model files exist in the right subfolders under `ComfyUI_windows_portable/ComfyUI/models/`.

## Model Sources

| Model | Repository |
|-------|------------|
| FLUX.1 Dev FP8 | https://huggingface.co/Comfy-Org/flux1-dev |
| Wan2.2 14B I2V (both experts) | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged |
| UMT5-XXL FP8 | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged |
| Wan2.2 VAE | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged |
