# Video Generation Workflow

Text-to-Image (FLUX.1 Dev FP8) and Image-to-Video (Wan2.2 5B) pipeline running locally on an RTX 4090 via ComfyUI.

## Prerequisites

- **GPU**: NVIDIA RTX 4090 (24 GB VRAM)
- **OS**: Windows 10 or later
- **Disk space**: ~40 GB free (ComfyUI ~2 GB + models ~33 GB + output)
- **Internet**: Required for initial model downloads
- **7-Zip** (recommended): Install from https://www.7-zip.org/ for fast extraction. The setup script falls back to Windows `tar` if 7z is unavailable.

## Quick Start

### 1. Install ComfyUI

```
setup_comfyui.bat
```

Downloads the official ComfyUI portable build for NVIDIA GPUs (~1.9 GB) and extracts it. Takes about 2 minutes.

### 2. Download Models

```
download_models.bat
```

Downloads all four model files (~33 GB total). The script supports **resume** -- if a download is interrupted, just run it again. Existing files are skipped automatically.

| Model | Size | Purpose |
|-------|------|---------|
| `flux1-dev-fp8.safetensors` | 17.2 GB | FLUX.1 Dev text-to-image (fp8 quantized) |
| `wan2.2_ti2v_5B_fp16.safetensors` | 10.7 GB | Wan2.2 5B image-to-video diffusion model |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 4.9 GB | UMT5-XXL text encoder for Wan2.2 |
| `wan2.2_vae.safetensors` | ~200 MB | Wan2.2 video VAE |

### 3. Launch ComfyUI

```
run_comfyui.bat
```

Starts ComfyUI and opens http://127.0.0.1:8188 in your browser.

## Workflows

Both workflow files are in the `workflows/` folder. Load them in ComfyUI via **Workflows > Open** (or Ctrl+O).

### Workflow 1: Text-to-Image (FLUX.1 Dev FP8)

**File**: `workflows/flux_dev_fp8_t2i.json`

1. Load the workflow in ComfyUI
2. Type your prompt in the green **Positive Prompt** node
3. Adjust resolution in **EmptyLatentImage** if desired (default 1024x1024)
4. Click **Queue Prompt** (or Ctrl+Enter)
5. Image appears in ~5-8 seconds, saved to `output/images/`

**Key settings**:
- CFG scale is fixed at **1.0** (FLUX ignores other values)
- Negative prompts have no effect on FLUX models
- Good resolutions: 1024x1024, 1152x896, 896x1152

**Tips for better images**:
- Be descriptive: include lighting, mood, camera angle, style
- Mention "photorealistic", "8k", "highly detailed" for photo-quality output
- FLUX excels at text rendering in images -- you can include text in prompts

### Workflow 2: Image-to-Video (Wan2.2 5B)

**File**: `workflows/wan22_5b_i2v.json`

1. Load the workflow in ComfyUI
2. In the **LoadImage** node, select the image you want to animate (e.g. output from Workflow 1)
3. Write a **motion prompt** in the green Positive Prompt node -- describe the *movement* you want, not the image contents
4. Adjust frame count in **Wan22ImageToVideoLatent** (default 81 frames = ~3.4 seconds at 24fps)
5. Click **Queue Prompt**
6. Video is saved as both animated WEBP and WEBM to `output/videos/`

**Key settings**:
- Resolution: **1280x704** (optimal for this model)
- Sampler: uni_pc, 30 steps, CFG 5.0
- Shift: 8.0 (via ModelSamplingSD3)
- Output FPS: 24

**Frame count guide**:
| Frames | Duration | Generation time (approx.) |
|--------|----------|---------------------------|
| 41 | ~1.7s | ~40 seconds |
| 81 | ~3.4s | ~1.5 minutes |
| 121 | ~5.0s | ~2.5 minutes |

**Tips for better videos**:
- The prompt should describe **motion**: "walking slowly", "hair blowing in the wind", "camera panning left"
- Keep motions simple and physically plausible for best results
- The negative prompt helps -- keep it as-is to avoid common artifacts
- For the source image, use a resolution close to 16:9 aspect ratio for best results with the 1280x704 output

## Typical Workflow

```
1. Generate image with FLUX         (flux_dev_fp8_t2i.json)
         |                           → saved to output/images/
         v
2. Load image in Wan2.2 workflow    (wan22_5b_i2v.json)
   (select from output/images/)      → the input directory points here
         |
         v
3. Describe the motion you want
         |
         v
4. Get your video!                   → saved to output/videos/
```

## File Structure

```
Video-gen-workflow/
├── setup_comfyui.bat              # Step 1: Install ComfyUI
├── download_models.bat            # Step 2: Download models
├── run_comfyui.bat                # Step 3: Launch
├── README.md                      # This file
├── workflows/
│   ├── flux_dev_fp8_t2i.json      # Text-to-Image workflow
│   └── wan22_5b_i2v.json          # Image-to-Video workflow
├── output/
│   ├── images/                    # FLUX generated images go here
│   └── videos/                    # Wan2.2 generated videos go here
└── ComfyUI_windows_portable/      # Created by setup script
    └── ComfyUI/
        └── models/
            ├── checkpoints/       # FLUX.1 Dev FP8
            ├── diffusion_models/  # Wan2.2 5B
            ├── text_encoders/     # UMT5-XXL
            └── vae/               # Wan2.2 VAE
```

## Troubleshooting

**"CUDA out of memory"**
- Close other GPU-intensive applications
- Reduce frame count in the Wan2.2 workflow (try 41 instead of 81)
- Reduce resolution (try 832x480 instead of 1280x704)
- ComfyUI automatically unloads models between workflows, but generation within a single workflow must fit in 24 GB

**Download interrupted**
- Just run `download_models.bat` again. It uses `curl -C -` for automatic resume and skips already-downloaded files.

**ComfyUI shows red nodes**
- This means a model file is missing or misnamed. Double-check that all four model files are in the correct subdirectories under `ComfyUI_windows_portable/ComfyUI/models/`.

**7z extraction fails**
- Install 7-Zip from https://www.7-zip.org/ and ensure `7z` is in your PATH, then re-run `setup_comfyui.bat`.
- Alternatively, manually extract `ComfyUI_windows_portable_nvidia.7z` into the `Video-gen-workflow` folder.

## Model Sources

| Model | Repository |
|-------|-----------|
| FLUX.1 Dev FP8 | https://huggingface.co/Comfy-Org/flux1-dev |
| Wan2.2 5B TI2V | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged |
| UMT5-XXL FP8 | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged |
| Wan2.2 VAE | https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged |
