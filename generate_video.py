import sys
import os
import argparse
import time
import random
from fractions import Fraction

COMFY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI_windows_portable", "ComfyUI")
sys.path.insert(0, COMFY_DIR)

import torch
import numpy as np
import av
from PIL import Image

import comfy.sd
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.latent_formats
import folder_paths

MODELS_DIR = os.path.join(COMFY_DIR, "models")
UNET_PATH = os.path.join(MODELS_DIR, "diffusion_models", "wan2.2_ti2v_5B_fp16.safetensors")
CLIP_PATH = os.path.join(MODELS_DIR, "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
VAE_PATH = os.path.join(MODELS_DIR, "vae", "wan2.2_vae.safetensors")


def build_i2v_latent(vae, start_image, width, height, length):
    """Replicates Wan22ImageToVideoLatent logic."""
    latent_length = ((length - 1) // 4) + 1
    device = comfy.model_management.intermediate_device()

    latent = torch.zeros([1, 48, latent_length, height // 16, width // 16], device=device)
    mask = torch.ones([1, 1, latent_length, latent.shape[-2], latent.shape[-1]], device=device)

    start_image = comfy.utils.common_upscale(
        start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)

    latent_temp = vae.encode(start_image)
    latent[:, :, :latent_temp.shape[-3]] = latent_temp
    mask[:, :, :latent_temp.shape[-3]] *= 0.0

    latent_format = comfy.latent_formats.Wan22()
    latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)

    return {"samples": latent, "noise_mask": mask}


def save_webm(frames, output_path, fps):
    """Save NHWC float [0,1] tensor as VP9 WEBM."""
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libvpx-vp9", rate=Fraction(round(fps * 1000), 1000))
    stream.width = frames.shape[-2]
    stream.height = frames.shape[-3]
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = 0
    stream.options = {"crf": "16"}

    for frame_tensor in frames:
        frame_np = torch.clamp(frame_tensor[..., :3] * 255, 0, 255).to(dtype=torch.uint8, device="cpu").numpy()
        frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    container.mux(stream.encode())
    container.close()


def generate_video(image_path, prompt, negative_prompt, width, height, frames, steps, cfg, sampler_name, scheduler, seed, fps, output_path):
    print(f"[1/7] Loading Wan2.2 5B diffusion model...")
    model = comfy.sd.load_diffusion_model(UNET_PATH)

    print(f"[2/7] Loading UMT5-XXL text encoder...")
    clip = comfy.sd.load_clip(
        ckpt_paths=[CLIP_PATH],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.WAN,
    )

    print(f"[3/7] Loading Wan2.2 VAE...")
    vae_sd, vae_meta = comfy.utils.load_torch_file(VAE_PATH, return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_meta)

    print(f"[4/7] Encoding prompts...")
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

    print(f"[5/7] Loading source image and building I2V latent ({width}x{height}, {frames} frames)...")
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

    latent_dict = build_i2v_latent(vae, img_tensor, width, height, frames)
    latent_image = latent_dict["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    noise = comfy.sample.prepare_noise(latent_image, seed)
    noise_mask = latent_dict.get("noise_mask")

    print(f"[6/7] Sampling ({steps} steps, {sampler_name}/{scheduler}, CFG {cfg}, seed {seed})...")
    samples = comfy.sample.sample(
        model, noise, steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=negative,
        latent_image=latent_image, denoise=1.0,
        noise_mask=noise_mask, seed=seed,
    )

    print(f"[7/7] Decoding and saving video ({fps} fps)...")
    video_frames = vae.decode(samples)
    if len(video_frames.shape) == 5:
        video_frames = video_frames.reshape(-1, video_frames.shape[-3], video_frames.shape[-2], video_frames.shape[-1])

    save_webm(video_frames, output_path, fps)
    print(f"Saved: {output_path} ({video_frames.shape[0]} frames, {video_frames.shape[0]/fps:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 5B — Image to Video")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("prompt", type=str, help="Motion prompt describing desired movement")
    parser.add_argument("--negative", type=str,
                        default="static, blurry, low quality, worst quality, JPEG artifacts, ugly, deformed, extra limbs, fused fingers, still frame, cluttered background, overexposed, underexposed",
                        help="Negative prompt")
    parser.add_argument("--width", type=int, default=704, help="Video width (default: 704)")
    parser.add_argument("--height", type=int, default=1280, help="Video height (default: 1280)")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames (default: 81, ~3.4s at 24fps)")
    parser.add_argument("--steps", type=int, default=30, help="Sampling steps (default: 30)")
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale (default: 5.0)")
    parser.add_argument("--sampler", type=str, default="uni_pc", help="Sampler (default: uni_pc)")
    parser.add_argument("--scheduler", type=str, default="simple", help="Scheduler (default: simple)")
    parser.add_argument("--seed", type=int, default=None, help="Seed (default: random)")
    parser.add_argument("--fps", type=float, default=24.0, help="Output FPS (default: 24)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: output/videos/wan22_TIMESTAMP.webm)")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    if not os.path.isfile(args.image):
        print(f"[ERROR] Input image not found: {args.image}")
        sys.exit(1)

    if args.output is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "videos")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(out_dir, f"wan22_{ts}.webm")

    generate_video(args.image, args.prompt, args.negative, args.width, args.height,
                   args.frames, args.steps, args.cfg, args.sampler, args.scheduler,
                   args.seed, args.fps, args.output)


if __name__ == "__main__":
    main()
