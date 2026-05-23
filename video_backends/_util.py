"""Shared helpers used by every video backend."""
import os
from fractions import Fraction

import av
import numpy as np
import torch
from PIL import Image


def load_image_as_tensor(image_path):
    """Load an RGB image as a (1, H, W, 3) float32 tensor in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def save_webm(frames, output_path, fps, crf="16"):
    """Save an (N, H, W, 3) float [0, 1] tensor as a VP9 WebM."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    container = av.open(output_path, mode="w")
    try:
        stream = container.add_stream("libvpx-vp9", rate=Fraction(round(fps * 1000), 1000))
        stream.width = frames.shape[-2]
        stream.height = frames.shape[-3]
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 0
        stream.options = {"crf": str(crf)}

        for frame_tensor in frames:
            frame_np = torch.clamp(frame_tensor[..., :3] * 255, 0, 255).to(
                dtype=torch.uint8, device="cpu"
            ).numpy()
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        container.mux(stream.encode())
    finally:
        container.close()
