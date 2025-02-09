# inference.py
import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
import cv2
import numpy as np
import os
from superres import apply_superres

def post_process_video(orig_video_path, mask_video_path, output_video_path, superres_method):
    """
    Post-process each frame in the generated video:
      - Use the corresponding mask frame to find the generated subframe region.
      - If the subframe region is smaller than the full frame, compute an upscale factor.
      - Apply superresolution (using GFPGAN and/or CodeFormer) only on that region.
      - Merge the enhanced region back into the frame.
    The final enhanced video is saved to output_video_path.
    """
    cap_orig = cv2.VideoCapture(orig_video_path)
    cap_mask = cv2.VideoCapture(mask_video_path)
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    width  = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Post-processing {frame_count} frames with superresolution...")

    frame_idx = 0
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_mask, frame_mask = cap_mask.read()
        if not ret_orig or not ret_mask:
            break

        # Convert mask to grayscale and threshold to create a binary mask
        gray_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

        # Find contours to locate the generated region
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Use the largest contour
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            # If the detected region is smaller than the full frame, process it
            if w < width or h < height:
                subframe = frame_orig[y:y+h, x:x+w]
                # Compute upscale factor as the max ratio between full frame and subframe dimensions
                factor_w = width / w
                factor_h = height / h
                upscale_factor = max(factor_w, factor_h)
                print(f"Frame {frame_idx}: Upscaling subframe (w={w}, h={h}) by factor {upscale_factor:.2f} using {superres_method}...")
                enhanced_subframe = apply_superres(subframe, upscale_factor, superres_method)
                # Resize the enhanced subframe back to (w, h) if needed
                enhanced_subframe = cv2.resize(enhanced_subframe, (w, h))
                frame_orig[y:y+h, x:x+w] = enhanced_subframe

        out.write(frame_orig)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames...")

    cap_orig.release()
    cap_mask.release()
    out.release()
    print(f"Superresolution post-processing completed. Output saved to {output_video_path}")

def main(config, args):
    # Determine data type based on GPU capabilities
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,  # load checkpoint
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
    )

    if args.superres.lower() != "none":
        mask_video_path = args.video_out_path.replace(".mp4", "_mask.mp4")
        # Define a temporary output path for the superresolution process.
        temp_output_path = args.video_out_path.replace(".mp4", "_tmp.mp4")
        # Post-process the video: read from the original video and write to a temporary file.
        post_process_video(args.video_out_path, mask_video_path, temp_output_path, args.superres)
        # Replace the original output video with the enhanced version.
        os.replace(temp_output_path, args.video_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/second_stage.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--superres", type=str, default="none", help="Superresolution method: GFPGAN, CodeFormer, or GFPGAN,CodeFormer")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)
    main(config, args)
