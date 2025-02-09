#!/bin/bash

sudo swapon -a

python -m scripts.inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 15 \
    --guidance_scale 1.5 \
    --video_path "assets/demo4.mp4" \
    --audio_path "assets/demo4_audio.wav" \
    --video_out_path "video_o3.mp4" \
    --superres GFPGAN,CodeFormer
