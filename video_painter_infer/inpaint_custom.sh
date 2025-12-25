#!/bin/bash
# V2V (Video-to-Video) inpainting script
# Supports direct input of RGB video and mask files


# ========== Model Paths ==========
model_path="/data/wlh/FreeDrive/model/CogVideoX-5b-I2V"
inpainting_branch="/data/wlh/FreeDrive/model/VideoPainter/VideoPainter/checkpoints/branch"
id_adapter_resample_learnable_path="/data/wlh/FreeDrive/model/VideoPainter/VideoPainterID/checkpoints"
rgb_video_path="/data/wlh/FreeDrive/code/video_painter/video_painter_infer/example/occluded_images.mp4"
# Mask path (REQUIRED) - can be:
#   - Video file: .mp4, .avi, etc.
#   - Numpy array: .npz, .npy
#   - Directory: folder containing mask images (.png, .jpg)
mask_video_path="/data/wlh/FreeDrive/code/video_painter/video_painter_infer/example/occluded_masks.mp4"
prompt="This scene is set on an urban road, but some elements such as vehicles, roadside trees, the road itself, and buildings are missing or contain artifacts, requiring inpainting restoration."
output_path="output.mp4"

# ========== Video Parameters ==========
fps=10                    # Video FPS (set to 0 or leave empty for auto-detection)
num_frames=49            # Number of frames to generate (None = use all frames)
start_frame=0            # Start frame index
end_frame=               # End frame index (empty = use all frames)
down_sample_fps=0        # Downsample FPS for processing, set to 0 to disable downsampling
overlap_frames=0         # Overlap frames for long video

# ========== Mask Parameters ==========
mask_background=0        # Set to 1 to mask background instead of foreground
dilate_size=2            # Mask dilation size (0 = no dilation)

# ========== Generation Parameters ==========
num_inference_steps=50
guidance_scale=6.0
num_videos_per_prompt=1
dtype="bfloat16"         # "float16" or "bfloat16"
seed=24

# ========== Inpainting Parameters ==========
first_frame_gt=0         # Set to 1 to use first frame as ground truth
replace_gt=0             # Set to 1 to replace ground truth frames
mask_add=0               # Set to 1 to add mask to input
prev_clip_weight=0.0      # Weight for previous clip conditioning

# ========== Video Mode ==========
long_video=0             # Set to 1 to enable long video mode

# ========== Build Command ==========
cmd="CUDA_VISIBLE_DEVICES=3 python inpaint_custom.py \
    --rgb_video_path \"$rgb_video_path\" \
    --mask_video_path \"$mask_video_path\" \
    --prompt \"$prompt\" \
    --model_path \"$model_path\" \
    --output_path \"$output_path\" \
    --num_inference_steps $num_inference_steps \
    --guidance_scale $guidance_scale \
    --num_videos_per_prompt $num_videos_per_prompt \
    --dtype $dtype \
    --seed $seed \
    --down_sample_fps $down_sample_fps \
    --overlap_frames $overlap_frames \
    --prev_clip_weight $prev_clip_weight \
    --dilate_size $dilate_size"

# Add optional parameters
if [ -n "$inpainting_branch" ]; then
    cmd="$cmd --inpainting_branch \"$inpainting_branch\""
fi

if [ -n "$id_adapter_resample_learnable_path" ]; then
    cmd="$cmd --id_adapter_resample_learnable_path \"$id_adapter_resample_learnable_path\""
fi

if [ -n "$fps" ] && [ "$fps" != "0" ]; then
    cmd="$cmd --fps $fps"
fi

if [ -n "$num_frames" ]; then
    cmd="$cmd --num_frames $num_frames"
fi

if [ "$start_frame" != "0" ]; then
    cmd="$cmd --start_frame $start_frame"
fi

if [ -n "$end_frame" ]; then
    cmd="$cmd --end_frame $end_frame"
fi

if [ "$mask_background" = "1" ]; then
    cmd="$cmd --mask_background"
fi

if [ "$first_frame_gt" = "1" ]; then
    cmd="$cmd --first_frame_gt"
fi

if [ "$replace_gt" = "1" ]; then
    cmd="$cmd --replace_gt"
fi

if [ "$mask_add" = "1" ]; then
    cmd="$cmd --mask_add"
fi

if [ "$long_video" = "1" ]; then
    cmd="$cmd --long_video"
fi

# Execute command
echo "Executing: $cmd"
eval $cmd
