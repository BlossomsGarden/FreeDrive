#!/bin/bash
# V2V (Video-to-Video) inpainting script
# Supports direct input of RGB video and mask files
# Thanks for your acknowledgement of our work!

# We used 8 NVIDIA H800 GPUs to train the VideoPainter in VPData for around 80000 steps.
# The training time is around a week.

# EZ9999: [PID: 51950] 2025-12-22-20:19:17.022.230 input datatype must be uint8 float16,float or double![FUNC:Upsample3dCheckDtypeSupportUint8][FILE:image_ops.cc][LINE:4125]
#         TraceBack (most recent call last):
#        Verifying UpsampleNearest3d71 failed.[FUNC:InferShapeAndType][FILE:infershape_pass.cc][LINE:130]
#        Sessin_id 0 does not exist, graph_id 70[FUNC:GetJsonObject][FILE:analyzer.cc][LINE:155]
#        Param:graph_info is nullptr, check invalid[FUNC:DoAnalyze][FILE:analyzer.cc][LINE:253]
#        Param:graph_info is nullptr, check invalid[FUNC:SaveAnalyzerDataToFile][FILE:analyzer.cc][LINE:210]
#        Call InferShapeAndType for node:UpsampleNearest3d71(UpsampleNearest3d) failed[FUNC:Infer][FILE:infershape_pass.cc][LINE:118]
#        process pass InferShapePass on node:UpsampleNearest3d71 failed, ret:4294967295[FUNC:RunPassesOnNode][FILE:base_pass.cc][LINE:563]
#        build graph failed, graph id:70, ret:1343242270[FUNC:BuildModelWithGraphId][FILE:ge_generator.cc][LINE:1618]
#        [Build][SingleOpModel]call ge interface generator.BuildSingleOpModel failed. ge result = 1343242270[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
#        [Build][Op]Fail to build op model[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
#        build op model failed, result = 500002[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
#  (function ExecFunc)
# Traceback (most recent call last):
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/inpaint_custom.py", line 609, in <module>
#     generate_video_v2v(
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/inpaint_custom.py", line 461, in generate_video_v2v
#     inpaint_outputs = pipe(
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
#     return func(*args, **kwargs)
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_inpainting_i2v_branch_anyl.py", line 1072, in __call__
#     video = self.decode_latents(frame_accumulator)
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_inpainting_i2v_branch_anyl.py", line 483, in decode_latents
#     frames = self.vae.decode(latents).sample
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/diffusers/src/diffusers/utils/accelerate_utils.py", line 46, in wrapper
#     return method(self, *args, **kwargs)
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py", line 1186, in decode
#     decoded = self._decode(z).sample
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py", line 1154, in _decode
#     z_intermediate = self.decoder(z_intermediate)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py", line 877, in forward
#     hidden_states = up_block(hidden_states, temb, sample)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py", line 606, in forward
#     hidden_states = upsampler(hidden_states)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/diffusers/src/diffusers/models/upsampling.py", line 409, in forward
#     inputs = self.conv(inputs)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 458, in forward
#     return self._conv_forward(input, self.weight, self.bias)
#   File "/home/ma-user/anaconda3/envs/videopainter/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 454, in _conv_forward
#     return F.conv2d(input, weight, bias, self.stride,
# RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is UpsampleNearest3d.
# Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, pleace set the environment variable ASCEND_LAUNCH_BLOCKING=1.
# [ERROR] 2025-12-22-20:19:17 (PID:51950, Device:0, RankID:-1) ERR00100 PTA call acl api failed

# """
#  CUDA_VISIBLE_DEVICES=3 python decode_latents.py \
#     --latent_path "output_latents.pth" \
#     --model_path "/data/wlh/FreeDrive/model/CogVideoX-5b-I2V" \
#     --output_path "output.mp4" \
#     --inpainting_branch "/data/wlh/FreeDrive/model/VideoPainter/VideoPainter/checkpoints/branch" \
#     --id_adapter_resample_learnable_path "/data/wlh/FreeDrive/model/VideoPainter/VideoPainterID/checkpoints" \
#     --dtype float16 \
#     --fps 10
# """

# ========== Model Paths ==========
model_path="/home/ma-user/modelarts/user-job-dir/wlh/model/PointCloudPainter/CogVideoX-5b-I2V"
inpainting_branch="/home/ma-user/modelarts/user-job-dir/wlh/model/PointCloudPainter/VideoPainter/VideoPainter/checkpoints/branch"
id_adapter_resample_learnable_path="/home/ma-user/modelarts/user-job-dir/wlh/model/PointCloudPainter/VideoPainter/VideoPainterID/checkpoints"
rgb_video_path="/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/example/occluded_images.mp4"
# Mask path (REQUIRED) - can be:
#   - Video file: .mp4, .avi, etc.
#   - Numpy array: .npz, .npy
#   - Directory: folder containing mask images (.png, .jpg)
mask_video_path="/home/ma-user/modelarts/user-job-dir/wlh/code/PointCloudPainter/example/occluded_masks.mp4"
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
dtype="bfloat16"
seed=24

# ========== Inpainting Parameters ==========
first_frame_gt=0         # Set to 1 to use first frame as ground truth
replace_gt=0             # Set to 1 to replace ground truth frames
mask_add=0               # Set to 1 to add mask to input
prev_clip_weight=0.0      # Weight for previous clip conditioning

# ========== Video Mode ==========
long_video=0             # Set to 1 to enable long video mode

# ========== Build Command ==========
cmd="ASCEND_RT_VISIBLE_DEVICES=7  python inpaint_custom.py \
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
