import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
import argparse
from typing import Optional, List
import numpy as np
import torch
# Try to import torch_npu for NPU 910B support
try:
    import torch_npu
    HAS_NPU = True
    print("✓ torch_npu imported successfully - using NPU 910B device")
except ImportError:
    HAS_NPU = False
    print("✓ torch_npu not available - using CUDA/CPU device")


from diffusers import (
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
    CogVideoXI2VDualInpaintAnyLPipeline,
)
import cv2
from diffusers.utils import export_to_video, load_video
from PIL import Image


def _visualize_video(pipe, mask_background, original_video, video, masks):
    """Visualize the original video, masked video, masks, and generated video"""
    original_video = pipe.video_processor.preprocess_video(original_video, height=video.shape[1], width=video.shape[2])
    masks = pipe.masked_video_processor.preprocess_video(masks, height=video.shape[1], width=video.shape[2])
    
    if mask_background:
        masked_video = original_video * (masks >= 0.5)
    else:
        masked_video = original_video * (masks < 0.5)
    
    original_video = pipe.video_processor.postprocess_video(video=original_video, output_type="np")[0]
    masked_video = pipe.video_processor.postprocess_video(video=masked_video, output_type="np")[0]
    
    masks = masks.squeeze(0).squeeze(0).numpy()
    masks = masks[..., np.newaxis].repeat(3, axis=-1)

    video_ = concatenate_images_horizontally(
        [original_video, masked_video, masks, video],
    )
    return video_


def load_video_fast(video_path: str, return_fps: bool = False):
    """
    Fast video loading using OpenCV (cv2) instead of imageio.
    This is much faster than imageio when decord is not available.
    
    Args:
        video_path: Path to video file
        return_fps: If True, also return FPS
        
    Returns:
        List[Image.Image] if return_fps=False, or (List[Image.Image], int) if return_fps=True
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get FPS if needed
    fps = None
    if return_fps:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    
    if return_fps:
        return frames, fps
    return frames


def concatenate_images_horizontally(images_list, output_type="np"):
    """Concatenate multiple image sequences horizontally"""
    concatenated_images = []
    length = len(images_list[0])
    for i in range(length):
        tmp_tuple = ()
        for item in images_list:
            tmp_tuple += (np.array(item[i]), )
        concatenated_img = np.concatenate(tmp_tuple, axis=1)
        if output_type == "pil":
            concatenated_img = Image.fromarray(concatenated_img)
        elif output_type == "np":
            pass
        else:
            raise NotImplementedError
        concatenated_images.append(concatenated_img)
    return concatenated_images


def load_mask_video(mask_path: str) -> List[Image.Image]:
    """
    Load mask video from file path.
    Supports:
    - Video file (.mp4, .avi, etc.): Each frame should be grayscale or RGB
    - Numpy array file (.npz, .npy): Should contain mask array
    - Image sequence directory: Directory containing mask images
    
    Args:
        mask_path: Path to mask file or directory
        
    Returns:
        List of PIL Images (RGB format, 255=mask region, 0=non-mask region)
    """
    if os.path.isdir(mask_path):
        # Directory of mask images
        mask_files = sorted([f for f in os.listdir(mask_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        masks = []
        for mask_file in mask_files:
            mask_img = Image.open(os.path.join(mask_path, mask_file)).convert("RGB")
            masks.append(mask_img)
        return masks
    elif mask_path.endswith(('.npz', '.npy')):
        # Numpy array file
        if mask_path.endswith('.npz'):
            data = np.load(mask_path)
            # Try common keys
            if 'arr_0' in data:
                mask_array = data['arr_0']
            elif 'masks' in data:
                mask_array = data['masks']
            else:
                # Use first array
                mask_array = data[list(data.keys())[0]]
        else:
            mask_array = np.load(mask_path)
        
        masks = []
        # Handle different array shapes: (T, H, W) or (T, H, W, C)
        if len(mask_array.shape) == 3:
            # (T, H, W) - grayscale
            for i in range(mask_array.shape[0]):
                mask_frame = mask_array[i]
                # Normalize to [0, 1] if needed
                if mask_frame.max() > 1.0:
                    mask_frame = mask_frame.astype(np.float32) / 255.0
                # Convert to binary: >0.5 means mask region
                mask_binary = (mask_frame > 0.5).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_binary).convert("RGB")
                masks.append(mask_img)
        elif len(mask_array.shape) == 4:
            # (T, H, W, C) - RGB or grayscale with channel
            for i in range(mask_array.shape[0]):
                mask_frame = mask_array[i]
                if mask_frame.shape[2] == 1:
                    mask_frame = mask_frame.squeeze(2)
                # Convert to grayscale if RGB
                if len(mask_frame.shape) == 2:
                    mask_gray = mask_frame
                else:
                    mask_gray = np.mean(mask_frame, axis=2)
                # Normalize to [0, 1] if needed
                if mask_gray.max() > 1.0:
                    mask_gray = mask_gray.astype(np.float32) / 255.0
                # Convert to binary: >0.5 means mask region
                mask_binary = (mask_gray > 0.5).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_binary).convert("RGB")
                masks.append(mask_img)
        else:
            raise ValueError(f"Unsupported mask array shape: {mask_array.shape}. Expected (T, H, W) or (T, H, W, C)")
        return masks
    else:
        # Video file - use fast OpenCV loading
        mask_video = load_video_fast(mask_path)
        masks = []
        for frame in mask_video:
            # Convert to grayscale if RGB
            if isinstance(frame, Image.Image):
                frame_array = np.array(frame)
                if len(frame_array.shape) == 3:
                    # RGB, convert to grayscale
                    frame_array = np.mean(frame_array, axis=2)
                # Binary mask: >128 means mask region
                mask_binary = (frame_array > 128).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_binary).convert("RGB")
                masks.append(mask_img)
        return masks


def prepare_video_and_masks(
    rgb_video_path: str,
    mask_video_path: str,
    mask_background: bool = False,
    fps: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None
) -> tuple:
    """
    Load RGB video and mask video, prepare them for v2v inpainting.
    
    Args:
        rgb_video_path: Path to RGB video file
        mask_video_path: Path to mask video/file/directory
        mask_background: If True, mask background (0=mask), else mask foreground (255=mask)
        fps: Video FPS (if None, will be read from video)
        start_frame: Start frame index (default: 0)
        end_frame: End frame index (if None, use all frames)
        
    Returns:
        tuple: (video, masked_video, binary_masks, fps)
            - video: List[PIL.Image] - original video frames
            - masked_video: List[PIL.Image] - masked video frames (black in mask regions)
            - binary_masks: List[PIL.Image] - binary masks (RGB, 255=mask, 0=non-mask)
            - fps: int - video frame rate
    """
    # Load RGB video - use fast OpenCV loading (also get FPS to avoid reopening)
    if fps is None or fps == 0:
        video, fps = load_video_fast(rgb_video_path, return_fps=True)
    else:
        video = load_video_fast(rgb_video_path, return_fps=False)
    
    video = [frame.convert("RGB") for frame in video]
    
    # Load mask video
    mask_images = load_mask_video(mask_video_path)
    
    # Check frame count match
    if len(mask_images) != len(video):
        print(f"Warning: Video has {len(video)} frames, mask has {len(mask_images)} frames")
        min_frames = min(len(video), len(mask_images))
        video = video[:min_frames]
        mask_images = mask_images[:min_frames]
    
    # Apply frame range
    video = video[start_frame:end_frame]
    mask_images = mask_images[start_frame:end_frame]
    
    # Process masks and create masked video (optimized for speed)
    masked_video = []
    binary_masks = []
    
    for frame, mask_img in zip(video, mask_images):
        # Convert to numpy arrays once
        frame_array = np.array(frame, dtype=np.uint8)
        mask_array = np.array(mask_img, dtype=np.uint8)
        
        # Convert RGB mask to grayscale binary mask (optimized)
        if len(mask_array.shape) == 3:
            # Use weighted average for better grayscale conversion (faster than mean)
            mask_gray = (mask_array[:, :, 0] * 0.299 + 
                        mask_array[:, :, 1] * 0.587 + 
                        mask_array[:, :, 2] * 0.114).astype(np.uint8)
        else:
            mask_gray = mask_array
        
        # Binary mask: >128 means mask region
        binary_mask = mask_gray > 128
        
        # Create masked frame (black in mask regions) - vectorized operation
        masked_frame = frame_array.copy()
        masked_frame[binary_mask] = 0
        masked_video.append(Image.fromarray(masked_frame))
        
        # Create binary mask image
        if mask_background:
            # Mask background: 0=mask region, 255=non-mask region
            binary_mask_image = np.where(binary_mask, 0, 255).astype(np.uint8)
        else:
            # Mask foreground: 255=mask region, 0=non-mask region
            binary_mask_image = np.where(binary_mask, 255, 0).astype(np.uint8)
        # Expand to RGB
        binary_mask_rgb = np.stack([binary_mask_image] * 3, axis=2)
        binary_masks.append(Image.fromarray(binary_mask_rgb))
    
    return video, masked_video, binary_masks, fps


def generate_video_v2v(
    rgb_video_path: str,
    mask_video_path: str,
    prompt: str,
    model_path: str,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    # Model paths
    inpainting_branch: Optional[str] = None,
    id_adapter_resample_learnable_path: Optional[str] = None,
    # Video parameters
    fps: Optional[int] = None,
    num_frames: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    down_sample_fps: int = 8,
    overlap_frames: int = 0,
    # Mask parameters
    mask_background: bool = False,
    dilate_size: int = 0,
    # Inpainting parameters
    first_frame_gt: bool = False,
    replace_gt: bool = False,
    mask_add: bool = False,
    prev_clip_weight: float = 0.0,
    long_video: bool = False,
):
    """
    Generate inpainted video using V2V (Video-to-Video) inpainting.
    
    Args:
        rgb_video_path: Path to RGB video file (.mp4, .avi, etc.)
        mask_video_path: Path to mask video/file/directory
        prompt: Text description of the video content
        model_path: Path to CogVideoX base model
        output_path: Output video path
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        num_videos_per_prompt: Number of videos to generate
        dtype: Data type (torch.bfloat16 or torch.float16)
        seed: Random seed
        inpainting_branch: Path to inpainting branch model (recommended)
        id_adapter_resample_learnable_path: Path to ID adapter weights
        fps: Video FPS (auto-detected if None)
        num_frames: Number of frames to generate (uses all frames if None)
        start_frame: Start frame index
        end_frame: End frame index (None = use all)
        down_sample_fps: Downsample FPS for processing
        overlap_frames: Overlap frames for long video
        mask_background: If True, mask background instead of foreground
        dilate_size: Mask dilation size (0 = no dilation)
        first_frame_gt: Use first frame as ground truth
        replace_gt: Replace ground truth frames
        mask_add: Add mask to input
        prev_clip_weight: Weight for previous clip conditioning
        long_video: Enable long video mode
    """
    
    print("="*100)
    print("V2V Video Inpainting")
    print("="*100)
    print(f"RGB Video: {rgb_video_path}")
    print(f"Mask: {mask_video_path}")
    print(f"Prompt: {prompt}")
    print("="*100)
    
    # Load video and masks
    video, masked_video, binary_masks, detected_fps = prepare_video_and_masks(
        rgb_video_path=rgb_video_path,
        mask_video_path=mask_video_path,
        mask_background=mask_background,
        fps=fps,
        start_frame=start_frame,
        end_frame=end_frame
    )
    
    fps = detected_fps if fps is None else fps
    print(f"Loaded {len(video)} frames at {fps} FPS")
    
    # Load pipeline
    print("Loading pipeline...")
    if inpainting_branch:
        print(f"Using inpainting branch: {inpainting_branch}")
        branch = CogvideoXBranchModel.from_pretrained(
            inpainting_branch, 
            torch_dtype=dtype
        ).to(dtype=dtype).to("cuda" if not HAS_NPU else "npu")
        
        if id_adapter_resample_learnable_path is None:
            pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
                model_path,
                branch=branch,
                torch_dtype=dtype,
            )
        else:
            print(f"Loading ID adapter from: {id_adapter_resample_learnable_path}")
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=dtype,
                id_pool_resample_learnable=True,
            ).to(dtype=dtype).to("cuda" if not HAS_NPU else "npu")
            
            pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
                model_path,
                branch=branch,
                transformer=transformer,
                torch_dtype=dtype,
            )
            
            pipe.load_lora_weights(
                id_adapter_resample_learnable_path,
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="test_1",
                target_modules=["transformer"]
            )
    else:
        print("No inpainting branch provided, using default branch...")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
        ).to(dtype=dtype).to("cuda" if not HAS_NPU else "npu")
        
        branch = CogvideoXBranchModel.from_transformer(
            transformer=transformer,
            num_layers=1,
            attention_head_dim=transformer.config.attention_head_dim,
            num_attention_heads=transformer.config.num_attention_heads,
            load_weights_from_transformer=True
        ).to(dtype=dtype).to("cuda" if not HAS_NPU else "npu")
        
        pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
            model_path,
            branch=branch,
            transformer=transformer,
            torch_dtype=dtype,
        )
    
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)
    
    # Apply mask dilation if needed
    if dilate_size > 0:
        print(f"Applying mask dilation with size {dilate_size}")
        for i in range(len(binary_masks)):
            mask = cv2.dilate(np.array(binary_masks[i]), np.ones((dilate_size, dilate_size), np.uint8))
            mask = mask.astype(np.uint8)
            binary_masks[i] = Image.fromarray(mask)
    
    # Configure scheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="trailing"
    )
    pipe.to("cuda" if not HAS_NPU else "npu")

    # 全注释 62139 MB
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    
    if long_video:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    
    # Prepare frames
    frames = num_frames if num_frames else len(video)
    down_sample_fps = fps if down_sample_fps == 0 else down_sample_fps
    
    # Downsample if needed
    if fps != down_sample_fps:
        step = int(fps // down_sample_fps)
        video = video[::step]
        masked_video = masked_video[::step]
        binary_masks = binary_masks[::step]
    
    # Limit frames
    if not long_video:
        video = video[:frames]
        masked_video = masked_video[:frames]
        binary_masks = binary_masks[:frames]
    
    if len(video) < frames:
        print(f"Warning: Video has {len(video)} frames, requested {frames}")
        frames = len(video)
    
    # Handle first frame
    if first_frame_gt:
        gt_mask_first_frame = binary_masks[0]
        gt_video_first_frame = video[0]
        if mask_background:
            binary_masks[0] = Image.fromarray(
                np.ones_like(np.array(binary_masks[0])) * 255
            ).convert("RGB")
        else:
            binary_masks[0] = Image.fromarray(
                np.zeros_like(np.array(binary_masks[0]))
            ).convert("RGB")
    
    # Generate video
    print("Generating video...")
    image = masked_video[0]
    
    # When using id_pool_resample_learnable, mask_add must be True to provide masks to transformer
    id_pool_resample_learnable = id_adapter_resample_learnable_path is not None
    if id_pool_resample_learnable and not mask_add:
        print("Warning: id_pool_resample_learnable requires mask_add=True. Setting mask_add=True.")
        mask_add = True
    
    # return dict
    inpaint_outputs = pipe(
        prompt=prompt,
        image=image,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        video=masked_video,
        masks=binary_masks,
        strength=1.0,
        replace_gt=replace_gt,
        mask_add=mask_add,
        stride=int(frames - overlap_frames),
        prev_clip_weight=prev_clip_weight,
        id_pool_resample_learnable=id_pool_resample_learnable,
        output_type="latent"
    )
    
    # Get latents from output
    latents = inpaint_outputs.frames
    print("Latents generated!")
    print("inpait_outputs.frames shape:", inpaint_outputs.frames.shape)
    
    # Save latents to pth file
    latent_output_path = output_path.replace(".mp4", "_latents.pth")
    torch.save({
        'latents': latents.cpu(),  # Move to CPU before saving
        'fps': down_sample_fps if down_sample_fps > 0 else fps,
        'num_frames': frames,
        'shape': latents.shape,
        'dtype': str(latents.dtype),
    }, latent_output_path)
    
    print(f"Latents saved to: {latent_output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V2V (Video-to-Video) inpainting with custom RGB video and mask inputs"
    )
    
    # Required arguments
    parser.add_argument(
        "--rgb_video_path",
        type=str,
        required=True,
        help="Path to RGB video file (.mp4, .avi, etc.)"
    )
    parser.add_argument(
        "--mask_video_path",
        type=str,
        required=True,
        help="Path to mask video/file/directory (.mp4, .npz, .npy, or directory of images)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of the video content"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to CogVideoX base model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output video path"
    )
    
    # Optional model paths
    parser.add_argument(
        "--inpainting_branch",
        type=str,
        default=None,
        help="Path to inpainting branch model (recommended)"
    )
    parser.add_argument(
        "--id_adapter_resample_learnable_path",
        type=str,
        default=None,
        help="Path to ID adapter weights"
    )
    
    # Video parameters
    parser.add_argument("--fps", type=int, default=None, help="Video FPS (auto-detected if None)")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to generate")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index")
    parser.add_argument("--down_sample_fps", type=int, default=8, help="Downsample FPS")
    parser.add_argument("--overlap_frames", type=int, default=0, help="Overlap frames")
    
    # Mask parameters
    parser.add_argument(
        "--mask_background",
        action='store_true',
        help="Mask background instead of foreground"
    )
    parser.add_argument("--dilate_size", type=int, default=0, help="Mask dilation size")
    
    # Inpainting parameters
    parser.add_argument(
        "--first_frame_gt",
        action='store_true',
        help="Use first frame as ground truth"
    )
    parser.add_argument(
        "--replace_gt",
        action='store_true',
        help="Replace ground truth frames"
    )
    parser.add_argument(
        "--mask_add",
        action='store_true',
        help="Add mask to input"
    )
    parser.add_argument("--prev_clip_weight", type=float, default=0.0, help="Previous clip weight")
    
    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Data type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Video mode
    parser.add_argument(
        "--long_video",
        action='store_true',
        help="Enable long video mode"
    )
    
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    generate_video_v2v(
        rgb_video_path=args.rgb_video_path,
        mask_video_path=args.mask_video_path,
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        inpainting_branch=args.inpainting_branch,
        id_adapter_resample_learnable_path=args.id_adapter_resample_learnable_path,
        fps=args.fps,
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        down_sample_fps=args.down_sample_fps,
        overlap_frames=args.overlap_frames,
        mask_background=args.mask_background,
        dilate_size=args.dilate_size,
        first_frame_gt=args.first_frame_gt,
        replace_gt=args.replace_gt,
        mask_add=args.mask_add,
        prev_clip_weight=args.prev_clip_weight,
        long_video=args.long_video,
    )
