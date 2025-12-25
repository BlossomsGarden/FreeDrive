import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
import argparse
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
from diffusers.utils import export_to_video
from PIL import Image


def decode_latents_to_video(
    latent_path: str,
    model_path: str,
    output_path: str,
    inpainting_branch: str = None,
    id_adapter_resample_learnable_path: str = None,
    dtype: torch.dtype = torch.bfloat16,
    fps: int = None,
):
    """
    Decode latents from pth file to video.
    
    Args:
        latent_path: Path to the saved latents pth file
        model_path: Path to CogVideoX base model
        output_path: Output video path
        inpainting_branch: Path to inpainting branch model (if used during generation)
        id_adapter_resample_learnable_path: Path to ID adapter weights (if used during generation)
        dtype: Data type (torch.bfloat16 or torch.float16)
        fps: Video FPS (if None, will use fps from saved latents)
    """
    print("="*100)
    print("Decoding Latents to Video")
    print("="*100)
    print(f"Latent file: {latent_path}")
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    print("="*100)
    
    # Load latents
    print("Loading latents...")
    checkpoint = torch.load(latent_path, map_location="cpu")
    latents = checkpoint['latents']
    saved_fps = checkpoint.get('fps', None)
    saved_num_frames = checkpoint.get('num_frames', None)
    
    print(f"Loaded latents shape: {latents.shape}")
    print(f"Saved FPS: {saved_fps}")
    print(f"Saved num_frames: {saved_num_frames}")
    
    # Use saved fps if not provided
    if fps is None:
        fps = saved_fps if saved_fps is not None else 8
        print(f"Using FPS from saved file: {fps}")
    
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
    
    # Configure scheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="trailing"
    )
    pipe.to("cuda" if not HAS_NPU else "npu")
    
    # Move latents to device
    device = "cuda" if not HAS_NPU else "npu"
    latents = latents.to(device=device, dtype=dtype)
    
    # Decode latents
    print("Decoding latents...")
    # The decode_latents method expects latents in shape [batch, channels, frames, height, width]
    # Pipeline returns latents in this shape when output_type="latent"
    # Ensure latents are on the correct device and dtype
    latents = latents.to(device=device, dtype=dtype)
    
    # Decode using pipeline's decode_latents method
    # This will permute to [batch, frames, channels, height, width] and decode
    decoded_frames = pipe.decode_latents(latents)
    
    print(f"Decoded frames shape: {decoded_frames.shape}")
    
    # Postprocess and save video
    print("Postprocessing and saving video...")
    video = pipe.video_processor.postprocess_video(video=decoded_frames, output_type="np")[0]
    
    export_to_video(
        video, 
        output_path, 
        fps=fps
    )
    
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode latents from pth file to video"
    )
    
    # Required arguments
    parser.add_argument(
        "--latent_path",
        type=str,
        required=True,
        help="Path to the saved latents pth file"
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
        help="Path to inpainting branch model (if used during generation)"
    )
    parser.add_argument(
        "--id_adapter_resample_learnable_path",
        type=str,
        default=None,
        help="Path to ID adapter weights (if used during generation)"
    )
    
    # Video parameters
    parser.add_argument("--fps", type=int, default=None, help="Video FPS (if None, uses fps from saved file)")
    
    # Generation parameters
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Data type")
    
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    decode_latents_to_video(
        latent_path=args.latent_path,
        model_path=args.model_path,
        output_path=args.output_path,
        inpainting_branch=args.inpainting_branch,
        id_adapter_resample_learnable_path=args.id_adapter_resample_learnable_path,
        dtype=dtype,
        fps=args.fps,
    )

