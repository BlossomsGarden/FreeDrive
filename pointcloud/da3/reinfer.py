"""
使用偏移后的相机内外参，对 render_from_npz.py 生成的RGB图像进行重新投影推理。

流程：
1. 加载前三个摄像机（cam_ids = [1, 0, 2]）的原始内外参
2. 对这些摄像机施加向右偏移1米的偏移，得到offset后的内外参
3. 读取 render_from_npz.py 生成的三个摄像机的rgb_images图片结果
4. 使用offset的内外参进行 da3_infer.py 的有参估计
5. 保存结果为npz文件
"""

import os
from glob import glob
from typing import Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Try to import torch_npu for NPU 910B support
try:
    import torch_npu
    HAS_NPU = True
    print("✓ torch_npu imported successfully - using NPU 910B device")
except ImportError:
    HAS_NPU = False
    print("✓ torch_npu not available - using CUDA/CPU device")

from depth_anything_3.api import DepthAnything3


# ------------------------------
# Configuration
# ------------------------------
if HAS_NPU:
    data_root = "/home/ma-user/modelarts/user-job-dir/wlh/code/FreeDrive/vda/toy_data/10050"
else:
    data_root = "/data/wlh/FreeDrive/data/waymo/processed/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels"

# Camera IDs to use (first three cameras)
cam_ids = [1, 0, 2]

# Render output directory (from render_from_npz.py)
render_output_dir = "rendered_views"

# Waymo raw camera resolution
orig_w, orig_h = 1920, 1280

# Camera offset: Right 1 meter in Waymo coordinates
# In Waymo: x=forward(+), y=left(+), z=up(+)
# Right = negative y direction, so [0.0, -1.0, 0.0]
camera_offset = [0.0, -1.0, 0.0]  # Right 1 meter

# Inference resolution configuration
process_resolution = 1008

# Chunk-based inference: number of frames to process per inference chunk
chunk_frame = 18


def waymo_to_cv_coordinate_transform():
    """Convert Waymo vehicle coordinate system to CV coordinate system.
    
    Waymo vehicle: x=forward, y=left, z=up
    CV: x=right, y=down, z=forward
    
    Returns 4x4 transformation matrix from Waymo vehicle frame to CV frame.
    """
    R = np.array([
        [0, -1, 0, 0],  # CV x = -Waymo y
        [0, 0, -1, 0],  # CV y = -Waymo z
        [1, 0, 0, 0],   # CV z = Waymo x
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return R


def load_intrinsics_original(intrinsics_dir: str, cam_ids, orig_size: Tuple[int, int], target_size: Tuple[int, int]):
    """Load K matrices (3x3) from Waymo intrinsics txt files and scale to target size."""
    ow, oh = orig_size
    tw, th = target_size
    sx, sy = tw / ow, th / oh
    intrinsics = []
    for cam_id in cam_ids:
        vals = np.loadtxt(os.path.join(intrinsics_dir, f"{cam_id}.txt"))
        fx, fy, cx, cy = vals[:4]
        fx_s, fy_s, cx_s, cy_s = fx * sx, fy * sy, cx * sx, cy * sy
        k_mat = np.array(
            [[fx_s, 0.0, cx_s], [0.0, fy_s, cy_s], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        intrinsics.append(k_mat)
    return intrinsics


def load_camera_extrinsics_original(extrinsics_dir: str, cam_ids):
    """Load camera->ego extrinsics (4x4) for each chosen camera id."""
    return {
        cam_id: np.loadtxt(os.path.join(extrinsics_dir, f"{cam_id}.txt")).reshape(4, 4)
        for cam_id in cam_ids
    }


def load_ego_poses(ego_pose_dir: str):
    """Load all ego->world poses (4x4)."""
    pose_files = sorted(
        glob(os.path.join(ego_pose_dir, "*.txt")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    return [np.loadtxt(pf).reshape(4, 4) for pf in pose_files], pose_files


def apply_camera_offset(cam_to_ego: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Apply camera offset to cam_to_ego extrinsic matrix.
    
    Args:
        cam_to_ego: (4, 4) camera-to-ego transform (Waymo format)
        offset: (3,) offset in Waymo vehicle coordinates [x, y, z]
               x=forward(+), y=left(+), z=up(+)
    
    Returns:
        Modified cam_to_ego with offset applied
    """
    cam_to_ego_offseted = cam_to_ego.copy()
    cam_to_ego_offseted[0, 3] += offset[0]  # forward/backward
    cam_to_ego_offseted[1, 3] += offset[1]  # left/right
    cam_to_ego_offseted[2, 3] += offset[2]  # up/down
    return cam_to_ego_offseted


def compute_world_to_camera(cam_to_ego: np.ndarray, ego_to_world: np.ndarray):
    """Compute world->camera matrix given camera->ego and ego->world."""
    # Coordinate system conversion: Waymo vehicle -> CV
    waymo_to_cv = waymo_to_cv_coordinate_transform()
    cv_to_waymo = np.linalg.inv(waymo_to_cv)
    
    # Convert camera->ego to CV coordinate system
    cam_to_ego_cv = waymo_to_cv @ cam_to_ego @ cv_to_waymo
    
    # Convert ego->world to CV coordinate system
    ego_to_world_cv = waymo_to_cv @ ego_to_world @ cv_to_waymo
    
    # Compute world->ego in CV frame
    world_to_ego_cv = np.linalg.inv(ego_to_world_cv)
    
    # Compute ego->camera in CV frame
    ego_to_cam_cv = np.linalg.inv(cam_to_ego_cv)
    
    # Combine transforms
    world_to_cam = ego_to_cam_cv @ world_to_ego_cv
    
    return world_to_cam


def load_rendered_images(render_output_dir: str, cam_ids: list):
    """Load rendered RGB images from render_from_npz.py output.
    
    Args:
        render_output_dir: Base directory containing cam_X subdirectories
        cam_ids: List of camera IDs
    
    Returns:
        List of image file paths in frame-major then cam_ids order
        Number of frames detected
    """
    images = []
    num_frames = None
    
    # Check first camera to determine number of frames
    first_cam_dir = os.path.join(render_output_dir, f"cam_{cam_ids[0]}", "rgb_images")
    if not os.path.exists(first_cam_dir):
        raise FileNotFoundError(f"Render output directory not found: {first_cam_dir}")
    
    # Find all frame files in first camera directory
    frame_files = sorted(
        glob(os.path.join(first_cam_dir, "frame_*.jpg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1])
    )
    
    if len(frame_files) == 0:
        raise FileNotFoundError(f"No rendered images found in {first_cam_dir}")
    
    num_frames = len(frame_files)
    print(f"Found {num_frames} frames in rendered images")
    
    # Load images in frame-major then cam_ids order
    for frame_idx in range(num_frames):
        for cam_id in cam_ids:
            cam_dir = os.path.join(render_output_dir, f"cam_{cam_id}", "rgb_images")
            image_path = os.path.join(cam_dir, f"frame_{frame_idx:05d}.jpg")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing rendered image: {image_path}")
            
            images.append(image_path)
    
    return images, num_frames


def main():
    print("="*60)
    print("Reinfer with Offset Camera Parameters")
    print("="*60)
    
    # Load original camera parameters
    print("\n1. Loading original camera parameters...")
    intrinsics_dir = os.path.join(data_root, "intrinsics")
    extrinsics_dir = os.path.join(data_root, "extrinsics")
    ego_pose_dir = os.path.join(data_root, "ego_pose")
    
    # Load original intrinsics and extrinsics
    cam_to_ego_original = load_camera_extrinsics_original(extrinsics_dir, cam_ids)
    ego_to_world_list, pose_files = load_ego_poses(ego_pose_dir)
    
    print(f"  Loaded extrinsics for cameras: {cam_ids}")
    print(f"  Loaded {len(ego_to_world_list)} ego poses")
    
    # Detect target image size from rendered images
    print("\n2. Loading rendered images...")
    rendered_images, num_frames = load_rendered_images(render_output_dir, cam_ids)
    print(f"  Loaded {len(rendered_images)} rendered images")
    
    # Load a sample image to get target resolution
    sample_image_path = rendered_images[0]
    with Image.open(sample_image_path) as im:
        target_w, target_h = im.size
    print(f"  Target image resolution: {target_w}x{target_h}")
    
    # Limit ego poses to match number of frames
    if len(ego_to_world_list) < num_frames:
        print(f"  Warning: Only {len(ego_to_world_list)} ego poses available, using all")
        num_frames = len(ego_to_world_list)
    else:
        ego_to_world_list = ego_to_world_list[:num_frames]
    
    # Apply camera offset to get offset extrinsics
    print(f"\n3. Applying camera offset: {camera_offset} (Waymo coordinates, Right 1m)...")
    camera_offset_arr = np.array(camera_offset, dtype=np.float32)
    cam_to_ego_offseted = {}
    for cam_id in cam_ids:
        cam_to_ego_offseted[cam_id] = apply_camera_offset(
            cam_to_ego_original[cam_id], camera_offset_arr
        )
        print(f"  Applied offset to camera {cam_id}")
    
    # Compute world-to-camera transforms with offset extrinsics
    print("\n4. Computing world-to-camera transforms with offset...")
    intrinsics_per_cam = load_intrinsics_original(
        intrinsics_dir, cam_ids, (orig_w, orig_h), (target_w, target_h)
    )
    
    extrinsics_list = []
    intrinsics_list = []
    
    for ego_to_world in ego_to_world_list:
        for cam_id in cam_ids:
            # Use offseted cam_to_ego
            w2c = compute_world_to_camera(cam_to_ego_offseted[cam_id], ego_to_world)
            extrinsics_list.append(w2c.astype(np.float32))
            intrinsics_list.append(intrinsics_per_cam[cam_ids.index(cam_id)])
    
    extrinsics_array = np.stack(extrinsics_list, axis=0)  # (num_frames*len(cam_ids), 4, 4)
    intrinsics_array = np.stack(intrinsics_list, axis=0)  # (num_frames*len(cam_ids), 3, 3)
    print(f"  Computed {len(extrinsics_list)} camera poses and intrinsics")
    
    # Run inference
    print("\n5. Running Depth-Anything-3 inference...")
    
    # Auto-detect device
    device = torch.device("cpu")
    if HAS_NPU:
        model = DepthAnything3.from_pretrained("/home/ma-user/modelarts/user-job-dir/wlh/model/da3nested-giant-large")
        device = torch.device("npu")
        print(f"  Using device: NPU 910B")
    elif torch.cuda.is_available():
        model = DepthAnything3.from_pretrained("/data/wlh/DA3/model/da3nested-giant-large")
        device = torch.device("cuda")
        print(f"  Using device: CUDA")
    else:
        raise RuntimeError("No GPU/NPU available")
    
    model = model.to(device=device)
    
    num_cams = len(cam_ids)
    num_chunks = (num_frames + chunk_frame - 1) // chunk_frame
    
    print(f"  Total frames: {num_frames}")
    print(f"  Frames per chunk: {chunk_frame}")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  Cameras per frame: {num_cams}")
    
    # Collect results from all chunks
    all_depths = []
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * chunk_frame
        end_frame = min(start_frame + chunk_frame, num_frames)
        chunk_num_frames = end_frame - start_frame
        
        print(f"\n  [Chunk {chunk_idx + 1}/{num_chunks}] Processing frames {start_frame} to {end_frame - 1}...")
        
        # Extract images for this chunk
        chunk_images = []
        for frame_idx in range(start_frame, end_frame):
            for cam_id in cam_ids:
                img_idx = frame_idx * num_cams + cam_ids.index(cam_id)
                chunk_images.append(rendered_images[img_idx])
        
        # Extract camera parameters for this chunk
        chunk_extrinsics_list = []
        chunk_intrinsics_list = []
        for frame_idx in range(start_frame, end_frame):
            for cam_id in cam_ids:
                param_idx = frame_idx * num_cams + cam_ids.index(cam_id)
                chunk_extrinsics_list.append(extrinsics_array[param_idx])
                chunk_intrinsics_list.append(intrinsics_array[param_idx])
        chunk_extrinsics = np.stack(chunk_extrinsics_list, axis=0)
        chunk_intrinsics = np.stack(chunk_intrinsics_list, axis=0)
        
        # Run inference for this chunk
        chunk_prediction = model.inference(
            chunk_images,
            export_dir=None,
            extrinsics=chunk_extrinsics,
            intrinsics=chunk_intrinsics,
            process_res=process_resolution,
            process_res_method="upper_bound_resize"
        )
        
        # Collect results
        all_depths.append(chunk_prediction.depth)
        print(f"    Chunk {chunk_idx + 1} completed: {chunk_prediction.depth.shape[0]} views processed")
    
    # Merge all chunks
    print("\n6. Merging results from all chunks...")
    merged_depth = np.concatenate(all_depths, axis=0)
    print(f"  Merged depth shape: {merged_depth.shape}")
    
    # Save to npz
    print("\n7. Saving results to npz...")
    output_dir = Path("reinfer_output")
    output_dir.mkdir(exist_ok=True)
    
    # Interpolate depth to original resolution before saving
    print(f"  Interpolating depth from {merged_depth.shape[1:]} to ({orig_h}, {orig_w})...")
    depth_tensor = torch.from_numpy(merged_depth).float()
    depth_tensor = depth_tensor.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
    depth_resized = F.interpolate(
        depth_tensor,
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=True
    )
    depth_resized = depth_resized.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
    depth_resized_np = depth_resized.numpy().astype(merged_depth.dtype)
    print(f"  Depth shape after interpolation: {depth_resized_np.shape}")
    
    # Save to npz
    npz_path = output_dir / "results.npz"
    np.savez(npz_path, depth=depth_resized_np)
    print(f"  ✓ Saved to {npz_path}")
    
    print("\n" + "="*60)
    print("✓ Reinfer completed!")
    print("="*60)


if __name__ == "__main__":
    main()

