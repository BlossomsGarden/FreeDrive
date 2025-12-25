"""
根据 深度图、相机内参、相机外参、RGB图 导出特定格式3D点云的逻辑在 depth_anythin_3/utils/export/glb.py 中

cd /data/wlh/DA3/code/Depth-Anything-3-main/src
CUDA_VISIBLE_DEVICES=3 python da3_infer.py
ASCEND_RT_VISIBLE_DEVICES=0 python da3_infer.py

100 帧 3 视角 336*504 MAX 46436MB
60 帧 5视角 238*504 MAX 40558MB
13 帧 3视角 504*1008 MAX 47246MB
"""

import os
from glob import glob
from typing import Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Try to import torch_npu for NPU 910B support
try:
    import torch_npu
    HAS_NPU = True
    print("✓ torch_npu imported successfully - using NPU 910B device")
except ImportError:
    HAS_NPU = False
    print("✓ torch_npu not available - using CUDA/CPU device")

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export import export

# ------------------------------
# Basic configuration
# ------------------------------
if HAS_NPU:
    data_root = "/home/ma-user/modelarts/user-job-dir/wlh/code/FreeDrive/vda/toy_data/10050"
else:
    data_root = "/data/wlh/FreeDrive/data/waymo/processed/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels"


num_frames = 18
cam_ids = [1, 0, 2]  # 0, 1, 2, 3, 4 for FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT

# Waymo raw camera resolution (use actual if known; defaults to 1920x1280).
orig_w, orig_h = 1920, 1280

# Inference resolution configuration
# process_res=1008 表示最长边缩放到 1008  即 (1920, 1280) -> (1008, 672)
process_resolution = 1008

# Inference mode: True = use known camera poses (有参估计), False = estimate poses from model (无参估计)
use_known_poses = True  # Set to False for pose-free estimation

# Chunk-based inference: number of frames to process per inference chunk (to avoid OOM)
chunk_frame = 18  # Process chunk_frame frames at a time


def load_intrinsics(intrinsics_dir: str, cam_ids, orig_size: Tuple[int, int], target_size: Tuple[int, int]):
    """Load K matrices (3x3) from Waymo intrinsics txt files and scale to target size.

    Files store [fx, fy, cx, cy, k1, k2, p1, p2, k3]; distortion terms are
    ignored for Depth-Anything input.
    """
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


def load_camera_extrinsics(extrinsics_dir: str, cam_ids):
    """Load camera->ego extrinsics (4x4) for each chosen camera id.
    
    Waymo stores camera->ego transformation matrix.
    """
    return {
        cam_id: np.loadtxt(os.path.join(extrinsics_dir, f"{cam_id}.txt")).reshape(4, 4)
        for cam_id in cam_ids
    }


def load_ego_poses(ego_pose_dir: str, num_frames: int):
    """Load the first num_frames ego->world poses (4x4)."""
    pose_files = sorted(
        glob(os.path.join(ego_pose_dir, "*.txt")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )[:num_frames]
    return [np.loadtxt(pf).reshape(4, 4) for pf in pose_files]


def waymo_to_cv_coordinate_transform():
    """Convert Waymo vehicle coordinate system to CV coordinate system.
    
    Waymo vehicle: x=forward, y=left, z=up
    CV: x=right, y=down, z=forward
    
    Returns 4x4 transformation matrix from Waymo vehicle frame to CV frame.
    """
    # Rotation matrix: Waymo -> CV
    # CV x-axis (right) = -Waymo y-axis (left's opposite)
    # CV y-axis (down) = -Waymo z-axis (up's opposite)  
    # CV z-axis (forward) = Waymo x-axis (forward)
    R = np.array([
        [0, -1, 0, 0],  # CV x = -Waymo y
        [0, 0, -1, 0],  # CV y = -Waymo z
        [1, 0, 0, 0],   # CV z = Waymo x
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return R


def compute_world_to_camera(cam_to_ego: np.ndarray, ego_to_world: np.ndarray):
    """Compute world->camera matrix given camera->ego and ego->world.
    
    Args:
        cam_to_ego: Camera->ego transform from Waymo (camera to ego/vehicle)
                   Note: Waymo camera may use vehicle coordinate system (x=forward, y=left, z=up)
        ego_to_world: Ego->world transform (both in Waymo vehicle coordinate system)
    
    Returns:
        world->camera transform in CV coordinate system (as expected by Depth-Anything-3)
    """
    # Coordinate system conversion: Waymo vehicle -> CV
    waymo_to_cv = waymo_to_cv_coordinate_transform()
    cv_to_waymo = np.linalg.inv(waymo_to_cv)
    
    # Transform chain:
    # We have: camera->ego (Waymo camera -> Waymo ego) and ego->world (Waymo ego -> Waymo world)
    # We need: world->camera (CV world -> CV camera)
    #
    # Step 1: Convert camera->ego to CV coordinate system
    # cam_to_ego: Waymo camera -> Waymo ego
    # If Waymo camera uses vehicle coordinate system, we need to convert both input and output frames
    # CV camera -> CV ego = waymo_to_cv @ (Waymo camera -> Waymo ego) @ cv_to_waymo
    cam_to_ego_cv = waymo_to_cv @ cam_to_ego @ cv_to_waymo
    
    # Step 2: Convert ego->world to CV coordinate system  
    # ego_to_world: Waymo ego -> Waymo world
    # We need: CV ego -> CV world
    # Transform both input and output frames: CV ego -> CV world = waymo_to_cv @ (Waymo ego -> Waymo world) @ cv_to_waymo
    ego_to_world_cv = waymo_to_cv @ ego_to_world @ cv_to_waymo
    
    # Step 3: Compute world->ego in CV frame
    world_to_ego_cv = np.linalg.inv(ego_to_world_cv)
    
    # Step 4: Compute ego->camera in CV frame
    ego_to_cam_cv = np.linalg.inv(cam_to_ego_cv)
    
    # Step 5: Combine transforms (all in CV frame now)
    # world->camera = ego->camera @ world->ego
    world_to_cam = ego_to_cam_cv @ world_to_ego_cv
    
    return world_to_cam


# ------------------------------------------------------------------------------------------------
# Prepare inputs
# ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    images_dir = os.path.join(data_root, "images")
    intrinsics_dir = os.path.join(data_root, "intrinsics")
    extrinsics_dir = os.path.join(data_root, "extrinsics")
    ego_pose_dir = os.path.join(data_root, "ego_pose")

    # Detect target image size from the first available image.
    sample_image = None
    for frame_id in range(num_frames):
        for cam_id in cam_ids:
            candidate = os.path.join(images_dir, f"{frame_id:03d}_{cam_id}.jpg")
            if os.path.exists(candidate):
                sample_image = candidate
                break
        if sample_image:
            break
    if sample_image is None:
        raise FileNotFoundError("No sample image found to infer target resolution.")
    from PIL import Image  # pillow is already a dependency in project

    with Image.open(sample_image) as im:
        target_w, target_h = im.size


    # Load camera parameters only if using known poses (有参估计)
    extrinsics_array = None
    intrinsics_array = None
    
    if use_known_poses:
        print("="*60)
        print("Mode: Using known camera poses (有参估计)")
        print("="*60)
        intrinsics_per_cam = load_intrinsics(intrinsics_dir, cam_ids, (orig_w, orig_h), (target_w, target_h))
        cam_to_ego_map = load_camera_extrinsics(extrinsics_dir, cam_ids)  # camera->ego transforms
        ego_to_world_list = load_ego_poses(ego_pose_dir, num_frames)  # ego->world transforms

        extrinsics_list = []
        intrinsics_list = []
        for ego_to_world in ego_to_world_list:
            for cam_id in cam_ids:
                w2c = compute_world_to_camera(cam_to_ego_map[cam_id], ego_to_world)
                extrinsics_list.append(w2c.astype(np.float32))
                intrinsics_list.append(intrinsics_per_cam[cam_ids.index(cam_id)])

        extrinsics_array = np.stack(extrinsics_list, axis=0)  # (num_frames*len(cam_ids), 4, 4)
        intrinsics_array = np.stack(intrinsics_list, axis=0)  # (num_frames*len(cam_ids), 3, 3)
        print(f"Loaded {len(extrinsics_list)} camera poses and intrinsics")
    else:
        print("="*60)
        print("Mode: Pose-free estimation (无参估计)")
        print("="*60)
        print("Camera poses and intrinsics will be estimated by the model")

    # Load images strictly in frame-major then cam_ids order to match arrays.
    images = []
    for frame_id in range(num_frames):
        for cam_id in cam_ids:
            fname = f"{frame_id:03d}_{cam_id}.jpg"
            fpath = os.path.join(images_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing image {fname} for frame {frame_id}, cam {cam_id}")
            images.append(fpath)



    # ------------------------------------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------------------------------------
    # Auto-detect device: NPU 910B (if torch_npu available) or CUDA/CPU
    device = torch.device("cpu")
    if HAS_NPU:
        model = DepthAnything3.from_pretrained("/home/ma-user/modelarts/user-job-dir/wlh/model/da3nested-giant-large")
        device = torch.device("npu")
        print(f"Using device: NPU 910B")
    elif torch.cuda.is_available():
        model = DepthAnything3.from_pretrained("/data/wlh/DA3/model/da3nested-giant-large")
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    
    model = model.to(device=device)

    num_cams = len(cam_ids)
    num_chunks = (num_frames + chunk_frame - 1) // chunk_frame  # Ceiling division
    
    print("\n" + "="*60)
    print(f"Chunk-based Inference Configuration")
    print("="*60)
    print(f"Total frames: {num_frames}")
    print(f"Frames per chunk: {chunk_frame}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Cameras per frame: {num_cams}")
    print(f"Total images: {len(images)}")
    
    # Collect results from all chunks
    all_depths = []
    all_confs = []
    all_skies = []
    all_processed_images = []
    all_extrinsics = []
    all_intrinsics = []
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * chunk_frame
        end_frame = min(start_frame + chunk_frame, num_frames)
        chunk_num_frames = end_frame - start_frame
        
        print(f"\n[Chunk {chunk_idx + 1}/{num_chunks}] Processing frames {start_frame} to {end_frame - 1} ({chunk_num_frames} frames)...")
        
        # Extract images for this chunk
        chunk_images = []
        for frame_idx in range(start_frame, end_frame):
            for cam_id in cam_ids:
                img_idx = frame_idx * num_cams + cam_ids.index(cam_id)
                chunk_images.append(images[img_idx])
        
        # Extract camera parameters for this chunk (if using known poses)
        chunk_extrinsics = None
        chunk_intrinsics = None
        if use_known_poses:
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
        if use_known_poses:
            # 有参估计: 传入已知的相机内外参
            chunk_prediction = model.inference(
                chunk_images,
                export_dir=None,
                extrinsics=chunk_extrinsics,
                intrinsics=chunk_intrinsics,
                process_res=process_resolution,
                process_res_method="upper_bound_resize"
            )
        else:
            # 无参估计: 不传入相机参数，让模型估计
            chunk_prediction = model.inference(
                chunk_images,
                export_dir=None,
                process_res=process_resolution,
                process_res_method="upper_bound_resize"
            )
        
        # Collect results from this chunk
        all_depths.append(chunk_prediction.depth)
        if chunk_prediction.conf is not None:
            all_confs.append(chunk_prediction.conf)
        if chunk_prediction.sky is not None:
            all_skies.append(chunk_prediction.sky)
        if chunk_prediction.processed_images is not None:
            all_processed_images.append(chunk_prediction.processed_images)
        if chunk_prediction.extrinsics is not None:
            chunk_ext = np.asarray(chunk_prediction.extrinsics)
            all_extrinsics.append(chunk_ext)
        if chunk_prediction.intrinsics is not None:
            chunk_int = np.asarray(chunk_prediction.intrinsics)
            all_intrinsics.append(chunk_int)
        
        print(f"  Chunk {chunk_idx + 1} completed: {chunk_prediction.depth.shape[0]} views processed")
    
    # Merge all chunks
    print("\n" + "="*60)
    print("Merging results from all chunks...")
    print("="*60)
    
    merged_depth = np.concatenate(all_depths, axis=0)
    print(f"Merged depth shape: {merged_depth.shape}")
    
    merged_conf = None
    if len(all_confs) > 0:
        merged_conf = np.concatenate(all_confs, axis=0)
        print(f"Merged confidence shape: {merged_conf.shape}")
    
    merged_sky = None
    if len(all_skies) > 0:
        merged_sky = np.concatenate(all_skies, axis=0)
        print(f"Merged sky mask shape: {merged_sky.shape}")
    
    merged_processed_images = None
    if len(all_processed_images) > 0:
        merged_processed_images = np.concatenate(all_processed_images, axis=0)
        print(f"Merged processed images shape: {merged_processed_images.shape}")
    
    merged_extrinsics = None
    if len(all_extrinsics) > 0:
        merged_extrinsics = np.concatenate(all_extrinsics, axis=0)
        print(f"Merged extrinsics shape: {merged_extrinsics.shape}")
    
    merged_intrinsics = None
    if len(all_intrinsics) > 0:
        merged_intrinsics = np.concatenate(all_intrinsics, axis=0)
        print(f"Merged intrinsics shape: {merged_intrinsics.shape}")
    
    # Create a unified prediction object
    # We'll use the last chunk's prediction as a template and update its attributes
    prediction = chunk_prediction
    prediction.depth = merged_depth
    prediction.conf = merged_conf
    prediction.sky = merged_sky
    prediction.processed_images = merged_processed_images
    # Keep extrinsics and intrinsics as numpy arrays (not lists) as expected by export functions
    if merged_extrinsics is not None:
        prediction.extrinsics = merged_extrinsics
    if merged_intrinsics is not None:
        prediction.intrinsics = merged_intrinsics
    
    if not use_known_poses:
        # 打印模型估计的相机内外参
        print("\n" + "="*60)
        print("Model-estimated camera parameters (模型估计的相机参数)")
        print("="*60)
        print(f"Extrinsics shape: {merged_extrinsics.shape if merged_extrinsics is not None else 'None'}")  # Camera poses (w2c): [N, 3, 4] or [N, 4, 4] float32
        print(f"Intrinsics shape: {merged_intrinsics.shape if merged_intrinsics is not None else 'None'}")   # Camera intrinsics: [N, 3, 3] float32
        print(f"Number of views: {len(prediction.extrinsics) if prediction.extrinsics is not None else 0}")


    # ------------------------------------------------------------------------------------------------
    # Prepare for export
    # ------------------------------------------------------------------------------------------------
    # Initialize output directory
    output_base_dir = Path("known_poses_output_5cam")
    
    # Validate inputs
    if prediction.depth is None or prediction.depth.ndim != 3:
        raise ValueError("prediction.depth must be a 3D array (N, H, W)")



    # ------------------------------------------------------------------------------------------------
    # Export to GLB/NPZ
    # ------------------------------------------------------------------------------------------------
    # Export format configuration: "glb", "npz", or "glb-npz" (both)
    export_formats = "npz"  # Change to "glb" or "npz" to export only one format
    formats_list = export_formats.split("-") if "-" in export_formats else [export_formats]
    
    print("\n" + "="*60)
    print(f"Exporting to: {formats_list}")
    print("="*60)
    
    # Ensure we have confidence maps
    if prediction.conf is None:
        print("  Creating default confidence maps (all ones)...")
        prediction.conf = np.ones_like(prediction.depth, dtype=np.float32)
    
    print(f"  Exporting with {len(prediction.depth)} views...")
    
    # Note: GLB saves to {export_dir}/scene.glb
    #       NPZ saves to {export_dir}/exports/npz/results.npz
    # Since paths differ, we handle each format separately
    
    export_dir = output_base_dir / "output"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export each format
    for fmt in formats_list:
        if fmt == "glb":
            # GLB export parameters (only used if exporting GLB)
            conf_thresh_percentile = 30.0
            num_max_points = 2_000_000
            show_cameras = True
            prediction.is_metric = 1
            glb_kwargs = {
                "num_max_points": num_max_points,
                "conf_thresh_percentile": conf_thresh_percentile,
                "show_cameras": show_cameras,
            } 
    
            export(
                prediction,
                export_format="glb",
                export_dir=str(export_dir),
                glb=glb_kwargs,
            )
            
        elif fmt == "npz":
            # Interpolate depth to original resolution before saving
            print(f"  Interpolating depth from {prediction.depth.shape[1:]} to ({orig_h}, {orig_w})...")
            # Convert depth to tensor for interpolation
            # prediction.depth shape: (N, H, W)
            depth_tensor = torch.from_numpy(prediction.depth).float()
            # Add batch and channel dimensions: (N, H, W) -> (N, 1, H, W)
            depth_tensor = depth_tensor.unsqueeze(1)
            # Interpolate to original resolution
            depth_resized = F.interpolate(
                depth_tensor,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=True
            )
            # Remove channel dimension: (N, 1, H, W) -> (N, H, W)
            depth_resized = depth_resized.squeeze(1)
            # Convert back to numpy
            depth_resized_np = depth_resized.numpy().astype(prediction.depth.dtype)
            print(f"  Depth shape after interpolation: {depth_resized_np.shape}")
            
            # Save only depth to npz file
            npz_path = export_dir / "results.npz"
            np.savez(npz_path, depth=depth_resized_np)
            print(f"  Saved depth to {npz_path}")
            
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
    
    # Print summary
    print(f"\n✓ Export completed!")
    print("="*60)


    