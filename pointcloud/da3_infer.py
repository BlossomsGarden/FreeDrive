"""
根据 深度图、相机内参、相机外参、RGB图 导出特定格式3D点云的逻辑在 depth_anythin_3/utils/export/glb.py 中

cd /data/wlh/DA3/code/Depth-Anything-3-main/src
CUDA_VISIBLE_DEVICES=3 python da3_infer.py
ASCEND_VISIBLE_DEVICES=0 python da3_infer.py

100 帧 3 视角 336*504 MAX 46436MB
60 帧 5视角 238*504 MAX 40558MB
13 帧 3视角 504*1008 MAX 47246MB
"""

import os
from glob import glob
import json
from typing import Tuple
from pathlib import Path
import cv2
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

from depth_anything_3.api import DepthAnything3
from align_depth_with_lidar import process, ID_TO_CAMERA_NAME
from depth_anything_3.utils.export import export

# ------------------------------
# Basic configuration
# ------------------------------
if HAS_NPU:
    data_root = "/home/ma-user/modelarts/user-job-dir/wlh/data/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels"
else:
    data_root = "/data/wlh/FreeDrive/data/waymo/processed/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels"


num_frames = 54
cam_ids = [1, 0, 2]  # 0, 1, 2, 3, 4 for FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
# cam_ids = [0, 1, 2]

# 统一分辨率配置：目标分辨率 672x1008 (width x height)
# process_res=1008 表示最长边缩放到 1008
# lidar_target_resolution=(1008, 672) 表示 (H, W) = (height, width)
# 这样在 lidar alignment 时会将所有数据（深度图、置信度图、lidar投影点）统一缩放到 672x1008
# process_resolution = 504
# lidar_target_resolution = (336, 504)  # (H, W) format for 336x504
# process_resolution = 756
# lidar_target_resolution = (504, 756)  # (H, W) format for 504x756
process_resolution = 1008
lidar_target_resolution = (672, 1008)  # (H, W) format for 672x1008
    

# Inference mode: True = use known camera poses (有参估计), False = estimate poses from model (无参估计)
# 其实我想到一个问题：当你传入有参估计的时候，这不就是已经对齐了尺度吗？
# 因为相机的尺度是真实的，那么为了保证结果可对齐，就会根据相机的尺度去匹配深度图让前后尽量能重合
# 而相机尺度就是真实世界尺度，那我还align个毛线
use_known_poses = True  # Set to False for pose-free estimation

# Lidar alignment control: whether to perform lidar alignment (independent of use_known_poses)
align_depth_with_lidar = False  # Set to True to enable lidar alignment

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
    images_dir = os.path.join(data_root, "images_raw")
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

    # Waymo raw camera resolution (use actual if known; defaults to 1920x1280).
    orig_w, orig_h = 1920, 1280

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
    # Lidar alignment (manually implemented)
    # Note: Lidar alignment requires known camera poses, so skip if pose-free estimation
    # ------------------------------------------------------------------------------------------------
    # Initialize output directory (used by both lidar alignment and GLB export)
    output_base_dir = Path("known_poses_output_5cam")
    
    # Validate inputs
    if prediction.depth is None or prediction.depth.ndim != 3:
        raise ValueError("prediction.depth must be a 3D array (N, H, W)")
    
    num_cams = len(cam_ids)
    total_views, H, W = prediction.depth.shape
    if total_views % num_cams != 0:
        raise ValueError(
            f"Depth count ({total_views}) is not divisible by number of cameras ({num_cams})"
        )
    num_frames_actual = total_views // num_cams
    
    # If target_resolution is None, use current depth resolution (no resize)
    if lidar_target_resolution is None:
        lidar_target_resolution = (H, W)  # Use current resolution
    
    target_h, target_w = lidar_target_resolution
    
    # Only perform lidar alignment if align_depth_with_lidar is True
    if align_depth_with_lidar:
        print("\n" + "="*60)
        print("Starting Lidar Alignment (using known camera poses)")
        print("="*60)
        
        # Configuration for lidar alignment
        lidar_frame_offset = 0
        lidar_save_png = True
        lidar_colormap = "inferno"
        ri_index = 0  # Use first return
    
        # Create directories for raw and aligned depths
        raw_dir = output_base_dir / "aligned_depths" / "raw"
        aligned_dir = output_base_dir / "aligned_depths" / "aligned"
        raw_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir.mkdir(parents=True, exist_ok=True)
        
        # 1) Save raw depths to per-camera npz files
        print(f"\n[Step 1] Saving raw depths to per-camera npz files...")
        depths_reshaped = prediction.depth.reshape(num_frames_actual, num_cams, H, W)
        for cam_idx, cam_id in enumerate(cam_ids):
            npz_path = raw_dir / f"{cam_id}_depths.npz"
            np.savez_compressed(str(npz_path), depths=depths_reshaped[:, cam_idx].astype(np.float32))
            print(f"  Saved raw depths for camera {cam_id}: {depths_reshaped[:, cam_idx].shape}")
        
        # 2) Run lidar alignment using align_depth_with_lidar.process for each camera
        print(f"\n[Step 2] Running lidar alignment for each camera...")
        data_root_path = Path(data_root)
        
        # Check for precomputed lidar projections directory
        pre_dir = data_root_path / "lidar_align"
        if not pre_dir.exists():
            print(f"  Warning: Precomputed lidar projections directory not found: {pre_dir}")
            print(f"  Lidar range statistics will not be available.")
        
        for cam_idx, cam_id in enumerate(cam_ids):
            # Convert camera ID to camera name
            camera_name = ID_TO_CAMERA_NAME.get(cam_id, f"camera_{cam_id}")
            if cam_id not in ID_TO_CAMERA_NAME:
                print(f"  Warning: Camera ID {cam_id} not in ID_TO_CAMERA_NAME mapping, using 'camera_{cam_id}'")
            
            depth_npz_path = raw_dir / f"{cam_id}_depths.npz"
            output_cam_dir = aligned_dir / f"cam_{cam_id}"
            
            print(f"  Processing camera {cam_id} ({camera_name})...")
            
            # Load and analyze lidar projection points for this camera to get range statistics
            if pre_dir.exists():
                lidar_depths_all_frames = []
                for frame_idx in range(num_frames_actual):
                    npz_path = pre_dir / f"{frame_idx:03d}_{cam_id}.npz"
                    if npz_path.exists():
                        data = np.load(npz_path)
                        if "projected" in data:
                            projected_points = data["projected"]
                            # projected_points shape: (N, 3) where columns are [x, y, z] and z is depth
                            if projected_points.shape[1] >= 3:
                                lidar_depths = projected_points[:, 2]  # Extract depth (z) values
                                # Filter valid depths (finite and positive)
                                valid_depths = lidar_depths[np.isfinite(lidar_depths) & (lidar_depths > 0)]
                                if len(valid_depths) > 0:
                                    lidar_depths_all_frames.append(valid_depths)
                
                if len(lidar_depths_all_frames) > 0:
                    # Concatenate all valid depths across all frames for this camera
                    all_lidar_depths = np.concatenate(lidar_depths_all_frames)
                    min_depth = float(np.min(all_lidar_depths))
                    max_depth = float(np.max(all_lidar_depths))
                    mean_depth = float(np.mean(all_lidar_depths))
                    median_depth = float(np.median(all_lidar_depths))
                    num_points = len(all_lidar_depths)
                    print(f"    LiDAR range for camera {cam_id} ({camera_name}):")
                    print(f"      - Min depth: {min_depth:.3f} m")
                    print(f"      - Max depth: {max_depth:.3f} m")
                    print(f"      - Mean depth: {mean_depth:.3f} m")
                    print(f"      - Median depth: {median_depth:.3f} m")
                    print(f"      - Total valid points: {num_points}")
                else:
                    print(f"    Warning: No valid LiDAR points found for camera {cam_id} ({camera_name})")
            else:
                print(f"    Skipping LiDAR range statistics (precomputed projections not found)")
            
            try:
                process(
                    depth_npz_path=depth_npz_path,
                    output_dir=output_cam_dir,
                    camera_name=camera_name,
                    ri_index=ri_index,
                    file_index=cam_idx,
                    total_files=num_cams,
                    data_root=data_root_path,
                    save_raw_overlay=lidar_save_png,
                    save_aligned_overlay=lidar_save_png,
                    target_resolution=lidar_target_resolution,
                )
            except Exception as e:
                print(f"  Error: Failed to process camera {cam_id}: {e}")
                raise
        
        # 3) Load aligned depths from output files
        print(f"\n[Step 3] Loading aligned depths from output files...")
        aligned_per_cam = {}
        for cam_id in cam_ids:
            aligned_path = aligned_dir / f"cam_{cam_id}" / f"{cam_id}_depths.npz"
            if not aligned_path.exists():
                raise FileNotFoundError(
                    f"Aligned depth file not found for camera {cam_id}: {aligned_path}"
                )
            aligned_per_cam[cam_id] = np.load(str(aligned_path))["depths"]
            print(f"  Loaded aligned depths for camera {cam_id}: {aligned_per_cam[cam_id].shape}")
        
        # 4) Reconstruct aligned depths in original order (frame-major, then camera)
        print(f"\n[Step 4] Reconstructing aligned depths in original order...")
        aligned_depths = []
        for frame_idx in range(num_frames_actual):
            for cam_id in cam_ids:
                cam_depths = aligned_per_cam[cam_id]
                if frame_idx >= cam_depths.shape[0]:
                    raise ValueError(
                        f"Aligned depth frames for camera {cam_id} are fewer than expected "
                        f"({cam_depths.shape[0]} < {num_frames_actual})"
                    )
                aligned_depths.append(cam_depths[frame_idx])
        
        aligned_depth_array = np.stack(aligned_depths, axis=0).astype(np.float32)
        
        # 5) Resize depth and confidence to target resolution if needed
        if aligned_depth_array.shape[1:] != (target_h, target_w):
            print(f"\n[Step 5] Resizing depths from {aligned_depth_array.shape[1:]} to {(target_h, target_w)}...")
            resized_depths = []
            for i in range(aligned_depth_array.shape[0]):
                depth_frame = aligned_depth_array[i]
                if depth_frame.shape[0] < target_h or depth_frame.shape[1] < target_w:
                    interpolation = cv2.INTER_CUBIC
                else:
                    interpolation = cv2.INTER_AREA
                resized = cv2.resize(depth_frame, (target_w, target_h), interpolation=interpolation)
                resized_depths.append(resized)
            aligned_depth_array = np.stack(resized_depths, axis=0)
        
        # Update prediction with aligned depths
        prediction.depth = aligned_depth_array
        print(f"  Updated prediction.depth with aligned scales: {prediction.depth.shape}")
        
        # 10) Save aligned depths as npz file
        print(f"\n[Step 10] Saving aligned depths as npz file...")
        npz_output_dir = output_base_dir / "aligned_depths_npz"
        npz_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per-camera npz files
        for cam_id in cam_ids:
            npz_path = npz_output_dir / f"{cam_id}_depths.npz"
            np.savez_compressed(str(npz_path), depths=aligned_per_cam[cam_id].astype(np.float32))
            print(f"  Saved aligned depths for camera {cam_id}: {aligned_per_cam[cam_id].shape} -> {npz_path}")
        
        print(f"\n✓ Lidar alignment completed!")
        print(f"  Aligned depths saved to: {npz_output_dir}")
    else:
        # Skip alignment when align_depth_with_lidar is False
        print("\n" + "="*60)
        print("Skipping Lidar Alignment (align_depth_with_lidar=False)")
        print("="*60)
        
        # Use original depths, but resize to target resolution if needed
        if prediction.depth.shape[1:] != (target_h, target_w):
            print(f"\nResizing depths from {prediction.depth.shape[1:]} to {(target_h, target_w)}...")
            resized_depths = []
            for i in range(prediction.depth.shape[0]):
                depth_frame = prediction.depth[i]
                if depth_frame.shape[0] < target_h or depth_frame.shape[1] < target_w:
                    interpolation = cv2.INTER_CUBIC
                else:
                    interpolation = cv2.INTER_AREA
                resized = cv2.resize(depth_frame, (target_w, target_h), interpolation=interpolation)
                resized_depths.append(resized)
            prediction.depth = np.stack(resized_depths, axis=0).astype(np.float32)
            print(f"  Updated prediction.depth: {prediction.depth.shape}")
    
    # Resize confidence map if it exists
    if prediction.conf is not None:
        if prediction.conf.shape[1:] != (target_h, target_w):
            print(f"\nResizing confidence from {prediction.conf.shape[1:]} to {(target_h, target_w)}...")
            resized_confs = []
            for i in range(prediction.conf.shape[0]):
                conf_frame = prediction.conf[i]
                if conf_frame.shape[0] < target_h or conf_frame.shape[1] < target_w:
                    interpolation = cv2.INTER_CUBIC
                else:
                    interpolation = cv2.INTER_AREA
                resized = cv2.resize(conf_frame, (target_w, target_h), interpolation=interpolation)
                resized_confs.append(resized)
            prediction.conf = np.stack(resized_confs, axis=0)
            print(f"  Updated prediction.conf: {prediction.conf.shape}")
    
    # Resize sky mask if it exists
    if prediction.sky is not None:
        if prediction.sky.shape[1:] != (target_h, target_w):
            print(f"\nResizing sky mask from {prediction.sky.shape[1:]} to {(target_h, target_w)}...")
            resized_skies = []
            for i in range(prediction.sky.shape[0]):
                sky_frame = prediction.sky[i]
                resized = cv2.resize(sky_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                resized_skies.append(resized)
            prediction.sky = np.stack(resized_skies, axis=0)
            print(f"  Updated prediction.sky: {prediction.sky.shape}")
    
    # Resize processed_images if they exist
    if prediction.processed_images is not None:
        if prediction.processed_images.shape[1:3] != (target_h, target_w):
            print(f"\nResizing processed_images from {prediction.processed_images.shape[1:3]} to {(target_h, target_w)}...")
            resized_images = []
            for i in range(prediction.processed_images.shape[0]):
                img_frame = prediction.processed_images[i]  # (H, W, 3)
                if img_frame.shape[0] < target_h or img_frame.shape[1] < target_w:
                    interpolation = cv2.INTER_CUBIC
                else:
                    interpolation = cv2.INTER_AREA
                resized = cv2.resize(img_frame, (target_w, target_h), interpolation=interpolation)
                resized_images.append(resized)
            prediction.processed_images = np.stack(resized_images, axis=0)
            print(f"  Updated prediction.processed_images: {prediction.processed_images.shape}")
    
    # Update intrinsics if resolution changed
    # Note: In pose-free mode, we use model-estimated intrinsics; in known poses mode, we use loaded intrinsics
    if prediction.intrinsics is not None and (H, W) != (target_h, target_w):
        print(f"\nUpdating intrinsics from {(H, W)} to {(target_h, target_w)}...")
        if use_known_poses:
            print("  Using known intrinsics (scaled to target resolution)")
        else:
            print("  Using model-estimated intrinsics (scaled to target resolution)")
        scale_x = target_w / float(W)
        scale_y = target_h / float(H)
        # Ensure intrinsics is numpy array
        intrinsics_array = np.asarray(prediction.intrinsics)
        for i in range(len(intrinsics_array)):
            K = intrinsics_array[i].copy()
            K[0, 0] *= scale_x  # fx
            K[0, 2] *= scale_x  # cx
            K[1, 1] *= scale_y  # fy
            K[1, 2] *= scale_y  # cy
            intrinsics_array[i] = K
        # Keep as numpy array (as expected by export functions)
        prediction.intrinsics = intrinsics_array
        print(f"  Updated intrinsics for {len(intrinsics_array)} views")



    # ------------------------------------------------------------------------------------------------
    # Export to GLB/NPZ
    # ------------------------------------------------------------------------------------------------
    # Export format configuration: "glb", "npz", or "glb-npz" (both)
    export_formats = "glb-npz"  # Change to "glb" or "npz" to export only one format
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
            if prediction.processed_images is None:
                assert False, f"  Warning: prediction.processed_images is None, cannot export NPZ format."
                
            export(
                prediction,
                export_format="npz",
                export_dir=str(export_dir),
            )
            
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
    
    # Print summary
    print(f"\n✓ Export completed!")
    print("="*60)


    