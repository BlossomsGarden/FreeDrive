"""
根据 深度图、相机内参、相机外参、RGB图 导出特定格式3D点云的逻辑在 depth_anythin_3/utils/export/glb.py 中

cd /data/wlh/DA3/code/Depth-Anything-3-main/src
CUDA_VISIBLE_DEVICES=3 

100 帧 3 视角 336*504 MAX 46436MB
60 帧 5视角 238*504 MAX 40558MB（有resize）
"""

import os
from glob import glob
import json
from typing import Tuple
from pathlib import Path
import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from align_depth_with_lidar import process, ID_TO_CAMERA_NAME
from depth_anything_3.utils.export import export


# ------------------------------
# Basic configuration
# ------------------------------
# Update these paths to match your processed Waymo export
# data_root = "/data/wlh/FreeDrive/data/waymo/processed/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels"
data_root = "input003"
num_frames = 40
# cam_ids = [3, 1, 0, 2, 4]  # 0, 1, 2, 3, 4 for FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
cam_ids = [0, 1, 2]

# Inference mode: True = use known camera poses (有参估计), False = estimate poses from model (无参估计)
use_known_poses = True  # Set to False for pose-free estimation


def load_intrinsics(
    intrinsics_dir: str, cam_ids, orig_size: Tuple[int, int], target_size: Tuple[int, int]
):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("/data/wlh/DA3/model/da3nested-giant-large")
    model = model.to(device=device)

    # 统一分辨率配置：目标分辨率 672x1008 (width x height)
    # process_res=1008 表示最长边缩放到 1008
    # lidar_target_resolution=(1008, 672) 表示 (H, W) = (height, width)
    # 这样在 lidar alignment 时会将所有数据（深度图、置信度图、lidar投影点）统一缩放到 672x1008
    process_resolution = 756
    lidar_target_resolution = (504, 756)  # (H, W) format for 336x504
    # process_resolution = 1008
    # lidar_target_resolution = (672, 1008)  # (H, W) format for 672x1008
    
    # Run inference based on mode
    if use_known_poses:
        # 有参估计: 传入已知的相机内外参
        prediction = model.inference(
            images,
            export_dir=None,
            extrinsics=extrinsics_array,
            intrinsics=intrinsics_array,
            process_res=process_resolution,
            process_res_method="upper_bound_resize"
        )
    else:
        # 无参估计: 不传入相机参数，让模型估计
        prediction = model.inference(
            images,
            export_dir=None,
            process_res=process_resolution,
            process_res_method="upper_bound_resize"
        )
        
        # 打印模型估计的相机内外参
        print("\n" + "="*60)
        print("Model-estimated camera parameters (模型估计的相机参数)")
        print("="*60)
        print(f"Extrinsics shape: {np.asarray(prediction.extrinsics).shape}")  # Camera poses (w2c): [N, 3, 4] or [N, 4, 4] float32
        print(f"Intrinsics shape: {np.asarray(prediction.intrinsics).shape}")   # Camera intrinsics: [N, 3, 3] float32
        print(f"Number of views: {len(prediction.extrinsics)}")
        
        # Convert extrinsics to numpy for easier inspection
        pred_extrinsics = np.asarray(prediction.extrinsics)
        pred_intrinsics = np.asarray(prediction.intrinsics)


    # ------------------------------------------------------------------------------------------------
    # Lidar alignment (manually implemented)
    # Note: Lidar alignment requires known camera poses, so skip if pose-free estimation
    # ------------------------------------------------------------------------------------------------
    # Initialize output directory (used by both lidar alignment and GLB export)
    output_base_dir = Path("output")
    
    # Validate inputs
    if prediction.depth is None or prediction.depth.ndim != 3:
        raise ValueError("prediction.depth must be a 3D array (N, H, W)")
    
    num_cams = len(cam_ids)
    total_views, H, W = prediction.depth.shape
    if total_views % num_cams != 0:
        raise ValueError(
            f"Depth count ({total_views}) is not divisible by number of cameras ({num_cams})"
        )
    num_frames = total_views // num_cams
    
    # If target_resolution is None, use current depth resolution (no resize)
    if lidar_target_resolution is None:
        lidar_target_resolution = (H, W)  # Use current resolution
    
    target_h, target_w = lidar_target_resolution
    
    # Only perform lidar alignment if using known poses
    if use_known_poses:
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
        depths_reshaped = prediction.depth.reshape(num_frames, num_cams, H, W)
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
                for frame_idx in range(num_frames):
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
        for frame_idx in range(num_frames):
            for cam_id in cam_ids:
                cam_depths = aligned_per_cam[cam_id]
                if frame_idx >= cam_depths.shape[0]:
                    raise ValueError(
                        f"Aligned depth frames for camera {cam_id} are fewer than expected "
                        f"({cam_depths.shape[0]} < {num_frames})"
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
        
        # Save per-camera npz files (same format as custom_data_to_glb.py expects)
        for cam_id in cam_ids:
            npz_path = npz_output_dir / f"{cam_id}_depths.npz"
            np.savez_compressed(str(npz_path), depths=aligned_per_cam[cam_id].astype(np.float32))
            print(f"  Saved aligned depths for camera {cam_id}: {aligned_per_cam[cam_id].shape} -> {npz_path}")
        
        print(f"\n✓ Lidar alignment completed!")
        print(f"  Aligned depths saved to: {npz_output_dir}")
    else:
        # Skip alignment when using pose-free estimation
        print("\n" + "="*60)
        print("Skipping Lidar Alignment (pose-free estimation mode)")
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
        for i in range(len(prediction.intrinsics)):
            K = prediction.intrinsics[i].copy()
            K[0, 0] *= scale_x  # fx
            K[0, 2] *= scale_x  # cx
            K[1, 1] *= scale_y  # fy
            K[1, 2] *= scale_y  # cy
            prediction.intrinsics[i] = K
        print(f"  Updated intrinsics for {len(prediction.intrinsics)} views")



    # ------------------------------------------------------------------------------------------------
    # Export to GLB using custom_data_to_glb logic
    # ------------------------------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Exporting to GLB")
    print("="*60)
    
    # GLB export parameters
    glb_output_dir = output_base_dir / "glb_output"
    glb_output_dir.mkdir(parents=True, exist_ok=True)
    
    conf_thresh_percentile = 40.0
    num_max_points = 2_000_000
    show_cameras = True
    prediction.is_metric = 1
    
    # Ensure we have confidence maps
    if prediction.conf is None:
        print("  Creating default confidence maps (all ones)...")
        prediction.conf = np.ones_like(prediction.depth, dtype=np.float32)
    
    print(f"  Exporting GLB with {len(prediction.depth)} views...")
    print(f"    - Depth shape: {prediction.depth.shape}")
    print(f"    - Confidence shape: {prediction.conf.shape}")
    print(f"    - Extrinsics shape: {np.asarray(prediction.extrinsics).shape}")
    print(f"    - Intrinsics shape: {np.asarray(prediction.intrinsics).shape}")
    print(f"    - Max points: {num_max_points}")
    print(f"    - Confidence threshold percentile: {conf_thresh_percentile}")
    
    # Export to GLB
    export(
        prediction,
        export_format="glb",
        export_dir=str(glb_output_dir),
        glb={
            "num_max_points": num_max_points,
            "conf_thresh_percentile": conf_thresh_percentile,
            "show_cameras": show_cameras,
        },
    )
    
    print(f"\n✓ GLB exported to: {glb_output_dir / 'scene.glb'}")
    print("="*60)


    