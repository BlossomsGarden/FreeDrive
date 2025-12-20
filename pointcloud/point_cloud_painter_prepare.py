"""
Prepare training data for DiT model to synthesize novel camera views from point cloud.

This script implements the following pipeline:
1. Run DA3 inference on original 3 cameras (1,0,2) with original poses to get point cloud
2. Apply trajectory offset (e.g., FRONT camera right 1m) to all 3 cameras
3. Render point cloud projections and masks using offset trajectory cameras
4. Run DA3 inference again on rendered projections with offset poses to get new point cloud
5. Render final projections using original FRONT camera trajectory from new point cloud
6. Save: occluded point cloud images, masks, and GT (original FRONT RGB images)

Usage:
    python point_cloud_painter_prepare.py
"""

import os
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Try to import torch_npu for NPU 910B support
try:
    import torch_npu
    HAS_NPU = True
    print("✓ torch_npu imported successfully - using NPU 910B device")
except ImportError:
    HAS_NPU = False
    print("✓ torch_npu not available - using CUDA/CPU device")

from depth_anything_3.api import DepthAnything3
from render_from_npz import (
    load_npz_data,
    depths_to_world_points_with_colors_single_frame,
    render_pointcloud_view,
    compute_world_to_camera_from_waymo,
    waymo_to_cv_coordinate_transform,
)
from da3_infer import (
    load_intrinsics as da3_load_intrinsics,
    load_camera_extrinsics,
    load_ego_poses,
    compute_world_to_camera,
)


# ============================================================================
# Configuration
# ============================================================================
if HAS_NPU:
    data_root = "/home/ma-user/modelarts/user-job-dir/wlh/data/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels"
else:
    data_root = "/data/wlh/FreeDrive/data/waymo/processed/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels"

num_frames = 54
cam_ids = [1, 0, 2]  # 0, 1, 2, 3, 4 for FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
render_cam_id = 0  # FRONT camera for final rendering and GT

# Resolution configuration
process_resolution = 1008
orig_size = (1920, 1280)  # Waymo default

# Camera offset in Waymo vehicle coordinates (meters)
# In Waymo coordinates: x=forward(+), y=left(+), z=up(+)
# camera_offset = [1.0, 0.0, 0.0]   # Example: Forward 1 meter
camera_offset = [0.0, -1.5, 0.0]  # Example: Right 1.5 meter
# camera_offset = [0.0, 0.0, 1.0]   # Example: Up 1 meter

# Output directories
output_base_dir = Path("point_cloud_painter_data")
step1_output_dir = output_base_dir / "step1_original_pointcloud"
step3_output_dir = output_base_dir / "step3_offset_rendered_images"
step4_output_dir = output_base_dir / "step4_offset_pointcloud"
final_output_dir = output_base_dir / "final_training_data"

# Chunk-based inference: number of frames to process per inference chunk
chunk_frame = 18


# ============================================================================
# Helper Functions
# ============================================================================

def waymo_to_cv_coordinate_transform():
    """Convert Waymo vehicle coordinate system to CV coordinate system."""
    R = np.array([
        [0, -1, 0, 0],  # CV x = -Waymo y
        [0, 0, -1, 0],  # CV y = -Waymo z
        [1, 0, 0, 0],   # CV z = Waymo x
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return R


def apply_camera_offset_to_extrinsic(cam_to_ego: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Apply camera offset to camera-to-ego extrinsic matrix.
    
    Args:
        cam_to_ego: (4, 4) camera-to-ego transform (Waymo format)
        offset: (3,) offset in Waymo vehicle coordinates [x, y, z]
        
    Returns:
        Modified cam_to_ego matrix with offset applied
    """
    cam_to_ego_modified = cam_to_ego.copy()
    cam_to_ego_modified[0, 3] += offset[0]  # forward/backward
    cam_to_ego_modified[1, 3] += offset[1]  # left/right
    cam_to_ego_modified[2, 3] += offset[2]  # up/down
    return cam_to_ego_modified


def save_images_and_params_for_inference(
    images: list,
    intrinsics_list: list,
    extrinsics_list: list,
    output_dir: Path,
    num_frames: int,
    cam_ids: list,
):
    """
    Save rendered images and camera parameters in format expected by da3_infer.py.
    
    Args:
        images: List of (H, W, 3) uint8 RGB images, frame-major then camera order
        intrinsics_list: List of (3, 3) camera intrinsics, same order
        extrinsics_list: List of (4, 4) camera extrinsics, same order
        output_dir: Directory to save images and parameters
        num_frames: Number of frames
        cam_ids: List of camera IDs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images_raw"
    intrinsics_dir = output_dir / "intrinsics"
    extrinsics_dir = output_dir / "extrinsics"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_dir.mkdir(parents=True, exist_ok=True)
    extrinsics_dir.mkdir(parents=True, exist_ok=True)
    
    num_cams = len(cam_ids)
    
    # Save images (frame-major then camera order: frame0_cam0, frame0_cam1, ...)
    for idx, (image, intrinsic, extrinsic) in enumerate(zip(images, intrinsics_list, extrinsics_list)):
        frame_idx = idx // num_cams
        cam_idx = idx % num_cams
        cam_id = cam_ids[cam_idx]
        
        # Save image
        image_path = images_dir / f"{frame_idx:03d}_{cam_id}.jpg"
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image_bgr)
        
        # Save intrinsics (per camera, same for all frames)
        # Format: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        # We only have fx, fy, cx, cy from the 3x3 matrix
        K = intrinsic
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        intrinsic_params = np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        intrinsic_path = intrinsics_dir / f"{cam_id}.txt"
        np.savetxt(str(intrinsic_path), intrinsic_params)
        
        # Save extrinsics (per camera, same for all frames)
        # We need to convert from world-to-camera (CV) back to camera-to-ego (Waymo)
        # This is complex, so we'll save the world-to-camera directly
        # But da3_infer.py expects camera-to-ego, so we need to convert
        # Actually, da3_infer.py expects extrinsics as world-to-camera in CV coordinates
        # So we can save them directly, but we need to make sure they're in the right format
        # For now, let's save the world-to-camera extrinsics
        # Note: We'll need to handle this differently - da3_infer.py loads camera-to-ego from files
        # and then computes world-to-camera. We need to provide camera-to-ego instead.
        # This is a problem - we need to invert the transform chain.
        # Actually, for the second inference, we can modify da3_infer.py logic or create a wrapper
        # For simplicity, let's save the world-to-camera matrices and handle the conversion later
        
        # Save as world-to-camera (CV format) - we'll handle conversion in a modified inference function
        extrinsic_path = extrinsics_dir / f"{cam_id}.txt"
        np.savetxt(str(extrinsic_path), extrinsic)
    
    print(f"Saved {len(images)} images and camera parameters to {output_dir}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("="*60)
    print("Point Cloud Painter Training Data Preparation")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Number of frames: {num_frames}")
    print(f"Camera IDs: {cam_ids}")
    print(f"Render camera ID: {render_cam_id}")
    print(f"Camera offset (Waymo coords): {camera_offset}")
    print("="*60)
    
    # Setup paths
    images_dir = os.path.join(data_root, "images_raw")
    intrinsics_dir = os.path.join(data_root, "intrinsics")
    extrinsics_dir = os.path.join(data_root, "extrinsics")
    ego_pose_dir = os.path.join(data_root, "ego_pose")
    
    # Detect target image size
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
    
    with Image.open(sample_image) as im:
        target_w, target_h = im.size
    
    # Load model
    print("\n" + "="*60)
    print("Loading DA3 model...")
    print("="*60)
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
    
    # ============================================================================
    # Step 1: Run DA3 inference on original cameras with original poses
    # ============================================================================
    print("\n" + "="*60)
    print("Step 1: Running DA3 inference on original cameras with original poses")
    print("="*60)
    
    # Load original camera parameters
    intrinsics_per_cam = da3_load_intrinsics(intrinsics_dir, cam_ids, orig_size, (target_w, target_h))
    cam_to_ego_map = load_camera_extrinsics(extrinsics_dir, cam_ids)
    ego_to_world_list = load_ego_poses(ego_pose_dir, num_frames)
    
    # Compute world-to-camera transforms for original poses
    extrinsics_list_original = []
    intrinsics_list_original = []
    images_original = []
    
    for ego_to_world in ego_to_world_list:
        for cam_id in cam_ids:
            w2c = compute_world_to_camera(cam_to_ego_map[cam_id], ego_to_world)
            extrinsics_list_original.append(w2c.astype(np.float32))
            intrinsics_list_original.append(intrinsics_per_cam[cam_ids.index(cam_id)])
            # Load original image
            fname = f"{len(images_original) // len(cam_ids):03d}_{cam_id}.jpg"
            fpath = os.path.join(images_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing image {fname}")
            images_original.append(fpath)
    
    extrinsics_array_original = np.stack(extrinsics_list_original, axis=0)
    intrinsics_array_original = np.stack(intrinsics_list_original, axis=0)
    
    # Chunk-based inference to avoid OOM
    num_cams = len(cam_ids)
    num_chunks = (num_frames + chunk_frame - 1) // chunk_frame  # Ceiling division
    
    print(f"\nChunk-based Inference Configuration:")
    print(f"  Total frames: {num_frames}")
    print(f"  Frames per chunk: {chunk_frame}")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  Cameras per frame: {num_cams}")
    print(f"  Total images: {len(images_original)}")
    
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
                chunk_images.append(images_original[img_idx])
        
        # Extract camera parameters for this chunk
        chunk_extrinsics_list = []
        chunk_intrinsics_list = []
        for frame_idx in range(start_frame, end_frame):
            for cam_id in cam_ids:
                param_idx = frame_idx * num_cams + cam_ids.index(cam_id)
                chunk_extrinsics_list.append(extrinsics_array_original[param_idx])
                chunk_intrinsics_list.append(intrinsics_array_original[param_idx])
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
    
    # For known poses mode, model doesn't return extrinsics/intrinsics, use original arrays
    # For pose-free mode, model returns them, so merge them
    merged_extrinsics = None
    if len(all_extrinsics) > 0:
        merged_extrinsics = np.concatenate(all_extrinsics, axis=0)
        print(f"Merged extrinsics shape: {merged_extrinsics.shape}")
    else:
        # Use original arrays (known poses mode)
        merged_extrinsics = extrinsics_array_original
        print(f"Using original extrinsics (known poses mode): {merged_extrinsics.shape}")
    
    merged_intrinsics = None
    if len(all_intrinsics) > 0:
        merged_intrinsics = np.concatenate(all_intrinsics, axis=0)
        print(f"Merged intrinsics shape: {merged_intrinsics.shape}")
    else:
        # Use original arrays (known poses mode)
        merged_intrinsics = intrinsics_array_original
        print(f"Using original intrinsics (known poses mode): {merged_intrinsics.shape}")
    
    # Create a unified prediction object
    prediction_original = chunk_prediction
    prediction_original.depth = merged_depth
    prediction_original.conf = merged_conf
    prediction_original.sky = merged_sky
    prediction_original.processed_images = merged_processed_images
    prediction_original.extrinsics = merged_extrinsics
    prediction_original.intrinsics = merged_intrinsics
    
    # Export to npz
    step1_output_dir.mkdir(parents=True, exist_ok=True)
    npz_output_path = step1_output_dir / "results.npz"
    
    save_dict = {
        "image": prediction_original.processed_images,
        "depth": np.round(prediction_original.depth, 6),
    }
    if prediction_original.conf is not None:
        save_dict["conf"] = np.round(prediction_original.conf, 2)
    save_dict["extrinsics"] = prediction_original.extrinsics
    save_dict["intrinsics"] = prediction_original.intrinsics
    
    np.savez_compressed(str(npz_output_path), **save_dict)
    print(f"Step 1 completed: Saved point cloud to {npz_output_path}")
    
    # ============================================================================
    # Step 2 & 3: Apply offset to cameras and render point cloud projections
    # ============================================================================
    print("\n" + "="*60)
    print("Step 2 & 3: Applying camera offset and rendering projections")
    print("="*60)
    
    # Load point cloud from Step 1
    npz_data = load_npz_data(str(npz_output_path))
    images_npz = npz_data['images']  # (N, H, W, 3) uint8
    depths_npz = npz_data['depths']  # (N, H, W)
    extrinsics_npz = npz_data['extrinsics']  # (N, 4, 4) or (N, 3, 4)
    intrinsics_npz = npz_data['intrinsics']  # (N, 3, 3)
    conf_npz = npz_data.get('conf', None)
    
    num_total_views = depths_npz.shape[0]
    num_cams = len(cam_ids)
    num_frames_actual = num_total_views // num_cams
    
    H_actual, W_actual = images_npz.shape[1], images_npz.shape[2]
    print(f"Detected actual resolution from processed_images: {H_actual} x {W_actual} (H x W)")
    
    # Apply offset to camera-to-ego extrinsics
    camera_offset_array = np.array(camera_offset, dtype=np.float32)
    cam_to_ego_map_offset = {}
    for cam_id in cam_ids:
        cam_to_ego_map_offset[cam_id] = apply_camera_offset_to_extrinsic(
            cam_to_ego_map[cam_id], camera_offset_array
        )
    
    # Compute world-to-camera transforms with offset
    extrinsics_list_offset = []
    intrinsics_list_offset = []
    rendered_images_offset = []
    rendered_masks_offset = []
    
    print(f"Rendering {num_frames_actual} frames with offset trajectory...")
    for frame_idx in tqdm(range(num_frames_actual), desc="Rendering"):
        # Build point cloud for this frame (from original point cloud)
        frame_depths = []
        frame_images = []
        frame_intrinsics = []
        frame_extrinsics = []
        frame_conf = []
        
        for cam_idx, _ in enumerate(cam_ids):
            view_idx = frame_idx * num_cams + cam_idx
            frame_depths.append(depths_npz[view_idx])
            frame_images.append(images_npz[view_idx])
            frame_intrinsics.append(intrinsics_npz[view_idx])
            frame_extrinsics.append(extrinsics_npz[view_idx])
            if conf_npz is not None:
                frame_conf.append(conf_npz[view_idx])
        
        # Convert to world points
        points_world_frame, colors_frame = depths_to_world_points_with_colors_single_frame(
            frame_depths,
            frame_intrinsics,
            frame_extrinsics,
            frame_images,
            conf_list=frame_conf,
            conf_thr=0.0
        )
        
        # Render for each camera with offset trajectory
        for cam_idx, cam_id in enumerate(cam_ids):
            view_idx = frame_idx * num_cams + cam_idx
            # Compute world-to-camera with offset
            ego_to_world = ego_to_world_list[frame_idx]
            w2c_offset = compute_world_to_camera_from_waymo(
                cam_to_ego_map_offset[cam_id],
                ego_to_world,
                convert_coordinates=True,
                camera_offset=None  # Already applied to cam_to_ego
            )
            
            # Get intrinsics from npz (these are already scaled to the actual processed_images resolution)
            # This ensures correct rendering without cropping
            K = intrinsics_npz[view_idx]  # Use intrinsics from npz which match the actual resolution
            
            # Render view at actual resolution
            rgb_image, mask = render_pointcloud_view(
                points_world_frame, colors_frame, w2c_offset, K, (H_actual, W_actual)
            )
            
            rendered_images_offset.append(rgb_image)
            rendered_masks_offset.append(mask)
            extrinsics_list_offset.append(w2c_offset)
            intrinsics_list_offset.append(K)
    
    print(f"Rendered {len(rendered_images_offset)} views with offset trajectory")
    
    # Save rendered images and parameters for Step 4 (optional, for debugging)
    step3_output_dir.mkdir(parents=True, exist_ok=True)
    save_images_and_params_for_inference(
        rendered_images_offset,
        intrinsics_list_offset,
        extrinsics_list_offset,
        step3_output_dir,
        num_frames_actual,
        cam_ids,
    )
    print(f"  Note: Step 4 will use API directly, saved files are for reference only")
    
    # ============================================================================
    # Step 4: Run DA3 inference again on rendered projections with offset poses
    # ============================================================================
    print("\n" + "="*60)
    print("Step 4: Running DA3 inference on rendered projections with offset poses")
    print("="*60)
    
    extrinsics_array_offset = np.stack(extrinsics_list_offset, axis=0)
    intrinsics_array_offset = np.stack(intrinsics_list_offset, axis=0)
    
    print(f"Running inference on {len(rendered_images_offset)} rendered images...")
    
    # Run DA3 inference with chunk-based processing
    num_total = len(rendered_images_offset)
    num_chunks = (num_total + chunk_frame * num_cams - 1) // (chunk_frame * num_cams)
    
    all_depths = []
    all_confs = []
    all_skies = []
    all_processed_images = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_frame * num_cams
        end_idx = min(start_idx + chunk_frame * num_cams, num_total)
        if start_idx >= num_total:
            break
        
        chunk_images = rendered_images_offset[start_idx:end_idx]
        chunk_intrinsics = intrinsics_array_offset[start_idx:end_idx]
        chunk_extrinsics = extrinsics_array_offset[start_idx:end_idx]
        
        # Save images temporarily (DA3 API requires file paths)
        temp_dir = Path(tempfile.mkdtemp())
        chunk_image_paths = []
        for i, img_array in enumerate(chunk_images):
            img_path = temp_dir / f"temp_{i:06d}.jpg"
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), img_bgr)
            chunk_image_paths.append(str(img_path))
        
        try:
            chunk_prediction = model.inference(
                chunk_image_paths,
                export_dir=None,
                extrinsics=chunk_extrinsics,
                intrinsics=chunk_intrinsics,
                process_res=process_resolution,
                process_res_method="upper_bound_resize"
            )
            
            all_depths.append(chunk_prediction.depth)
            if chunk_prediction.conf is not None:
                all_confs.append(chunk_prediction.conf)
            if chunk_prediction.sky is not None:
                all_skies.append(chunk_prediction.sky)
            if chunk_prediction.processed_images is not None:
                all_processed_images.append(chunk_prediction.processed_images)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Merge results
    merged_depth = np.concatenate(all_depths, axis=0)
    merged_conf = np.concatenate(all_confs, axis=0) if all_confs else None
    merged_sky = np.concatenate(all_skies, axis=0) if all_skies else None
    merged_processed_images = np.concatenate(all_processed_images, axis=0) if all_processed_images else None
    
    # Create prediction object
    prediction_offset = chunk_prediction
    prediction_offset.depth = merged_depth
    prediction_offset.conf = merged_conf
    prediction_offset.sky = merged_sky
    prediction_offset.processed_images = merged_processed_images
    prediction_offset.extrinsics = extrinsics_array_offset
    prediction_offset.intrinsics = intrinsics_array_offset
    
    # Export to npz
    step4_output_dir.mkdir(parents=True, exist_ok=True)
    npz_output_path_offset = step4_output_dir / "results.npz"
    
    save_dict_offset = {
        "image": prediction_offset.processed_images,
        "depth": np.round(prediction_offset.depth, 6),
        "extrinsics": prediction_offset.extrinsics,
        "intrinsics": prediction_offset.intrinsics,
    }
    if prediction_offset.conf is not None:
        save_dict_offset["conf"] = np.round(prediction_offset.conf, 2)
    
    np.savez_compressed(str(npz_output_path_offset), **save_dict_offset)
    print(f"Step 4 completed: Saved offset point cloud to {npz_output_path_offset}")
    
    # ============================================================================
    # Step 5: Render final projections using original FRONT camera trajectory
    # ============================================================================
    print("\n" + "="*60)
    print("Step 5: Rendering final projections using original FRONT camera trajectory")
    print("="*60)
    
    # Load offset point cloud from Step 4
    npz_data_offset = load_npz_data(str(npz_output_path_offset))
    images_npz_offset = npz_data_offset['images']
    depths_npz_offset = npz_data_offset['depths']
    extrinsics_npz_offset = npz_data_offset['extrinsics']
    intrinsics_npz_offset = npz_data_offset['intrinsics']
    conf_npz_offset = npz_data_offset.get('conf', None)
    
    num_total_views_offset = depths_npz_offset.shape[0]
    num_frames_offset = num_total_views_offset // num_cams
    
    # Get original FRONT camera parameters
    # Use intrinsics from npz (actual resolution) for rendering
    # Find the first view index for render_cam_id in the offset point cloud
    render_cam_idx_in_npz = cam_ids.index(render_cam_id) if render_cam_id in cam_ids else 0
    K_render = intrinsics_npz_offset[render_cam_idx_in_npz]  # Use intrinsics from npz which match actual resolution
    cam_to_ego_render = cam_to_ego_map[render_cam_id]
    
    # Get actual resolution from offset point cloud
    H_offset, W_offset = images_npz_offset.shape[1], images_npz_offset.shape[2]
    print(f"Detected actual resolution from offset processed_images: {H_offset} x {W_offset} (H x W)")
    
    # Render using original FRONT camera trajectory
    final_images = []
    final_masks = []
    gt_images = []
    
    num_frames_to_render = min(num_frames_offset, num_frames_actual, len(ego_to_world_list))
    print(f"Rendering {num_frames_to_render} frames with original FRONT camera trajectory...")
    print(f"  Building point cloud from ALL {len(cam_ids)} cameras ({cam_ids}) for each frame")
    print(f"  Then projecting to FRONT camera (cam_id={render_cam_id}) view")
    for frame_idx in tqdm(range(num_frames_to_render), desc="Rendering final views"):
        frame_depths = []
        frame_images = []
        frame_intrinsics = []
        frame_extrinsics = []
        frame_conf = []
        
        # Collect data from ALL cameras for this frame
        for cam_idx, _ in enumerate(cam_ids):
            view_idx = frame_idx * num_cams + cam_idx
            if view_idx >= num_total_views_offset:
                raise IndexError(f"View index {view_idx} out of bounds (total: {num_total_views_offset})")
            frame_depths.append(depths_npz_offset[view_idx])
            frame_images.append(images_npz_offset[view_idx])
            frame_intrinsics.append(intrinsics_npz_offset[view_idx])
            frame_extrinsics.append(extrinsics_npz_offset[view_idx])
            if conf_npz_offset is not None:
                frame_conf.append(conf_npz_offset[view_idx])
        
        points_world_frame, colors_frame = depths_to_world_points_with_colors_single_frame(
            frame_depths,
            frame_intrinsics,
            frame_extrinsics,
            frame_images,
            conf_list=frame_conf,
            conf_thr=0.0
        )
        
        if points_world_frame.shape[0] == 0:
            raise RuntimeError(f"No points in frame {frame_idx}")
        
        ego_to_world = ego_to_world_list[frame_idx]
        w2c_render = compute_world_to_camera_from_waymo(
            cam_to_ego_render,
            ego_to_world,
            convert_coordinates=True,
            camera_offset=None
        )
        
        rgb_image, mask = render_pointcloud_view(
            points_world_frame, colors_frame, w2c_render, K_render, (H_offset, W_offset)
        )
        
        final_images.append(rgb_image)
        final_masks.append(mask)
        
        gt_path = os.path.join(images_dir, f"{frame_idx:03d}_{render_cam_id}.jpg")
        if os.path.exists(gt_path):
            gt_img = np.array(Image.open(gt_path))
            # Resize to match rendered image resolution
            if gt_img.shape[:2][::-1] != (W_offset, H_offset):  # PIL Image size is (W, H), numpy shape is (H, W, 3)
                gt_img = cv2.resize(gt_img, (W_offset, H_offset), interpolation=cv2.INTER_LINEAR)
            gt_images.append(gt_img)
        else:
            print(f"Warning: GT image not found: {gt_path}")
            gt_images.append(np.zeros((H_offset, W_offset, 3), dtype=np.uint8))
    
    print(f"Rendered {len(final_images)} final views")
    
    # ============================================================================
    # Step 6: Save final training data
    # ============================================================================
    print("\n" + "="*60)
    print("Step 6: Saving final training data")
    print("="*60)
    
    final_output_dir.mkdir(parents=True, exist_ok=True)
    occluded_images_dir = final_output_dir / "occluded_images"
    masks_dir = final_output_dir / "masks"
    gt_images_dir = final_output_dir / "gt_images"
    
    occluded_images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    gt_images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(final_images)} training samples...")
    for frame_idx in tqdm(range(len(final_images)), desc="Saving"):
        # Save occluded point cloud image
        occluded_path = occluded_images_dir / f"{frame_idx:05d}.jpg"
        occluded_bgr = cv2.cvtColor(final_images[frame_idx], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(occluded_path), occluded_bgr)
        
        # Save mask
        mask_path = masks_dir / f"{frame_idx:05d}.png"
        cv2.imwrite(str(mask_path), final_masks[frame_idx])
        
        # Save GT image
        gt_path = gt_images_dir / f"{frame_idx:05d}.jpg"
        gt_bgr = cv2.cvtColor(gt_images[frame_idx], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(gt_path), gt_bgr)
    
    # Create videos from saved images
    print("\n" + "="*60)
    print("Creating videos from saved images...")
    print("="*60)
    
    if len(final_images) > 0:
        H_video, W_video = final_images[0].shape[:2]
        fps = 10  # FPS for output videos
        
        # Create video writers
        occluded_video_path = final_output_dir / "occluded_images.mp4"
        masks_video_path = final_output_dir / "masks.mp4"
        gt_video_path = final_output_dir / "gt_images.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        occluded_writer = cv2.VideoWriter(str(occluded_video_path), fourcc, fps, (W_video, H_video))
        masks_writer = cv2.VideoWriter(str(masks_video_path), fourcc, fps, (W_video, H_video))
        gt_writer = cv2.VideoWriter(str(gt_video_path), fourcc, fps, (W_video, H_video))
        
        for frame_idx in tqdm(range(len(final_images)), desc="Writing videos"):
            occluded_bgr = cv2.cvtColor(final_images[frame_idx], cv2.COLOR_RGB2BGR)
            occluded_writer.write(occluded_bgr)
            
            mask_3ch = cv2.cvtColor(final_masks[frame_idx], cv2.COLOR_GRAY2BGR)
            masks_writer.write(mask_3ch)
            
            gt_bgr = cv2.cvtColor(gt_images[frame_idx], cv2.COLOR_RGB2BGR)
            gt_writer.write(gt_bgr)
        
        # Release video writers
        occluded_writer.release()
        masks_writer.release()
        gt_writer.release()
        
        print(f"✓ Videos created:")
        print(f"  - Occluded images: {occluded_video_path}")
        print(f"  - Masks: {masks_video_path}")
        print(f"  - GT images: {gt_video_path}")
    
    print("\n" + "="*60)
    print("Training data preparation completed!")
    print("="*60)
    print(f"Output directory: {final_output_dir}")
    print(f"  - Occluded images: {occluded_images_dir}")
    print(f"  - Masks: {masks_dir}")
    print(f"  - GT images: {gt_images_dir}")
    print(f"Total samples: {len(final_images)}")


if __name__ == "__main__":
    main()

