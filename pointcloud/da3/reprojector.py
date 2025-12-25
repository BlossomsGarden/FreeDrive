"""
使用offset内外参将reinfer.py生成的深度图投影为3D点云，然后用origin内外参渲染到三个摄像机视角。

流程：
1. 加载前三个摄像机（cam_ids = [1, 0, 2]）的原始内外参（origin）
2. 对这些摄像机施加向右偏移1米的偏移，得到offset后的内外参
3. 从reinfer.py生成的npz文件中加载深度图（使用offset参数生成的）
4. 从render_from_npz.py生成的rgb_images中加载RGB图像（用于给点云着色）
5. 使用offset内外参将深度图和RGB图像投影为3D点云
6. 使用origin内外参将3D点云渲染回三个摄像机视角，保存结果
"""

import os
from glob import glob
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


# ------------------------------
# Configuration
# ------------------------------
# Try to import torch_npu for NPU 910B support (not needed for this script, but keep for consistency)
try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

# Data paths
if HAS_NPU:
    data_root = "/home/ma-user/modelarts/user-job-dir/wlh/code/FreeDrive/vda/toy_data/10050"
else:
    data_root = "/data/wlh/FreeDrive/data/waymo/processed/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels"

# Camera IDs to use (first three cameras)
cam_ids = [1, 0, 2]

# Input paths
reinfer_npz_path = "reinfer_output/results.npz"  # Depth maps from reinfer.py (using offset params)
render_output_dir = "rendered_views"  # RGB images from render_from_npz.py

# Output directory
output_dir = "reprojector_output"

# Waymo raw camera resolution
orig_w, orig_h = 1920, 1280

# Camera offset: Right 1 meter in Waymo coordinates
camera_offset = [0.0, -1.0, 0.0]  # Right 1 meter

# Video output settings
fps = 10


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix."""
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def waymo_to_cv_coordinate_transform():
    """Convert Waymo vehicle coordinate system to CV coordinate system."""
    R = np.array([
        [0, -1, 0, 0],  # CV x = -Waymo y
        [0, 0, -1, 0],  # CV y = -Waymo z
        [1, 0, 0, 0],   # CV z = Waymo x
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return R


def load_intrinsics(intrinsics_dir: str, cam_ids: list) -> list:
    """Load camera intrinsics for multiple cameras (original resolution, no scaling)."""
    intrinsics = []
    for cam_id in cam_ids:
        intrinsics_file = os.path.join(intrinsics_dir, f"{cam_id}.txt")
        if not os.path.exists(intrinsics_file):
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_file}")
        
        # Load intrinsics: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        vals = np.loadtxt(intrinsics_file)
        fx, fy, cx, cy = vals[:4]
        
        # Use intrinsics directly without scaling
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        intrinsics.append(K)
    
    return intrinsics


def load_camera_extrinsics(extrinsics_dir: str, cam_ids):
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
    """Apply camera offset to cam_to_ego extrinsic matrix."""
    cam_to_ego_offseted = cam_to_ego.copy()
    cam_to_ego_offseted[0, 3] += offset[0]  # forward/backward
    cam_to_ego_offseted[1, 3] += offset[1]  # left/right
    cam_to_ego_offseted[2, 3] += offset[2]  # up/down
    return cam_to_ego_offseted


def compute_world_to_camera_from_waymo(
    cam_to_ego: np.ndarray,
    ego_to_world: np.ndarray,
    convert_coordinates: bool = True,
    camera_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute world-to-camera matrix from Waymo format."""
    # Apply camera offset BEFORE coordinate conversion
    cam_to_ego_modified = cam_to_ego.copy()
    if camera_offset is not None:
        camera_offset = np.array(camera_offset, dtype=np.float32)
        if camera_offset.shape == (3,):
            cam_to_ego_modified[0, 3] += camera_offset[0]
            cam_to_ego_modified[1, 3] += camera_offset[1]
            cam_to_ego_modified[2, 3] += camera_offset[2]
    
    if convert_coordinates:
        waymo_to_cv = waymo_to_cv_coordinate_transform()
        cv_to_waymo = np.linalg.inv(waymo_to_cv)
        
        cam_to_ego_cv = waymo_to_cv @ cam_to_ego_modified @ cv_to_waymo
        ego_to_world_cv = waymo_to_cv @ ego_to_world @ cv_to_waymo
        
        world_to_ego_cv = np.linalg.inv(ego_to_world_cv)
        ego_to_camera_cv = np.linalg.inv(cam_to_ego_cv)
        
        world_to_camera = ego_to_camera_cv @ world_to_ego_cv
    else:
        world_to_ego = np.linalg.inv(ego_to_world)
        ego_to_camera = np.linalg.inv(cam_to_ego_modified)
        world_to_camera = ego_to_camera @ world_to_ego
    
    return world_to_camera


def depths_to_world_points_with_colors_single_frame(
    depth_frame_list: list,
    K_list: list,
    ext_w2c_list: list,
    images_u8_list: list,
    conf_list: Optional[list] = None,
    conf_thr: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth maps from a single frame (multiple cameras) to world coordinate points with colors.
    
    Args:
        depth_frame_list: List of (H, W) depth maps for this frame (one per camera)
        K_list: List of (3, 3) camera intrinsics (one per camera)
        ext_w2c_list: List of (4, 4) or (3, 4) world-to-camera extrinsics (one per camera)
        images_u8_list: List of (H, W, 3) RGB images (one per camera)
        conf_list: Optional list of (H, W) confidence maps (one per camera)
        conf_thr: Confidence threshold
        
    Returns:
        points_world: (M, 3) world coordinate points
        colors: (M, 3) RGB colors
    """
    if len(depth_frame_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    
    H, W = depth_frame_list[0].shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all = [], []

    for i in range(len(depth_frame_list)):
        d = depth_frame_list[i]  # (H,W)
        valid = np.isfinite(d) & (d > 0)
        if conf_list is not None and conf_list[i] is not None:
            valid &= conf_list[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K_list[i])  # (3,3)
        ext_w2c_h = _as_homogeneous44(ext_w2c_list[i])  # (4,4)
        c2w = np.linalg.inv(ext_w2c_h)  # (4,4) camera-to-world

        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

        cols = images_u8_list[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)


def render_pointcloud_view(
    points_world: np.ndarray,
    colors: np.ndarray,
    ext_w2c: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int],
    depth_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render a single view of the point cloud.
    
    Returns:
        rgb_image: (H, W, 3) RGB image
        mask: (H, W) binary mask (0 = valid/black, 255 = missing/white)
    """
    H, W = image_size
    
    if points_world.shape[0] == 0:
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        return rgb_image, mask
    
    # Transform all points to camera coordinates
    points_h = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
    ext_w2c_h = _as_homogeneous44(ext_w2c)
    points_cam = (ext_w2c_h @ points_h.T).T  # (N, 4)
    
    # Extract depths and filter points in front of camera
    depths = points_cam[:, 2]  # (N,)
    valid = depths > 0
    
    if not np.any(valid):
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        return rgb_image, mask
    
    # Get valid points and colors
    points_cam_valid = points_cam[valid, :3]  # (M, 3)
    depths_valid = depths[valid]  # (M,)
    colors_valid = colors[valid]  # (M, 3)
    
    # Project to image plane
    points_2d = (K @ points_cam_valid.T).T  # (M, 3)
    points_2d = points_2d / (points_2d[:, 2:3] + 1e-8)  # Normalize by z
    
    # Extract pixel coordinates
    uv = points_2d[:, :2]  # (M, 2) [u, v]
    
    # Filter points within image bounds
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    
    if not np.any(in_bounds):
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        return rgb_image, mask
    
    # Get final valid data
    uv_final = uv[in_bounds]  # (K, 2)
    depths_final = depths_valid[in_bounds]  # (K,)
    colors_final = colors_valid[in_bounds]  # (K, 3)
    
    # Round to integer pixel coordinates
    uv_int = np.round(uv_final).astype(np.int32)  # (K, 2)
    
    # Final bounds check
    valid_bounds = (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    if not np.any(valid_bounds):
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        return rgb_image, mask
    
    uv_int = uv_int[valid_bounds]
    depths_final = depths_final[valid_bounds]
    colors_final = colors_final[valid_bounds]
    
    # Initialize output images
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
    
    # Vectorized Z-buffering
    linear_indices = uv_int[:, 1] * W + uv_int[:, 0]  # (K,)
    
    # Sort by depth (ascending: closest points come last)
    sort_indices = np.argsort(depths_final)
    sorted_linear = linear_indices[sort_indices]
    sorted_depths = depths_final[sort_indices]
    sorted_colors = colors_final[sort_indices]
    
    # Reverse to get last occurrence of each pixel (closest point)
    reversed_linear = sorted_linear[::-1]
    reversed_depths = sorted_depths[::-1]
    reversed_colors = sorted_colors[::-1]
    
    # Find unique pixels (keeping last occurrence = closest point)
    _, unique_idx = np.unique(reversed_linear, return_index=True)
    
    # Get final points (closest for each pixel)
    final_linear = reversed_linear[unique_idx]
    final_depths = reversed_depths[unique_idx]
    final_colors = reversed_colors[unique_idx]
    
    # Vectorized update
    depth_flat = depth_buffer.ravel()
    rgb_flat = rgb_image.reshape(-1, 3)
    
    depth_flat[final_linear] = final_depths
    rgb_flat[final_linear] = final_colors
    
    # Create mask: pixels with valid depth (0 = valid/black, 255 = missing/white)
    mask = (depth_buffer >= np.inf).astype(np.uint8) * 255
    
    return rgb_image, mask


def load_npz_depth(npz_path: str) -> np.ndarray:
    """Load only depth data from npz file."""
    data = np.load(npz_path)
    
    if 'depth' not in data:
        raise KeyError(f"Key 'depth' not found in npz file: {npz_path}")
    
    depths = data['depth']  # (N, H, W)
    
    print(f"Loaded depth from npz: {depths.shape}")
    return depths


def load_rendered_images_list(render_output_dir: str, cam_ids: list, num_frames: int) -> list:
    """Load rendered RGB images from render_from_npz.py output as numpy arrays."""
    images = []
    
    for frame_idx in range(num_frames):
        for cam_id in cam_ids:
            cam_dir = os.path.join(render_output_dir, f"cam_{cam_id}", "rgb_images")
            image_path = os.path.join(cam_dir, f"frame_{frame_idx:05d}.jpg")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing rendered image: {image_path}")
            
            # Load image
            with Image.open(image_path) as im:
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                img_array = np.array(im, dtype=np.uint8)  # (H, W, 3)
                images.append(img_array)
    
    return images


def main():
    print("="*60)
    print("Reprojector: Project with Offset, Render with Origin")
    print("="*60)
    
    # Load original camera parameters
    print("\n1. Loading original camera parameters...")
    intrinsics_dir = os.path.join(data_root, "intrinsics")
    extrinsics_dir = os.path.join(data_root, "extrinsics")
    ego_pose_dir = os.path.join(data_root, "ego_pose")
    
    # Load original intrinsics and extrinsics
    cam_to_ego_original = load_camera_extrinsics(extrinsics_dir, cam_ids)
    ego_to_world_list, pose_files = load_ego_poses(ego_pose_dir)
    intrinsics_origin = load_intrinsics(intrinsics_dir, cam_ids)
    
    print(f"  Loaded extrinsics for cameras: {cam_ids}")
    print(f"  Loaded {len(ego_to_world_list)} ego poses")
    
    # Apply camera offset to get offset extrinsics
    print(f"\n2. Applying camera offset: {camera_offset} (Waymo coordinates, Right 1m)...")
    camera_offset_arr = np.array(camera_offset, dtype=np.float32)
    cam_to_ego_offseted = {}
    for cam_id in cam_ids:
        cam_to_ego_offseted[cam_id] = apply_camera_offset(
            cam_to_ego_original[cam_id], camera_offset_arr
        )
        print(f"  Applied offset to camera {cam_id}")
    
    # Load depth maps from reinfer.py output
    print("\n3. Loading depth maps from reinfer.py output...")
    depths = load_npz_depth(reinfer_npz_path)  # (N, H, W)
    
    num_total_views = depths.shape[0]
    num_cams = len(cam_ids)
    
    if num_total_views % num_cams != 0:
        raise ValueError(
            f"Total views ({num_total_views}) is not divisible by number of cameras ({num_cams})"
        )
    
    num_frames = num_total_views // num_cams
    print(f"  Depth maps: {num_frames} frames, {num_cams} cameras per frame")
    print(f"  Total views: {num_total_views}")
    
    # Limit ego poses to match number of frames
    if len(ego_to_world_list) < num_frames:
        print(f"  Warning: Only {len(ego_to_world_list)} ego poses available, using all")
        num_frames = len(ego_to_world_list)
    else:
        ego_to_world_list = ego_to_world_list[:num_frames]
    
    # Load rendered RGB images from render_from_npz.py output
    print("\n4. Loading rendered RGB images from render_from_npz.py output...")
    rendered_images = load_rendered_images_list(render_output_dir, cam_ids, num_frames)
    print(f"  Loaded {len(rendered_images)} rendered images")
    
    if len(rendered_images) != num_total_views:
        raise ValueError(
            f"Mismatch: {len(rendered_images)} rendered images but {num_total_views} depth maps"
        )
    
    # Create output directories
    print("\n5. Setting up output directories...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for each camera
    for cam_id in cam_ids:
        cam_output_dir = os.path.join(output_dir, f"cam_{cam_id}")
        os.makedirs(cam_output_dir, exist_ok=True)
        rgb_images_dir = os.path.join(cam_output_dir, "rgb_images")
        mask_images_dir = os.path.join(cam_output_dir, "mask_images")
        os.makedirs(rgb_images_dir, exist_ok=True)
        os.makedirs(mask_images_dir, exist_ok=True)
    
    H, W = orig_h, orig_w  # height, width
    
    # Process each frame
    print("\n6. Processing frames (projecting with offset, rendering with origin)...")
    print("="*60)
    
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        # Extract depth maps, images, intrinsics, extrinsics for this frame (using OFFSET params for projection)
        frame_depths = []
        frame_images = []
        frame_intrinsics_offset = []
        frame_extrinsics_offset = []
        
        for cam_idx, cam_id in enumerate(cam_ids):
            view_idx = frame_idx * num_cams + cam_idx
            if view_idx >= num_total_views:
                raise ValueError(f"View index {view_idx} exceeds total views {num_total_views}")
            
            frame_depths.append(depths[view_idx])  # (H, W)
            frame_images.append(rendered_images[view_idx])  # (H, W, 3)
            frame_intrinsics_offset.append(intrinsics_origin[cam_idx])  # (3, 3) - same intrinsics
            
            # Compute world-to-camera transform using OFFSET extrinsics (for projection)
            cam_to_ego_offset = cam_to_ego_offseted[cam_id]
            ego_to_world = ego_to_world_list[frame_idx]
            world_to_camera_offset = compute_world_to_camera_from_waymo(
                cam_to_ego_offset, ego_to_world,
                convert_coordinates=True,
                camera_offset=None  # Already applied to cam_to_ego_offseted
            )
            frame_extrinsics_offset.append(world_to_camera_offset)  # (4, 4)
        
        # Build 3D point cloud from this frame's depth maps using OFFSET extrinsics
        points_world_frame, colors_frame = depths_to_world_points_with_colors_single_frame(
            frame_depths,
            frame_intrinsics_offset,
            frame_extrinsics_offset,
            frame_images,
            conf_list=None,
            conf_thr=0.0
        )
        
        # Render to each camera using ORIGIN extrinsics
        for cam_idx, cam_id in enumerate(cam_ids):
            if points_world_frame.shape[0] == 0:
                print(f"  Warning: No points found for frame {frame_idx}, cam {cam_id}, creating empty image")
                rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
                mask = np.zeros((H, W), dtype=np.uint8) * 255
            else:
                # Load ego-to-world pose
                ego_to_world = ego_to_world_list[frame_idx]
                
                # Compute world-to-camera transform using ORIGIN extrinsics (for rendering)
                cam_to_ego_origin = cam_to_ego_original[cam_id]
                world_to_camera_origin = compute_world_to_camera_from_waymo(
                    cam_to_ego_origin, ego_to_world,
                    convert_coordinates=True,
                    camera_offset=None  # No offset for origin
                )
                
                # Render view using ORIGIN intrinsics and extrinsics
                K_origin = intrinsics_origin[cam_idx]
                rgb_image, mask = render_pointcloud_view(
                    points_world_frame, colors_frame, world_to_camera_origin, K_origin, (H, W)
                )
            
            # Save images
            cam_output_dir = os.path.join(output_dir, f"cam_{cam_id}")
            rgb_images_dir = os.path.join(cam_output_dir, "rgb_images")
            mask_images_dir = os.path.join(cam_output_dir, "mask_images")
            
            rgb_image_path = os.path.join(rgb_images_dir, f"frame_{frame_idx:05d}.jpg")
            mask_image_path = os.path.join(mask_images_dir, f"frame_{frame_idx:05d}.png")
            
            # Convert RGB to BGR for OpenCV
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_image_path, rgb_bgr)
            cv2.imwrite(mask_image_path, mask)
    
    # Create videos from saved images
    print("\n7. Creating videos from saved images...")
    
    for cam_id in cam_ids:
        cam_output_dir = os.path.join(output_dir, f"cam_{cam_id}")
        rgb_images_dir = os.path.join(cam_output_dir, "rgb_images")
        mask_images_dir = os.path.join(cam_output_dir, "mask_images")
        
        rgb_video_path = os.path.join(cam_output_dir, "rgb_video.mp4")
        mask_video_path = os.path.join(cam_output_dir, "mask_video.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        rgb_writer = cv2.VideoWriter(rgb_video_path, fourcc, fps, (W, H))
        mask_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (W, H), isColor=False)
        
        # Load and write images in order
        for frame_idx in tqdm(range(num_frames), desc=f"Creating videos cam_{cam_id}"):
            rgb_image_path = os.path.join(rgb_images_dir, f"frame_{frame_idx:05d}.jpg")
            mask_image_path = os.path.join(mask_images_dir, f"frame_{frame_idx:05d}.png")
            
            rgb_frame = cv2.imread(rgb_image_path)
            mask_frame = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
            
            if rgb_frame is not None:
                rgb_writer.write(rgb_frame)
            if mask_frame is not None:
                mask_writer.write(mask_frame)
        
        rgb_writer.release()
        mask_writer.release()
        
        print(f"  ✓ Videos saved for camera {cam_id}:")
        print(f"    RGB: {rgb_video_path}")
        print(f"    Mask: {mask_video_path}")
    
    print("\n" + "="*60)
    print("✓ Reprojection completed!")
    print(f"✓ Output saved in: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

