"""
整合的遮挡场景处理流程

Step1: 在原数据集三摄像机上推理深度（如果npz文件已存在则跳过）
Step2: 在原数据集三摄像机上project，然后在向右偏移1米后的三摄像机上渲染
Step3: 根据Step2的渲染结果，在向右偏移1米后的三摄像机上重新进行深度推理
Step4: 根据Step3的depth，先在向右偏移1米后的三摄像机上project，然后在原数据集三摄像机上渲染
Step5: 将Step4生成的图片保存为视频 occluded_images.mp4 和 occluded_masks.mp4


ASCEND_RT_VISIBLE_DEVICES=0  python occluded_pipeline_RANK0_tmp_npu.py
"""

import os
from glob import glob
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
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


# ------------------------------
# Configuration
# ------------------------------
if HAS_NPU:
    data_root_base = "/home/ma-user/modelarts/user-job-dir/wlh/data/RANK0"  # Base directory containing multiple data folders
    model_path = "/home/ma-user/modelarts/user-job-dir/wlh/model/da3nested-giant-large"
    output_base = "/home/ma-user/modelarts/user-job-dir/wlh/data/PCP_trainer/RANK0/right@1.0/"
else:
    data_root_base = "/data/wlh/FreeDrive/data/waymo/processed"  # Base directory containing multiple data folders
    model_path = "/data/wlh/DA3/model/da3nested-giant-large"
    output_base = "occluded_output"

cam_ids = [1, 0, 2]  # First three cameras


# Waymo raw camera resolution
orig_w, orig_h = 1920, 1280

# Camera offset: right 1 meter in Waymo coordinates
camera_offset = [0.0, -1.0, 0.0]  # right 1 meter

# Inference settings
num_frames = 162  # None means use all available frames
process_resolution = 1008
chunk_frame = 18

# Video output settings
fps = 10

# Video segments: [(start, end), ...] - each segment will be saved as a separate video
segments = [
    (0, 48), 
    (56, 104),
    (112, 160),  
]


# ============================================================================
# Utility Functions
# ============================================================================

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


def load_intrinsics(intrinsics_dir: str, cam_ids: list, orig_size: Tuple[int, int] = None, target_size: Tuple[int, int] = None) -> list:
    """Load camera intrinsics, optionally scaled to target size."""
    intrinsics = []
    for cam_id in cam_ids:
        intrinsics_file = os.path.join(intrinsics_dir, f"{cam_id}.txt")
        if not os.path.exists(intrinsics_file):
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_file}")
        
        vals = np.loadtxt(intrinsics_file)
        fx, fy, cx, cy = vals[:4]
        
        if target_size is not None and orig_size is not None:
            # Scale intrinsics
            ow, oh = orig_size
            tw, th = target_size
            sx, sy = tw / ow, th / oh
            fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
        
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


def load_ego_poses(ego_pose_dir: str, num_frames: Optional[int] = None):
    """Load ego->world poses (4x4)."""
    pose_files = sorted(
        glob(os.path.join(ego_pose_dir, "*.txt")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    if num_frames is not None:
        pose_files = pose_files[:num_frames]
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
    """Convert depth maps from a single frame (multiple cameras) to world coordinate points with colors."""
    if len(depth_frame_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    
    H, W = depth_frame_list[0].shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)

    pts_all, col_all = [], []

    for i in range(len(depth_frame_list)):
        d = depth_frame_list[i]
        valid = np.isfinite(d) & (d > 0)
        if conf_list is not None and conf_list[i] is not None:
            valid &= conf_list[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K_list[i])
        ext_w2c_h = _as_homogeneous44(ext_w2c_list[i])
        c2w = np.linalg.inv(ext_w2c_h)

        rays = K_inv @ pix[vidx].T
        Xc = rays * d_flat[vidx][None, :]
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)

        cols = images_u8_list[i].reshape(-1, 3)[vidx].astype(np.uint8)

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a single view of the point cloud."""
    H, W = image_size
    
    if points_world.shape[0] == 0:
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        return rgb_image, mask
    
    points_h = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
    ext_w2c_h = _as_homogeneous44(ext_w2c)
    points_cam = (ext_w2c_h @ points_h.T).T
    
    depths = points_cam[:, 2]
    valid = depths > 0
    
    if not np.any(valid):
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        return rgb_image, mask
    
    points_cam_valid = points_cam[valid, :3]
    depths_valid = depths[valid]
    colors_valid = colors[valid]
    
    points_2d = (K @ points_cam_valid.T).T
    points_2d = points_2d / (points_2d[:, 2:3] + 1e-8)
    
    uv = points_2d[:, :2]
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    
    if not np.any(in_bounds):
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        return rgb_image, mask
    
    uv_final = uv[in_bounds]
    depths_final = depths_valid[in_bounds]
    colors_final = colors_valid[in_bounds]
    
    uv_int = np.round(uv_final).astype(np.int32)
    valid_bounds = (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    if not np.any(valid_bounds):
        rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        return rgb_image, mask
    
    uv_int = uv_int[valid_bounds]
    depths_final = depths_final[valid_bounds]
    colors_final = colors_final[valid_bounds]
    
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)
    
    linear_indices = uv_int[:, 1] * W + uv_int[:, 0]
    sort_indices = np.argsort(depths_final)
    sorted_linear = linear_indices[sort_indices]
    sorted_depths = depths_final[sort_indices]
    sorted_colors = colors_final[sort_indices]
    
    reversed_linear = sorted_linear[::-1]
    reversed_depths = sorted_depths[::-1]
    reversed_colors = sorted_colors[::-1]
    
    _, unique_idx = np.unique(reversed_linear, return_index=True)
    
    final_linear = reversed_linear[unique_idx]
    final_depths = reversed_depths[unique_idx]
    final_colors = reversed_colors[unique_idx]
    
    depth_flat = depth_buffer.ravel()
    rgb_flat = rgb_image.reshape(-1, 3)
    
    depth_flat[final_linear] = final_depths
    rgb_flat[final_linear] = final_colors
    
    mask = (depth_buffer >= np.inf).astype(np.uint8) * 255
    
    return rgb_image, mask


# ============================================================================
# Step Functions
# ============================================================================

def step1_inference_on_original_cameras(
    images_dir: str,
    intrinsics_dir: str,
    extrinsics_dir: str,
    ego_pose_dir: str,
    output_npz_path: str,
    cam_ids: list,
    num_frames: Optional[int],
    process_resolution: int,
    chunk_frame: int,
    orig_w: int,
    orig_h: int,
    model: DepthAnything3,
) -> np.ndarray:
    """Step1: Inference on original cameras, save to npz if not exists."""
    print("\n" + "="*60)
    print("Step 1: Inference on Original Cameras")
    print("="*60)
    
    # Check if npz file already exists
    if os.path.exists(output_npz_path):
        print(f"✓ NPZ file already exists: {output_npz_path}")
        print("  Loading existing depth maps...")
        data = np.load(output_npz_path)
        depths = data['depth']
        print(f"  Loaded depth shape: {depths.shape}")
        return depths
    
    print(f"NPZ file not found, running inference...")
    
    # Detect target image size
    sample_image = None
    max_frame_check = num_frames if num_frames is not None else 100
    for frame_id in range(max_frame_check):
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
    
    # Load camera parameters
    intrinsics_per_cam = load_intrinsics(intrinsics_dir, cam_ids, (orig_w, orig_h), (target_w, target_h))
    cam_to_ego_map = load_camera_extrinsics(extrinsics_dir, cam_ids)
    ego_to_world_list, _ = load_ego_poses(ego_pose_dir, num_frames)
    
    # Count frames
    actual_num_frames = len(ego_to_world_list)
    if num_frames is not None:
        actual_num_frames = min(num_frames, actual_num_frames)
    else:
        num_frames = actual_num_frames
    
    extrinsics_list = []
    intrinsics_list = []
    for ego_to_world in ego_to_world_list[:num_frames]:
        for cam_id in cam_ids:
            w2c = compute_world_to_camera_from_waymo(cam_to_ego_map[cam_id], ego_to_world, camera_offset=None)
            extrinsics_list.append(w2c.astype(np.float32))
            intrinsics_list.append(intrinsics_per_cam[cam_ids.index(cam_id)])
    
    extrinsics_array = np.stack(extrinsics_list, axis=0)
    intrinsics_array = np.stack(intrinsics_list, axis=0)
    
    # Load images
    images = []
    for frame_id in range(num_frames):
        for cam_id in cam_ids:
            fpath = os.path.join(images_dir, f"{frame_id:03d}_{cam_id}.jpg")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing image {fpath}")
            images.append(fpath)
    
    # Run inference
    num_cams = len(cam_ids)
    num_chunks = (num_frames + chunk_frame - 1) // chunk_frame
    
    all_depths = []
    
    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * chunk_frame
        end_frame = min(start_frame + chunk_frame, num_frames)
        
        chunk_images = []
        for frame_idx in range(start_frame, end_frame):
            for cam_id in cam_ids:
                img_idx = frame_idx * num_cams + cam_ids.index(cam_id)
                chunk_images.append(images[img_idx])
        
        chunk_extrinsics_list = []
        chunk_intrinsics_list = []
        for frame_idx in range(start_frame, end_frame):
            for cam_id in cam_ids:
                param_idx = frame_idx * num_cams + cam_ids.index(cam_id)
                chunk_extrinsics_list.append(extrinsics_array[param_idx])
                chunk_intrinsics_list.append(intrinsics_array[param_idx])
        chunk_extrinsics = np.stack(chunk_extrinsics_list, axis=0)
        chunk_intrinsics = np.stack(chunk_intrinsics_list, axis=0)
        
        print(f"  Processing chunk {chunk_idx + 1}/{num_chunks} (frames {start_frame} to {end_frame - 1})...")
        chunk_prediction = model.inference(
            chunk_images,
            export_dir=None,
            extrinsics=chunk_extrinsics,
            intrinsics=chunk_intrinsics,
            process_res=process_resolution,
            process_res_method="upper_bound_resize"
        )
        
        all_depths.append(chunk_prediction.depth)
        print(f"  Chunk {chunk_idx + 1}/{num_chunks} completed")
    
    merged_depth = np.concatenate(all_depths, axis=0)
    
    # Interpolate to original resolution and save
    print(f"  Interpolating depth to ({orig_h}, {orig_w})...")
    depth_tensor = torch.from_numpy(merged_depth).float().unsqueeze(1)
    depth_resized = F.interpolate(depth_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
    depth_resized = depth_resized.squeeze(1).numpy().astype(merged_depth.dtype)
    
    # Save to npz
    output_dir = Path(output_npz_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz_path, depth=depth_resized)
    print(f"  ✓ Saved to {output_npz_path}")
    
    return depth_resized


def step2_render_to_offset_cameras(
    depths: np.ndarray,
    images_dir: str,
    intrinsics_dir: str,
    extrinsics_dir: str,
    ego_pose_dir: str,
    output_dir: str,
    cam_ids: list,
    camera_offset: list,
    orig_w: int,
    orig_h: int,
    fps: int,
) -> int:
    """Step2: Render to offset cameras."""
    print("\n" + "="*60)
    print("Step 2: Render to Offset Cameras")
    print("="*60)
    
    num_total_views = depths.shape[0]
    num_cams = len(cam_ids)
    num_frames = num_total_views // num_cams
    
    # Load camera parameters
    intrinsics_per_cam = load_intrinsics(intrinsics_dir, cam_ids)
    cam_to_ego_map = load_camera_extrinsics(extrinsics_dir, cam_ids)
    ego_to_world_list, pose_files = load_ego_poses(ego_pose_dir, num_frames)
    
    camera_offset_arr = np.array(camera_offset, dtype=np.float32)
    cam_to_ego_offseted_map = {
        cam_id: apply_camera_offset(cam_to_ego_map[cam_id], camera_offset_arr)
        for cam_id in cam_ids
    }
    
    K_render_map = {cam_id: intrinsics_per_cam[cam_ids.index(cam_id)] for cam_id in cam_ids}
    
    # Load images
    images_list = []
    for frame_id in range(num_frames):
        for cam_id in cam_ids:
            fpath = os.path.join(images_dir, f"{frame_id:03d}_{cam_id}.jpg")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing image {fpath}")
            with Image.open(fpath) as im:
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                images_list.append(np.array(im, dtype=np.uint8))
    images = np.stack(images_list, axis=0)
    
    H, W = orig_h, orig_w
    
    # Render for each camera
    for cam_id in cam_ids:
        cam_output_dir = os.path.join(output_dir, f"cam_{cam_id}")
        os.makedirs(cam_output_dir, exist_ok=True)
        rgb_images_dir = os.path.join(cam_output_dir, "rgb_images")
        mask_images_dir = os.path.join(cam_output_dir, "mask_images")
        os.makedirs(rgb_images_dir, exist_ok=True)
        os.makedirs(mask_images_dir, exist_ok=True)
        
        cam_to_ego_render = cam_to_ego_offseted_map[cam_id]
        K_render = K_render_map[cam_id]
        
        for frame_idx in tqdm(range(num_frames), desc=f"Rendering cam_{cam_id}"):
            frame_depths = []
            frame_images_list = []
            frame_intrinsics = []
            frame_extrinsics = []
            
            for cam_idx, cam_id_npz in enumerate(cam_ids):
                view_idx = frame_idx * num_cams + cam_idx
                frame_depths.append(depths[view_idx])
                frame_images_list.append(images[view_idx])
                frame_intrinsics.append(intrinsics_per_cam[cam_idx])
                
                cam_to_ego = cam_to_ego_map[cam_id_npz]
                ego_to_world = ego_to_world_list[frame_idx]
                world_to_camera = compute_world_to_camera_from_waymo(cam_to_ego, ego_to_world, camera_offset=None)
                frame_extrinsics.append(world_to_camera)
            
            points_world_frame, colors_frame = depths_to_world_points_with_colors_single_frame(
                frame_depths, frame_intrinsics, frame_extrinsics, frame_images_list,
                conf_list=None, conf_thr=0.0
            )
            
            if points_world_frame.shape[0] == 0:
                rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
                mask = np.zeros((H, W), dtype=np.uint8) * 255
            else:
                ego_to_world = ego_to_world_list[frame_idx]
                world_to_camera = compute_world_to_camera_from_waymo(cam_to_ego_render, ego_to_world, camera_offset=None)
                rgb_image, mask = render_pointcloud_view(points_world_frame, colors_frame, world_to_camera, K_render, (H, W))
            
            rgb_image_path = os.path.join(rgb_images_dir, f"frame_{frame_idx:05d}.jpg")
            mask_image_path = os.path.join(mask_images_dir, f"frame_{frame_idx:05d}.png")
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_image_path, rgb_bgr)
            cv2.imwrite(mask_image_path, mask)
    
    return num_frames


def step3_reinference_on_offset_cameras(
    render_output_dir: str,
    intrinsics_dir: str,
    extrinsics_dir: str,
    ego_pose_dir: str,
    cam_ids: list,
    num_frames: int,
    camera_offset: list,
    process_resolution: int,
    chunk_frame: int,
    orig_w: int,
    orig_h: int,
    model: DepthAnything3,
) -> np.ndarray:
    """Step3: Re-inference on offset cameras (returns depth array, doesn't save)."""
    print("\n" + "="*60)
    print("Step 3: Re-inference on Offset Cameras")
    print("="*60)
    
    # Load camera parameters
    cam_to_ego_original = load_camera_extrinsics(extrinsics_dir, cam_ids)
    ego_to_world_list, _ = load_ego_poses(ego_pose_dir, num_frames)
    
    camera_offset_arr = np.array(camera_offset, dtype=np.float32)
    cam_to_ego_offseted = {
        cam_id: apply_camera_offset(cam_to_ego_original[cam_id], camera_offset_arr)
        for cam_id in cam_ids
    }
    
    # Load rendered images
    rendered_images = []
    for frame_idx in range(num_frames):
        for cam_id in cam_ids:
            image_path = os.path.join(render_output_dir, f"cam_{cam_id}", "rgb_images", f"frame_{frame_idx:05d}.jpg")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing rendered image: {image_path}")
            rendered_images.append(image_path)
    
    # Get target resolution from first image
    with Image.open(rendered_images[0]) as im:
        target_w, target_h = im.size
    
    # Compute camera parameters with offset
    intrinsics_per_cam = load_intrinsics(intrinsics_dir, cam_ids, (orig_w, orig_h), (target_w, target_h))
    
    extrinsics_list = []
    intrinsics_list = []
    for ego_to_world in ego_to_world_list:
        for cam_id in cam_ids:
            w2c = compute_world_to_camera_from_waymo(cam_to_ego_offseted[cam_id], ego_to_world, camera_offset=None)
            extrinsics_list.append(w2c.astype(np.float32))
            intrinsics_list.append(intrinsics_per_cam[cam_ids.index(cam_id)])
    
    extrinsics_array = np.stack(extrinsics_list, axis=0)
    intrinsics_array = np.stack(intrinsics_list, axis=0)
    
    # Run inference
    num_cams = len(cam_ids)
    num_chunks = (num_frames + chunk_frame - 1) // chunk_frame
    
    all_depths = []
    
    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * chunk_frame
        end_frame = min(start_frame + chunk_frame, num_frames)
        
        chunk_images = []
        for frame_idx in range(start_frame, end_frame):
            for cam_id in cam_ids:
                img_idx = frame_idx * num_cams + cam_ids.index(cam_id)
                chunk_images.append(rendered_images[img_idx])
        
        chunk_extrinsics_list = []
        chunk_intrinsics_list = []
        for frame_idx in range(start_frame, end_frame):
            for cam_id in cam_ids:
                param_idx = frame_idx * num_cams + cam_ids.index(cam_id)
                chunk_extrinsics_list.append(extrinsics_array[param_idx])
                chunk_intrinsics_list.append(intrinsics_array[param_idx])
        chunk_extrinsics = np.stack(chunk_extrinsics_list, axis=0)
        chunk_intrinsics = np.stack(chunk_intrinsics_list, axis=0)
        
        print(f"  Processing chunk {chunk_idx + 1}/{num_chunks} (frames {start_frame} to {end_frame - 1})...")
        chunk_prediction = model.inference(
            chunk_images,
            export_dir=None,
            extrinsics=chunk_extrinsics,
            intrinsics=chunk_intrinsics,
            process_res=process_resolution,
            process_res_method="upper_bound_resize"
        )
        
        all_depths.append(chunk_prediction.depth)
        print(f"  Chunk {chunk_idx + 1}/{num_chunks} completed")
    
    merged_depth = np.concatenate(all_depths, axis=0)
    
    # Interpolate to original resolution
    print(f"  Interpolating depth to ({orig_h}, {orig_w})...")
    depth_tensor = torch.from_numpy(merged_depth).float().unsqueeze(1)
    depth_resized = F.interpolate(depth_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
    depth_resized = depth_resized.squeeze(1).numpy().astype(merged_depth.dtype)
    
    print(f"  Depth shape: {depth_resized.shape}")
    
    return depth_resized


def step4_reproject_with_double_project(
    depths: np.ndarray,
    render_output_dir: str,
    intrinsics_dir: str,
    extrinsics_dir: str,
    ego_pose_dir: str,
    output_dir: str,
    cam_ids: list,
    num_frames: int,
    camera_offset: list,
    orig_w: int,
    orig_h: int,
) -> None:
    """Step4: Project with offset, render with origin (only front camera)."""
    print("\n" + "="*60)
    print("Step 4: Double Project (Offset -> Origin) - Front Camera Only")
    print("="*60)
    
    # Only render to front camera (cam_id=0)
    front_cam_id = 0
    num_cams = len(cam_ids)
    
    # Load camera parameters (need all cameras for point cloud building)
    intrinsics_origin = load_intrinsics(intrinsics_dir, cam_ids)
    cam_to_ego_original = load_camera_extrinsics(extrinsics_dir, cam_ids)
    cam_to_ego_offseted = {
        cam_id: apply_camera_offset(cam_to_ego_original[cam_id], np.array(camera_offset, dtype=np.float32))
        for cam_id in cam_ids
    }
    ego_to_world_list, _ = load_ego_poses(ego_pose_dir, num_frames)
    
    # Load rendered RGB images and masks from step2 (all three cameras for point cloud building)
    rendered_images = []
    rendered_masks = []
    for frame_idx in range(num_frames):
        for cam_id in cam_ids:
            image_path = os.path.join(render_output_dir, f"cam_{cam_id}", "rgb_images", f"frame_{frame_idx:05d}.jpg")
            mask_path = os.path.join(render_output_dir, f"cam_{cam_id}", "mask_images", f"frame_{frame_idx:05d}.png")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing rendered image: {image_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing rendered mask: {mask_path}")
            with Image.open(image_path) as im:
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                rendered_images.append(np.array(im, dtype=np.uint8))
            # Load mask (white=255=invalid, black=0=valid)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"Failed to load mask: {mask_path}")
            rendered_masks.append(mask_img)
    
    H, W = orig_h, orig_w
    
    # Create output directory for front camera only
    cam_output_dir = os.path.join(output_dir, f"cam_{front_cam_id}")
    os.makedirs(cam_output_dir, exist_ok=True)
    rgb_images_dir = os.path.join(cam_output_dir, "rgb_images")
    mask_images_dir = os.path.join(cam_output_dir, "mask_images")
    os.makedirs(rgb_images_dir, exist_ok=True)
    os.makedirs(mask_images_dir, exist_ok=True)
    
    # Get front camera index
    front_cam_idx = cam_ids.index(front_cam_id)
    
    # Process each frame
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        frame_depths = []
        frame_images = []
        frame_intrinsics_offset = []
        frame_extrinsics_offset = []
        
        # Build point cloud using all three cameras' depth and RGB images
        # Filter depth values based on step2 masks (white=255=invalid, black=0=valid)
        for cam_idx, cam_id in enumerate(cam_ids):
            view_idx = frame_idx * num_cams + cam_idx
            depth_map = depths[view_idx].copy()  # Make a copy to avoid modifying original
            
            # Load corresponding mask from step2
            step2_mask = rendered_masks[view_idx]
            
            # If mask is white (255), set depth to NaN (invalid)
            # White in mask means invisible/invalid, so we shouldn't use those depths for point cloud
            invalid_mask = (step2_mask > 127)  # Threshold: >127 means white (255) or near-white
            depth_map[invalid_mask] = np.nan
            
            frame_depths.append(depth_map)
            frame_images.append(rendered_images[view_idx])
            frame_intrinsics_offset.append(intrinsics_origin[cam_idx])
            
            cam_to_ego_offset = cam_to_ego_offseted[cam_id]
            ego_to_world = ego_to_world_list[frame_idx]
            world_to_camera_offset = compute_world_to_camera_from_waymo(cam_to_ego_offset, ego_to_world, camera_offset=None)
            frame_extrinsics_offset.append(world_to_camera_offset)
        
        # Build 3D point cloud using OFFSET extrinsics (from all three cameras)
        # Depth values at invalid mask locations are now NaN and will be filtered out
        points_world_frame, colors_frame = depths_to_world_points_with_colors_single_frame(
            frame_depths, frame_intrinsics_offset, frame_extrinsics_offset, frame_images,
            conf_list=None, conf_thr=0.0
        )
        
        # Render only to front camera using ORIGIN extrinsics
        if points_world_frame.shape[0] == 0:
            rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
            mask = np.zeros((H, W), dtype=np.uint8) * 255
        else:
            ego_to_world = ego_to_world_list[frame_idx]
            cam_to_ego_origin = cam_to_ego_original[front_cam_id]
            world_to_camera_origin = compute_world_to_camera_from_waymo(cam_to_ego_origin, ego_to_world, camera_offset=None)
            K_origin = intrinsics_origin[front_cam_idx]
            rgb_image, mask = render_pointcloud_view(points_world_frame, colors_frame, world_to_camera_origin, K_origin, (H, W))
        
        rgb_image_path = os.path.join(rgb_images_dir, f"frame_{frame_idx:05d}.jpg")
        mask_image_path = os.path.join(mask_images_dir, f"frame_{frame_idx:05d}.png")
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_image_path, rgb_bgr)
        cv2.imwrite(mask_image_path, mask)


def step5_create_final_videos(
    input_dir: str,
    output_dir: str,
    front_cam_id: int,
    segments: list,
    orig_w: int,
    orig_h: int,
    fps: int,
) -> None:
    """Step5: Create final videos from step4 results (front camera only, split by segments)."""
    print("\n" + "="*60)
    print("Step 5: Create Final Videos (Front Camera Only, Split by Segments)")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    H, W = orig_h, orig_w
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for seg_idx, (start_frame, end_frame) in enumerate(segments):
        print(f"\nProcessing segment {seg_idx + 1}/{len(segments)}: frames {start_frame} to {end_frame}")
        
        # Generate video filenames: occluded_images(start-end).mp4
        rgb_video_path = os.path.join(output_dir, f"occluded_images({start_frame}-{end_frame}).mp4")
        mask_video_path = os.path.join(output_dir, f"occluded_masks({start_frame}-{end_frame}).mp4")
        
        rgb_writer = cv2.VideoWriter(rgb_video_path, fourcc, fps, (W, H))
        mask_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (W, H))
        
        for frame_idx in tqdm(range(start_frame, end_frame + 1), desc=f"Segment {seg_idx + 1}"):
            rgb_path = os.path.join(input_dir, f"cam_{front_cam_id}", "rgb_images", f"frame_{frame_idx:05d}.jpg")
            mask_path = os.path.join(input_dir, f"cam_{front_cam_id}", "mask_images", f"frame_{frame_idx:05d}.png")
            
            rgb_frame = cv2.imread(rgb_path)
            mask_frame = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if rgb_frame is None or mask_frame is None:
                raise FileNotFoundError(f"Missing frame {frame_idx} for camera {front_cam_id}")
            
            # Convert grayscale mask to BGR (3 channels) for VideoWriter compatibility
            mask_frame_bgr = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
            
            rgb_writer.write(rgb_frame)
            mask_writer.write(mask_frame_bgr)
        
        rgb_writer.release()
        mask_writer.release()
        
        print(f"  ✓ Saved: {os.path.basename(rgb_video_path)}")
        print(f"  ✓ Saved: {os.path.basename(mask_video_path)}")
    
    print(f"\n✓ All {len(segments)} segment videos saved to: {output_dir}")


# ============================================================================
# Main Pipeline
# ============================================================================

def process_single_data_root(
    data_root: str,
    output_dir: str,
    cam_ids: list,
    num_frames: Optional[int],
    process_resolution: int,
    chunk_frame: int,
    orig_w: int,
    orig_h: int,
    camera_offset: list,
    fps: int,
    segments: list,
    model: DepthAnything3,
) -> None:
    """Process a single data_root folder."""
    print("\n" + "="*80)
    print(f"Processing: {data_root}")
    print(f"Output to: {output_dir}")
    print("="*80)
    
    # Setup paths for this data_root
    images_dir = os.path.join(data_root, "images")
    intrinsics_dir = os.path.join(data_root, "intrinsics")
    extrinsics_dir = os.path.join(data_root, "extrinsics")
    ego_pose_dir = os.path.join(data_root, "ego_pose")
    
    # Setup output paths for this data_root
    step1_npz_path = os.path.join(output_dir, "known_poses_output_5cam/output/results.npz")
    step2_output_dir = os.path.join(output_dir, "rendered_views")
    step4_output_dir = os.path.join(output_dir, "reprojector_output")
    step5_output_dir = os.path.join(output_dir, "occluded_images")
    
    # Step 1: Inference on original cameras
    depths_step1 = step1_inference_on_original_cameras(
        images_dir, intrinsics_dir, extrinsics_dir, ego_pose_dir,
        step1_npz_path, cam_ids, num_frames, process_resolution,
        chunk_frame, orig_w, orig_h, model
    )
    
    # Step 2: Render to offset cameras
    num_frames_actual = step2_render_to_offset_cameras(
        depths_step1, images_dir, intrinsics_dir, extrinsics_dir, ego_pose_dir,
        step2_output_dir, cam_ids, camera_offset, orig_w, orig_h, fps
    )
    
    # Step 3: Re-inference on offset cameras (depth in memory)
    depths_step3 = step3_reinference_on_offset_cameras(
        step2_output_dir, intrinsics_dir, extrinsics_dir, ego_pose_dir,
        cam_ids, num_frames_actual, camera_offset, process_resolution,
        chunk_frame, orig_w, orig_h, model
    )
    
    # Step 4: Double project (offset -> origin)
    step4_reproject_with_double_project(
        depths_step3, step2_output_dir, intrinsics_dir, extrinsics_dir, ego_pose_dir,
        step4_output_dir, cam_ids, num_frames_actual, camera_offset, orig_w, orig_h
    )
    
    # Step 5: Create final videos (front camera only, split by segments)
    step5_create_final_videos(
        step4_output_dir, step5_output_dir, 0, segments,
        orig_w, orig_h, fps
    )
    
    print(f"\n✓ Completed processing: {data_root}")


def main():
    print("="*60)
    print("Occluded Scene Pipeline - Batch Processing")
    print("="*60)
    
    # Load model once (reused for all data folders)
    print("\nLoading model...")
    device = torch.device("cpu")
    if HAS_NPU:
        model = DepthAnything3.from_pretrained(model_path)
        device = torch.device("npu")
        print(f"Using device: NPU 910B")
    elif torch.cuda.is_available():
        model = DepthAnything3.from_pretrained(model_path)
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    else:
        raise RuntimeError("No GPU/NPU available")
    
    model = model.to(device=device)
    
    # Check if data_root_base exists
    if not os.path.exists(data_root_base):
        raise FileNotFoundError(f"Data root base directory not found: {data_root_base}")
    
    # Get all subdirectories in data_root_base
    data_folders = []
    for item in os.listdir(data_root_base):
        item_path = os.path.join(data_root_base, item)
        if os.path.isdir(item_path):
            # Check if it has the required subdirectories (images, intrinsics, etc.)
            required_dirs = ["images", "intrinsics", "extrinsics", "ego_pose"]
            if all(os.path.exists(os.path.join(item_path, req_dir)) for req_dir in required_dirs):
                data_folders.append((item, item_path))
            else:
                print(f"  Skipping {item}: missing required subdirectories")
    
    if len(data_folders) == 0:
        raise ValueError(f"No valid data folders found in {data_root_base}")
    
    print(f"\nFound {len(data_folders)} data folder(s) to process:")
    for folder_name, folder_path in data_folders:
        print(f"  - {folder_name}: {folder_path}")
    
    # Create output_base directory
    os.makedirs(output_base, exist_ok=True)
    
    # Track failed folders
    failed_folders = []
    
    # Process each data folder
    for folder_idx, (folder_name, folder_path) in enumerate(data_folders, 1):
        print(f"\n{'='*80}")
        print(f"Processing folder {folder_idx}/{len(data_folders)}: {folder_name}")
        print(f"{'='*80}")
        
        # Create output directory for this folder
        folder_output_dir = os.path.join(output_base, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)
        
        try:
            process_single_data_root(
                folder_path,  # data_root
                folder_output_dir,  # output_dir
                cam_ids,
                num_frames,
                process_resolution,
                chunk_frame,
                orig_w,
                orig_h,
                camera_offset,
                fps,
                segments,
                model
            )
        except Exception as e:
            print(f"\n✗ Error processing {folder_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"  Continuing with next folder...")
            failed_folders.append(folder_name)
            continue
    
    print("\n" + "="*80)
    print(f"✓ Batch processing completed!")
    print(f"  Total folders: {len(data_folders)}")
    print(f"  Successfully processed: {len(data_folders) - len(failed_folders)}")
    print(f"  Failed: {len(failed_folders)}")
    print(f"  Results saved to: {output_base}")
    
    if len(failed_folders) > 0:
        print(f"\n✗ Failed folders ({len(failed_folders)}):")
        for folder_name in failed_folders:
            print(f"  - {folder_name}")
    else:
        print(f"\n✓ All folders processed successfully!")
    
    print("="*80)


if __name__ == "__main__":
    main()

