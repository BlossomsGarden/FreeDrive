"""
从自定义数据（RGB图像、深度图、相机内外参）构造 Prediction 对象并导出 GLB 格式点云。

这个脚本允许用户不经过 model.inference，直接使用自己的数据来生成 GLB 点云。

数据组织结构（类似 infer_demo.py）:
- data_root/
    images/          (frame_id_cam_id.jpg, 例如 000_0.jpg)
    intrinsics/      (cam_id.txt 存储内参，原始分辨率)
    extrinsics/      (cam_id.txt 存储 camera->ego 4x4 矩阵)
    ego_pose/        (frame_id.txt 存储 ego->world 4x4 矩阵)
- depth_dir/
    0_depths.npz     (shape: [num_frames, H, W])
    1_depths.npz
    ...

Usage:
    python custom_data_to_glb.py     --data_root input003     --depth_dir /data/wlh/DA3/code/Depth-Anything-3-main/src/output/aligned_depths/aligned   --output_dir ./output-aligned-pcd     --cam_ids 0 1 2     --num_frames 40     --is_metric 0     --num_max_points 1000000
"""

import os
from glob import glob
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

from infer_demo import (
    load_intrinsics,
    load_camera_extrinsics,
    load_ego_poses,
    compute_world_to_camera,
)
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export


def load_depth_npz(path: str) -> np.ndarray:
    """Load depth npz file. Support both 'depths' key and default arr_0."""
    data = np.load(path)
    if "depths" in data:
        return data["depths"]
    # fallback: use first array if key name is unknown
    first_key = list(data.files)[0]
    return data[first_key]


def load_images(
    images_dir: str,
    num_frames: int,
    cam_ids: list[int],
    target_hw: Tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """
    Load RGB images in frame-major then camera order.
    
    Order: frame 0 cam_ids[0], frame 0 cam_ids[1], ..., frame 0 cam_ids[-1],
           frame 1 cam_ids[0], ..., frame num_frames-1 cam_ids[-1]
    
    This order MUST match the order used in create_prediction_from_custom_data
    for depth, intrinsics, and extrinsics.
    
    Args:
        images_dir: Directory containing images named as {frame_id:03d}_{cam_id}.jpg
        num_frames: Number of frames
        cam_ids: List of camera IDs (order matters!)
        target_hw: Optional (H, W) to resize images. If None, uses original size.
    
    Returns:
        List of RGB images, each (H, W, 3), uint8, 0-255
        Length: num_frames * len(cam_ids)
    """
    images = []
    for frame_id in range(num_frames):
        for cam_id in cam_ids:  # Iterate in cam_ids order
            fname = f"{frame_id:03d}_{cam_id}.jpg"
            fpath = os.path.join(images_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Missing image {fname} for frame {frame_id}, cam {cam_id}"
                )
            
            with Image.open(fpath) as im:
                im = im.convert("RGB")
                if target_hw is not None:
                    H, W = target_hw
                    if im.size != (W, H):
                        im = im.resize((W, H), Image.BILINEAR)
                images.append(np.array(im, dtype=np.uint8))
    
    return images


def load_depths_from_folder(
    depth_dir: str,
    cam_ids: list[int],
    num_frames: int | None = None,
) -> Tuple[list[np.ndarray], Tuple[int, int]]:
    """
    Load depth npz files from folder, one per camera.
    
    IMPORTANT: The order of returned depth_per_cam list matches the order of cam_ids.
    So depth_per_cam[0] corresponds to cam_ids[0], depth_per_cam[1] corresponds to cam_ids[1], etc.
    
    Args:
        depth_dir: Directory containing {cam_id}_depths.npz files
        cam_ids: List of camera IDs (order matters! The returned list will match this order)
        num_frames: Optional number of frames to load. If None, uses all frames.
    
    Returns:
        Tuple of (list of depth arrays per camera, (H, W) shape)
        The list order matches cam_ids order: depth_per_cam[i] = depth from {cam_ids[i]}_depths.npz
    """
    depth_per_cam = []
    for idx, cam_id in enumerate(cam_ids):
        depth_path = os.path.join(depth_dir, f"{cam_id}_depths.npz")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Missing depth file: {depth_path}")
        
        depths = load_depth_npz(depth_path)  # (num_frames, H, W)
        if num_frames is not None:
            if depths.shape[0] < num_frames:
                raise ValueError(
                    f"Depth file {depth_path} has only {depths.shape[0]} frames, "
                    f"but {num_frames} frames requested"
                )
            depths = depths[:num_frames]
        
        depth_per_cam.append(depths)
        # Verify: depth_per_cam[idx] should correspond to cam_ids[idx]
        assert len(depth_per_cam) == idx + 1, "Depth list length mismatch"
    
    # Get shape from first camera
    _, H, W = depth_per_cam[0].shape
    
    # Sanity check: all cameras should have same shape
    for idx, depths in enumerate(depth_per_cam):
        if depths.shape[1:] != (H, W):
            raise ValueError(
                f"Depth shape mismatch: camera {cam_ids[idx]} has shape {depths.shape[1:]}, "
                f"expected {(H, W)}"
            )
    
    return depth_per_cam, (H, W)


def create_prediction_from_custom_data(
    data_root: str,
    depth_dir: str,
    cam_ids: list[int],
    num_frames: int,
    orig_size: Tuple[int, int] = (1920, 1280),
    is_metric: int = 0,
    confidence: list[np.ndarray] | None = None,
) -> Prediction:
    """
    从自定义数据构造 Prediction 对象。
    
    Args:
        data_root: 根目录，包含 images/, intrinsics/, extrinsics/, ego_pose/
        depth_dir: 深度 npz 文件所在目录
        cam_ids: 相机ID列表
        num_frames: 帧数
        orig_size: 原始图像分辨率 (W, H)，用于缩放内参
        is_metric: 是否为 metric depth (0=否, 1=是)
        confidence: 可选的置信度图列表，每个 (H, W)。如果不提供，会创建全1数组
    
    Returns:
        Prediction 对象
    """
    # 1. Load depths from npz files
    print(f"Loading depths from {depth_dir}...")
    print(f"  Camera IDs (order matters!): {cam_ids}")
    depth_per_cam, (H, W) = load_depths_from_folder(depth_dir, cam_ids, num_frames)
    target_size = (W, H)
    print(f"  Depth shape: {num_frames} frames, {len(cam_ids)} cameras, resolution {W}x{H}")
    # Verify depth file order matches cam_ids order
    for idx, cam_id in enumerate(cam_ids):
        print(f"    depth_per_cam[{idx}] = depth from {cam_id}_depths.npz")
    
    # 2. Load images, intrinsics, extrinsics (similar to infer_demo.py)
    images_dir = os.path.join(data_root, "images")
    intrinsics_dir = os.path.join(data_root, "intrinsics")
    extrinsics_dir = os.path.join(data_root, "extrinsics")
    ego_pose_dir = os.path.join(data_root, "ego_pose")
    
    print(f"Loading images from {images_dir}...")
    # Load images in frame-major then camera order (matching cam_ids order)
    rgb_images = load_images(images_dir, num_frames, cam_ids, target_hw=(H, W))
    print(f"  Loaded {len(rgb_images)} images (expected {num_frames * len(cam_ids)})")
    print(f"  Image order: frame 0->{num_frames-1}, cameras {cam_ids}")
    
    print(f"Loading intrinsics from {intrinsics_dir}...")
    intrinsics_per_cam = load_intrinsics(
        intrinsics_dir, cam_ids, orig_size, target_size
    )
    # Verify intrinsics order matches cam_ids order
    for idx, cam_id in enumerate(cam_ids):
        print(f"    intrinsics_per_cam[{idx}] = intrinsics from {cam_id}.txt")
    
    print(f"Loading extrinsics from {extrinsics_dir}...")
    cam_to_ego_map = load_camera_extrinsics(extrinsics_dir, cam_ids)
    # Verify extrinsics keys match cam_ids
    for cam_id in cam_ids:
        if cam_id not in cam_to_ego_map:
            raise ValueError(f"Missing extrinsics for camera {cam_id}")
        print(f"    cam_to_ego_map[{cam_id}] = extrinsics from {cam_id}.txt")
    
    print(f"Loading ego poses from {ego_pose_dir}...")
    ego_to_world_list = load_ego_poses(ego_pose_dir, num_frames)
    if len(ego_to_world_list) != num_frames:
        raise ValueError(
            f"Ego pose count mismatch: got {len(ego_to_world_list)} poses, "
            f"expected {num_frames} frames"
        )
    
    # 3. Flatten to frame-major then camera order (matching depth and image order)
    # IMPORTANT: The order must be exactly the same for all arrays:
    #   frame 0, cam_ids[0]
    #   frame 0, cam_ids[1]
    #   ...
    #   frame 0, cam_ids[-1]
    #   frame 1, cam_ids[0]
    #   ...
    #   frame num_frames-1, cam_ids[-1]
    #
    # Both load_images() and this loop iterate in the SAME order:
    #   for frame_id in range(num_frames):
    #       for cam_id in cam_ids:  (or for cam_idx, cam_id in enumerate(cam_ids))
    depth_list = []
    intrinsics_list = []
    extrinsics_list = []
    image_list = []
    
    # Verify that rgb_images were loaded in the correct order
    expected_image_count = num_frames * len(cam_ids)
    if len(rgb_images) != expected_image_count:
        raise ValueError(
            f"Image count mismatch: got {len(rgb_images)} images, "
            f"expected {expected_image_count} (num_frames={num_frames}, num_cams={len(cam_ids)})"
        )
    
    # Build aligned arrays: frame-major, then camera order (matching cam_ids order)
    # This loop order MUST match load_images() loop order exactly
    print(f"\nBuilding aligned arrays (order: frame-major, then cameras {cam_ids})...")
    image_idx = 0  # Track position in rgb_images list
    for frame_id in range(num_frames):
        ego_to_world = ego_to_world_list[frame_id]
        for cam_idx, cam_id in enumerate(cam_ids):
            # CRITICAL: Verify we're using the correct cam_id at this position
            assert cam_id == cam_ids[cam_idx], f"Camera ID mismatch: cam_id={cam_id} != cam_ids[{cam_idx}]={cam_ids[cam_idx]}"
            
            # Get depth for this frame and camera
            # depth_per_cam[cam_idx] corresponds to cam_ids[cam_idx] (i.e., cam_id)
            # depth_per_cam[cam_idx] has shape (num_frames, H, W)
            depth_frame_cam = depth_per_cam[cam_idx][frame_id]  # (H, W)
            depth_list.append(depth_frame_cam)
            
            # Get image for this frame and camera
            # rgb_images were loaded in the same order: frame-major, then cam_ids order
            # So image_idx should match current position
            image_list.append(rgb_images[image_idx])  # (H, W, 3)
            image_idx += 1
            
            # Compute world-to-camera transform for this frame and camera
            # Use cam_id (actual camera ID) to lookup extrinsics
            w2c = compute_world_to_camera(cam_to_ego_map[cam_id], ego_to_world)
            extrinsics_list.append(w2c.astype(np.float32))
            
            # Get intrinsics for this camera (same intrinsics for all frames of same camera)
            # intrinsics_per_cam[cam_idx] corresponds to cam_ids[cam_idx] (i.e., cam_id)
            intrinsics_list.append(intrinsics_per_cam[cam_idx])
            
            # Debug output for first few entries
            if frame_id == 0 and cam_idx < 3:
                print(f"  Frame {frame_id}, cam_idx={cam_idx}, cam_id={cam_id}: "
                      f"depth_per_cam[{cam_idx}][{frame_id}], "
                      f"image_idx={image_idx-1}, "
                      f"intrinsics_per_cam[{cam_idx}], "
                      f"extrinsics from cam_to_ego_map[{cam_id}]")
    
    # Verify we consumed all images
    assert image_idx == len(rgb_images), f"Image index mismatch: used {image_idx}, but {len(rgb_images)} images available"
    
    # 4. Verify alignment before stacking
    N = len(depth_list)
    assert len(image_list) == N, f"Image list length ({len(image_list)}) != depth list length ({N})"
    assert len(intrinsics_list) == N, f"Intrinsics list length ({len(intrinsics_list)}) != depth list length ({N})"
    assert len(extrinsics_list) == N, f"Extrinsics list length ({len(extrinsics_list)}) != depth list length ({N})"
    
    # Stack into arrays
    depth_array = np.stack(depth_list, axis=0).astype(np.float32)  # (N, H, W)
    processed_images = np.stack(image_list, axis=0)  # (N, H, W, 3), already uint8
    intrinsics_array = np.stack(intrinsics_list, axis=0).astype(np.float32)  # (N, 3, 3)
    
    # Ensure extrinsics are (N, 4, 4)
    extrinsics_44 = []
    for ext in extrinsics_list:
        if ext.shape == (4, 4):
            extrinsics_44.append(ext)
        elif ext.shape == (3, 4):
            ext_44 = np.eye(4, dtype=ext.dtype)
            ext_44[:3, :4] = ext
            extrinsics_44.append(ext_44)
        else:
            raise ValueError(f"extrinsics must be (4,4) or (3,4), got {ext.shape}")
    extrinsics_array = np.stack(extrinsics_44, axis=0).astype(np.float32)  # (N, 4, 4)
    
    # 5. Create confidence maps (if not provided)
    if confidence is None:
        conf_array = np.ones_like(depth_array, dtype=np.float32)
    else:
        if len(confidence) != len(depth_list):
            raise ValueError(
                f"Confidence list length ({len(confidence)}) does not match "
                f"depth list length ({len(depth_list)})"
            )
        conf_array = np.stack(confidence, axis=0).astype(np.float32)
    
    # 6. Final verification: shapes and alignment
    N = len(depth_list)
    assert depth_array.shape == (N, H, W), f"Depth shape mismatch: {depth_array.shape} != {(N, H, W)}"
    assert processed_images.shape == (N, H, W, 3), f"Image shape mismatch: {processed_images.shape} != {(N, H, W, 3)}"
    assert intrinsics_array.shape == (N, 3, 3), f"Intrinsics shape mismatch: {intrinsics_array.shape} != {(N, 3, 3)}"
    assert extrinsics_array.shape == (N, 4, 4), f"Extrinsics shape mismatch: {extrinsics_array.shape} != {(N, 4, 4)}"
    assert conf_array.shape == (N, H, W), f"Confidence shape mismatch: {conf_array.shape} != {(N, H, W)}"
    
    # Verify expected count
    expected_count = num_frames * len(cam_ids)
    assert N == expected_count, f"Total views ({N}) != expected ({expected_count}) = {num_frames} frames × {len(cam_ids)} cameras"
    
    # 7. Final verification: ensure alignment at a sample position
    print(f"\n✓ All arrays aligned: {N} views ({num_frames} frames × {len(cam_ids)} cameras)")
    print(f"  Array shapes: depth={depth_array.shape}, images={processed_images.shape}, "
          f"intrinsics={intrinsics_array.shape}, extrinsics={extrinsics_array.shape}")
    
    # Verify alignment for first frame to help debug
    print(f"\nAlignment verification (first frame, cameras in order {cam_ids}):")
    for cam_idx, cam_id in enumerate(cam_ids):
        view_idx = 0 * len(cam_ids) + cam_idx  # First frame (frame 0)
        print(f"  View index {view_idx}: frame=0, cam_idx={cam_idx}, cam_id={cam_id}")
        print(f"    - depth[{view_idx}] from depth_per_cam[{cam_idx}][0] (file: {cam_id}_depths.npz)")
        print(f"    - image[{view_idx}] from image file: 000_{cam_id}.jpg")
        print(f"    - intrinsics[{view_idx}] from intrinsics_per_cam[{cam_idx}] (file: {cam_id}.txt)")
        print(f"    - extrinsics[{view_idx}] from cam_to_ego_map[{cam_id}] (file: {cam_id}.txt)")
    
    print(f"\nConstructing Prediction object...")
    
    # 7. Create Prediction object
    prediction = Prediction(
        depth=depth_array,
        is_metric=is_metric,
        conf=conf_array,
        extrinsics=extrinsics_array,
        intrinsics=intrinsics_array,
        processed_images=processed_images,
        sky=None,
        gaussians=None,
        aux=None,
        scale_factor=None,
    )
    
    return prediction


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="从自定义数据（RGB图像、深度图、相机内外参）构造 Prediction 并导出 GLB"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="根目录，包含 images/, intrinsics/, extrinsics/, ego_pose/",
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        required=True,
        help="深度 npz 文件所在目录（包含 {cam_id}_depths.npz 文件）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="输出目录",
    )
    parser.add_argument(
        "--cam_ids",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="相机ID列表",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        help="帧数",
    )
    parser.add_argument(
        "--orig_w",
        type=int,
        default=1920,
        help="原始图像宽度（用于缩放内参）",
    )
    parser.add_argument(
        "--orig_h",
        type=int,
        default=1280,
        help="原始图像高度（用于缩放内参）",
    )
    parser.add_argument(
        "--is_metric",
        type=int,
        default=0,
        choices=[0, 1],
        help="是否为 metric depth (0=否, 1=是)",
    )
    parser.add_argument(
        "--num_max_points",
        type=int,
        default=1_000_000,
        help="点云下采样后的最大点数",
    )
    parser.add_argument(
        "--conf_thresh_percentile",
        type=float,
        default=40.0,
        help="置信度阈值百分位数",
    )
    parser.add_argument(
        "--show_cameras",
        action="store_true",
        default=True,
        help="在 GLB 中显示相机线框",
    )
    
    args = parser.parse_args()
    
    # Create prediction from custom data
    prediction = create_prediction_from_custom_data(
        data_root=args.data_root,
        depth_dir=args.depth_dir,
        cam_ids=args.cam_ids,
        num_frames=args.num_frames,
        orig_size=(args.orig_w, args.orig_h),
        is_metric=args.is_metric,
        confidence=None,  # Will create all-ones confidence
    )
    
    # Export to GLB
    print(f"Exporting to GLB in {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    export(
        prediction,
        export_format="glb",
        export_dir=args.output_dir,
        glb={
            "num_max_points": args.num_max_points,
            "conf_thresh_percentile": args.conf_thresh_percentile,
            "show_cameras": args.show_cameras,
        },
    )
    
    print(f"✓ GLB exported to {args.output_dir}/scene.glb")


if __name__ == "__main__":
    main()
