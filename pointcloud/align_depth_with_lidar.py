"""Align predicted per-frame depth maps with LiDAR by scale calibration and save visualizations."""

from pathlib import Path
import sys
from typing import Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
# pip install tensorflow==2.11.0
# import tensorflow as tf

CAMERA_NAME_TO_ID = {
    "front": 0,
    "front_left": 1,
    "front_right": 2,
    "side_left": 3,
    "side_right": 4,
}

ID_TO_CAMERA_NAME = {
    0: "front",
    1: "front_left",
    2: "front_right",
    3: "side_left",
    4: "side_right",
}



def get_camera_image(frame: any, camera_id: int):
    for image in frame.images:
        if image.name == camera_id:
            return image
    raise ValueError(f"Camera {camera_id} not found in frame.")


def compute_projected_points(points, points_cp, camera_id: int) -> np.ndarray:
    """Filter camera projections for the requested camera and concatenate xy + depth."""
    points_all = np.concatenate(list(points), axis=0)
    points_cp_all = np.concatenate(list(points_cp), axis=0)

    distances = tf.norm(points_all, axis=-1, keepdims=True)
    points_cp_tensor = tf.constant(points_cp_all, dtype=tf.int32)

    mask = tf.equal(points_cp_tensor[..., 0], camera_id)
    selected_cp = tf.cast(tf.gather_nd(points_cp_tensor, tf.where(mask)), tf.float32)
    selected_distances = tf.gather_nd(distances, tf.where(mask))

    return tf.concat([selected_cp[..., 1:3], selected_distances], -1).numpy()


def load_precomputed_projection(pre_dir: Path, frame_idx: int, camera_id: int):
    """Load precomputed projected points and image size if available."""
    npz_path = pre_dir / f"{str(frame_idx).zfill(3)}_{camera_id}.npz"
    if not npz_path.exists():
        return None, None
    data = np.load(npz_path)
    projected = data["projected"]
    hw = data["hw"]
    return projected, hw


def infer_camera_from_filename(npz_path: Path) -> str | None:
    """Infer camera name from npz filename.
    
    Supports formats like:
    - 0_depths.npz -> front
    - 1_depths.npz -> front_left
    - 2_depths.npz -> front_right
    - front_depths.npz -> front
    - front_left_depths.npz -> front_left
    """
    stem = npz_path.stem  # filename without extension
    
    # Try to extract camera ID from filename (e.g., "0_depths" -> 0)
    parts = stem.split('_')
    if len(parts) > 0:
        first_part = parts[0]
        # Check if first part is a digit (camera ID)
        if first_part.isdigit():
            camera_id = int(first_part)
            if camera_id in ID_TO_CAMERA_NAME:
                return ID_TO_CAMERA_NAME[camera_id]
        
        # Check if first part(s) match a camera name
        # Try matching progressively longer prefixes
        for i in range(1, len(parts) + 1):
            candidate = '_'.join(parts[:i])
            if candidate in CAMERA_NAME_TO_ID:
                return candidate
    
    return None


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Fast depth visualization using OpenCV colormap."""
    mask = np.isfinite(depth) & (depth > 0)
    if not np.any(mask):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmin, vmax = np.percentile(depth[mask], (1, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-3
    norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    depth_u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)


def project_lidar_front(frame: any, ri_index: int, camera_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return projected lidar points (x,y,depth) and image size for the given camera."""
    range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    frame.lasers.sort(key=lambda laser: laser.name)
    points, points_cp = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=ri_index
    )
    camera_image = get_camera_image(frame, camera_id)
    projected_points = compute_projected_points(points, points_cp, camera_id)
    image = tf.image.decode_png(camera_image.image)
    h, w, _ = image.shape
    return projected_points, (h, w)


def compute_scale(projected_points: np.ndarray, depth_map: np.ndarray) -> float:
    """Compute robust scale to align predicted depth to LiDAR depth."""
    h, w = depth_map.shape
    x = projected_points[:, 0].astype(int)
    y = projected_points[:, 1].astype(int)
    z = projected_points[:, 2]

    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(in_bounds):
        return 1.0

    x, y, z = x[in_bounds], y[in_bounds], z[in_bounds]
    pred_depth = depth_map[y, x]
    valid = np.isfinite(pred_depth) & (pred_depth > 0) & (z > 0)
    if not np.any(valid):
        return 1.0

    ratios = z[valid] / pred_depth[valid]
    scale = np.median(ratios)
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return float(scale)


def overlay_points(image_bgr: np.ndarray, projected_points: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """Fast overlay of LiDAR points on image with distance-based coloring."""
    out = image_bgr.copy()
    h, w, _ = out.shape
    xy = projected_points[:, :2].astype(int)
    z = projected_points[:, 2]

    in_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if not np.any(in_bounds):
        return out
    xy = xy[in_bounds]
    z = z[in_bounds]

    # Normalize distances based on min/max of actual LiDAR points
    z_min, z_max = np.min(z), np.max(z)
    if z_max > z_min:
        norm = (z - z_min) / (z_max - z_min)
    else:
        norm = np.zeros_like(z)
    norm = np.clip(norm, 0, 1)
    
    # Apply colormap to get colors based on distance
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Draw thicker points using circles instead of single pixels
    point_radius = 2  # Increase radius for better visibility
    for i in range(len(xy)):
        cv2.circle(out, tuple(xy[i]), point_radius, colors[i, 0, :].tolist(), -1)
    
    return out


def process(
    depth_npz_path: Path,
    output_dir: Path,
    camera_name: str,
    ri_index: int,
    file_index: int | None = None,
    total_files: int | None = None,
    data_root: Path | None = None,
    save_raw_overlay: bool = True,
    save_aligned_overlay: bool = True,
    target_resolution: Tuple[int, int] | None = None,
) -> None:
    """Align depth npz with LiDAR for each frame and save visualizations.
    
    Args:
        target_resolution: Target resolution (H, W) to resize all data to. 
                          If None, uses original depth resolution (no resize).
    """
    data = np.load(depth_npz_path)
    depths = data["depths"]  # (N, H, W)
    camera_id = CAMERA_NAME_TO_ID[camera_name]
    
    # If target_resolution is None, use original depth resolution
    if target_resolution is None:
        _, orig_h, orig_w = depths.shape
        target_resolution = (orig_h, orig_w)
    target_h, target_w = target_resolution

    pre_dir = None
    if data_root is not None:
        candidate = data_root / "lidar_align"
        if candidate.exists():
            pre_dir = candidate

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_overlay"
    overlay_dir = output_dir / "aligned_overlay"
    if save_raw_overlay:
        raw_dir.mkdir(exist_ok=True)
    if save_aligned_overlay:
        overlay_dir.mkdir(exist_ok=True)

    # Pre-allocate aligned_depths with target resolution
    num_frames = depths.shape[0]
    aligned_depths = np.empty((num_frames, target_h, target_w), dtype=np.float32)

    for idx in range(depths.shape[0]):
        projected_points = None
        h = w = None

        if pre_dir is not None:
            projected_points, hw = load_precomputed_projection(pre_dir, idx, camera_id)
            if projected_points is not None and hw is not None:
                h, w = int(hw[0]), int(hw[1])

        if projected_points is None or h is None or w is None:
            assert False, "No precomputed projection in processed/lidar_align folder!"

        depth_map = depths[idx]  # Original depth map
        
        # Resize depth map to target resolution using high-quality interpolation
        if depth_map.shape != (target_h, target_w):
            # Use INTER_CUBIC for upscaling (better quality) or INTER_AREA for downscaling
            if depth_map.shape[0] < target_h or depth_map.shape[1] < target_w:
                interpolation = cv2.INTER_CUBIC  # Better for upscaling
            else:
                interpolation = cv2.INTER_AREA  # Better for downscaling
            depth_map = cv2.resize(depth_map, (target_w, target_h), interpolation=interpolation)
        
        # Resize projected_points coordinates to match target resolution
        if (h, w) != (target_h, target_w):
            scale_x = target_w / float(w)
            scale_y = target_h / float(h)
            projected_points = projected_points.copy()
            projected_points[:, 0] = projected_points[:, 0] * scale_x  # x coordinates
            projected_points[:, 1] = projected_points[:, 1] * scale_y  # y coordinates
            # Depth values (z) remain unchanged
            h, w = target_h, target_w

        # Compute scales and align
        scale = compute_scale(projected_points, depth_map)
        aligned_depth = depth_map * scale

        def depth_range(d):
            mask = np.isfinite(d) & (d > 0)
            return (float(np.min(d[mask])), float(np.max(d[mask]))) if np.any(mask) else (0.0, 0.0)

        before_range = depth_range(depth_map)
        after_range = depth_range(aligned_depth)
        prefix = ""
        if file_index is not None and total_files is not None:
            prefix = f"[npz {file_index + 1}/{total_files}] "
        print(f"{prefix}Frame {idx}: depth range before {before_range}, after {after_range}, scale {scale:.3f}")

        aligned_depths[idx] = aligned_depth.astype(np.float32)

        # Save visualizations
        depth_bgr_raw = colorize_depth(depth_map)
        depth_bgr = colorize_depth(aligned_depth)

        if save_raw_overlay:
            cv2.imwrite(str(raw_dir / f"depth_raw_overlay_{idx:05d}.png"), overlay_points(depth_bgr_raw, projected_points, depth_map))
        if save_aligned_overlay:
            cv2.imwrite(str(overlay_dir / f"depth_aligned_overlay_{idx:05d}.png"), overlay_points(depth_bgr, projected_points, aligned_depth))

    # Save aligned depths with target resolution
    np.savez_compressed(output_dir / depth_npz_path.name, depths=aligned_depths)

def main(argv: list[str]) -> None:
    # Configure your paths and settings here
    depth_npz_folder = Path("/data/wlh/DA3/code/Depth-Anything-3-main/src/test-lidar/raw-1920x1080")  # folder containing multiple *.npz files
    output_root = Path("depth_alignment_outputs")
    data_root = Path("/data/wlh/FreeDrive/data/waymo/processed/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels")  # root where waymo_preprocess saves lidar_align
    camera = "front"
    # ri_index 指定使用哪一次激光回波（range image return）来生成点云并投影到图像
    # 0 for first return, 1 for second
    ri_index = 0  
    
    save_raw_overlay = True       # save LiDAR overlay on raw depth
    save_aligned_overlay = True   # save LiDAR overlay on aligned depth
    
    # Target resolution: 336x504 (width x height) -> (504, 336) for (H, W) format
    target_resolution = (504, 336)  # (H, W) = (height, width)

    npz_files = sorted(depth_npz_folder.glob("*.npz"))

    def _run(args):
        npz_path, i = args
        # Infer camera from filename, fallback to default if not found
        inferred_camera = infer_camera_from_filename(npz_path)
    
        # use stem (no extension) for folder name to avoid dots in path
        out_dir = output_root / npz_path.stem
        process(
            npz_path,
            out_dir,
            inferred_camera,
            ri_index,
            file_index=i,
            total_files=len(npz_files),
            data_root=data_root,
            save_raw_overlay=save_raw_overlay,
            save_aligned_overlay=save_aligned_overlay,
            target_resolution=target_resolution,
        )

    # Use one worker per npz to maximize concurrency
    with ThreadPoolExecutor(max_workers=len(npz_files)) as executor:
        executor.map(_run, zip(npz_files, range(len(npz_files))))


if __name__ == "__main__":
    main(sys.argv[1:])

