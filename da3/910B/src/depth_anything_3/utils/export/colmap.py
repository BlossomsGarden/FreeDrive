# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np

from PIL import Image

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .glb import _depths_to_world_points_with_colors

try:
    from scipy.spatial.transform import Rotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warn("Dependency 'scipy' not found. Using pure numpy implementation for quaternion conversion.")


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z).
    
    Uses scipy if available, otherwise falls back to pure numpy implementation.
    """
    if HAS_SCIPY:
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # Returns (x, y, z, w)
        # COLMAP uses (w, x, y, z) format
        return np.array([quat[3], quat[0], quat[1], quat[2]])
    else:
        # Pure numpy implementation
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        # COLMAP uses (w, x, y, z) format
        return np.array([w, x, y, z])


def _w2c_to_c2w(w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert world-to-camera transformation to camera-to-world.
    
    Returns:
        R_c2w: Rotation matrix (3x3)
        t_c2w: Translation vector (3,)
    """
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]
    
    # c2w = (w2c)^(-1)
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    
    return R_c2w, t_c2w


def export_to_colmap(
    prediction: Prediction,
    export_dir: str,
    image_paths: list[str],
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
) -> None:
    # 1. Data preparation
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.processed_images,
        prediction.conf,
        conf_thresh,
    )
    num_points = len(points)
    logger.info(f"Exporting to COLMAP with {num_points} points")
    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]

    # Rebuild points_xyf to match the order of points returned by _depths_to_world_points_with_colors
    # This function processes frames sequentially and concatenates valid points
    points_xyf_list = []
    point_idx_counter = 0
    frame_start_indices = {}  # Maps frame_id to start index in points array
    
    for fidx in range(num_frames):
        d = prediction.depth[fidx]
        valid = np.isfinite(d) & (d > 0)
        if prediction.conf is not None:
            valid &= prediction.conf[fidx] >= conf_thresh
        
        if not np.any(valid):
            continue
        
        # Get pixel coordinates for valid points in this frame
        y_grid, x_grid = np.indices((h, w), dtype=np.int32)
        valid_y = y_grid[valid]
        valid_x = x_grid[valid]
        valid_f = np.full(len(valid_x), fidx, dtype=np.int32)
        
        # Stack to form (x, y, frame_idx) for this frame
        frame_xyf = np.stack([valid_x, valid_y, valid_f], axis=-1)
        points_xyf_list.append(frame_xyf)
        
        frame_start_indices[fidx] = point_idx_counter
        point_idx_counter += len(valid_x)
    
    if points_xyf_list:
        points_xyf = np.concatenate(points_xyf_list, axis=0)
    else:
        points_xyf = np.zeros((0, 3), dtype=np.int32)

    # Create sparse directory structure
    sparse_dir = os.path.join(export_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    # 2. Build mapping from point indices to 3D point IDs and track point-to-image mappings
    point3d_id_map = {}  # Maps original point index to COLMAP point3D_id
    point3d_id_counter = 1
    point_to_images = {}  # Maps point3d_id to list of (image_id, point2d_idx)
    
    # Process each frame to build point-to-image mappings
    for fidx in range(num_frames):
        if fidx not in frame_start_indices:
            continue
        
        start_idx = frame_start_indices[fidx]
        # Find points in this frame
        frame_mask = points_xyf[:, 2] == fidx
        frame_point_indices = np.where(frame_mask)[0]
        
        for local_idx, global_point_idx in enumerate(frame_point_indices):
            if global_point_idx not in point3d_id_map:
                point3d_id_map[global_point_idx] = point3d_id_counter
                point_to_images[point3d_id_counter] = []
                point3d_id_counter += 1
            
            point3d_id = point3d_id_map[global_point_idx]
            point_to_images[point3d_id].append((fidx + 1, local_idx))

    # 3. Write cameras.txt
    cameras_file = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(num_frames))
        
        for fidx in range(num_frames):
            orig_w, orig_h = Image.open(image_paths[fidx]).size
            
            intrinsic = prediction.intrinsics[fidx].copy()
            if process_res_method.endswith("resize"):
                intrinsic[0, :] *= orig_w / w
                intrinsic[1, :] *= orig_h / h
            elif process_res_method == "crop":
                raise NotImplementedError("COLMAP export for crop method is not implemented")
            else:
                raise ValueError(f"Unknown process_res_method: {process_res_method}")
            
            # PINHOLE model: fx, fy, cx, cy
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            
            camera_id = fidx + 1
            f.write(f"{camera_id} PINHOLE {orig_w} {orig_h} {fx} {fy} {cx} {cy}\n")

    # 4. Write images.txt
    images_file = os.path.join(sparse_dir, "images.txt")
    with open(images_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(num_frames))
        
        for fidx in range(num_frames):
            orig_w, orig_h = Image.open(image_paths[fidx]).size
            
            # Convert w2c to c2w
            extrinsic = prediction.extrinsics[fidx]
            R_c2w, t_c2w = _w2c_to_c2w(extrinsic)
            
            # Convert rotation matrix to quaternion
            quat = _rotation_matrix_to_quaternion(R_c2w)
            
            image_id = fidx + 1
            camera_id = fidx + 1
            image_name = os.path.basename(image_paths[fidx])
            
            # Write image line
            f.write(f"{image_id} {quat[0]} {quat[1]} {quat[2]} {quat[3]} "
                   f"{t_c2w[0]} {t_c2w[1]} {t_c2w[2]} {camera_id} {image_name}\n")
            
            # Write points2D line
            if fidx not in frame_start_indices:
                f.write("\n")
                continue
            
            frame_mask = points_xyf[:, 2] == fidx
            frame_point_indices = np.where(frame_mask)[0]
            
            point2d_lines = []
            for local_idx, global_point_idx in enumerate(frame_point_indices):
                point2d = points_xyf[global_point_idx][:2].copy().astype(np.float64)
                point2d[0] *= orig_w / w
                point2d[1] *= orig_h / h
                point3d_id = point3d_id_map[global_point_idx]
                point2d_lines.append(f"{point2d[0]:.6f} {point2d[1]:.6f} {point3d_id}")
            
            if point2d_lines:
                f.write(" ".join(point2d_lines) + "\n")
            else:
                f.write("\n")

    # 5. Write points3D.txt
    points3d_file = os.path.join(sparse_dir, "points3D.txt")
    num_exported_points = len(point3d_id_map)
    with open(points3d_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}\n".format(num_exported_points))
        
        for vidx in sorted(point3d_id_map.keys()):
            point3d_id = point3d_id_map[vidx]
            pt = points[vidx]
            col = colors[vidx]
            
            # Build track list
            track_list = []
            for image_id, point2d_idx in point_to_images[point3d_id]:
                track_list.append(f"{image_id} {point2d_idx}")
            
            # ERROR is typically set to 0 for synthetic/manually created points
            error = 0.0
            
            f.write(f"{point3d_id} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                   f"{int(col[0])} {int(col[1])} {int(col[2])} {error:.6f} "
                   f"{' '.join(track_list)}\n")
    
    logger.info(f"COLMAP export completed. Files written to {sparse_dir}")


def _create_xyf(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices (fidx) for all frames.
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.int32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.int32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf
