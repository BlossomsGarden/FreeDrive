"""
Point Cloud Painter Prepare Pipeline

This script implements a multi-step pipeline:
Step 1: Load 3-view videos from Waymo dataset and infer depth maps (180 frames)
Step 2 & 3: Build point cloud from Step 1 depths and render to offset trajectory (right 1.5m) for 3 cameras (180 frames)
Step 4: Load rendered point cloud videos from Step 2&3 and infer depth maps again (180 frames)
Step 5: Build point cloud from Step 4 depths and render to original FRONT camera trajectory (180 frames)
        - Saves original (non-resized) images only: occluded_images/, occluded_masks/
Step 6: Apply smart resize to Step 5 results (images only, no videos): resized_occluded_images/, resized_occluded_masks/
Step 7: Save segment videos for three segments:
        - Segment 1: frames 0-48
        - Segment 2: frames 64-112
        - Segment 3: frames 128-176
        For each segment: occluded_images(start-end).mp4, occluded_masks(start-end).mp4,
                          resized_occluded_images(start-end).mp4, resized_occluded_masks(start-end).mp4

All steps 1-5 execute at 1920x1280 resolution.

conda activate vda
CUDA_VISIBLE_DEVICES=2 nohup python3 point_cloud_painter_prepare_CUDA2_RANK19.py > RANK19_left@1.0.log 2>&1 &

30个scene 180帧 约12小时啊
"""

import os
import cv2
import numpy as np
from pathlib import Path
from glob import glob
from typing import Tuple
from tqdm import tqdm
from types import SimpleNamespace
import torch
from functools import partial

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video
from render_from_npz import (
    _as_homogeneous44,
    depths_to_world_points_with_colors_single_frame,
    render_pointcloud_view,
    smart_resize_drop_black,
    compute_world_to_camera_from_waymo,
    load_intrinsics,
)


def step1_infer_original_depths(
    video_folder: str,
    cam_ids: list,
    model: VideoDepthAnything,
    render_frames: int,
    input_size: int,
    target_size: Tuple[int, int],
    device: str,
    fp32: bool,
) -> dict:
    """
    Step 1: Load 3-view videos from Waymo dataset and infer depth maps.
    
    Returns:
        Dictionary with keys:
            - depths: Dict mapping cam_id to (N, H, W) depth array
            - images: Dict mapping cam_id to (N, H, W, 3) RGB array
            - fps: FPS of videos
    """
    print("="*60)
    print("Step 1: Inferring depth maps from original Waymo videos")
    print("="*60)
    
    depths_dict = {}
    images_dict = {}
    fps_value = None
    
    for cam_id in cam_ids:
        video_path = os.path.join(video_folder, f'{cam_id}.mp4')
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Processing camera {cam_id}...")
        # Read frames at target resolution (1920x1280)
        frames, target_fps = read_video_frames(video_path, render_frames, -1, max(target_size))
        
        # Resize frames to target size if needed
        if frames.shape[1:3] != (target_size[1], target_size[0]):
            resized_frames = []
            for frame in frames:
                resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized_frame)
            frames = np.stack(resized_frames, axis=0)
        
        # Infer depth
        depths, fps = model.infer_video_depth(frames, target_fps, input_size=input_size, device=device, fp32=fp32)
        
        depths_dict[cam_id] = depths
        images_dict[cam_id] = frames
        if fps_value is None:
            fps_value = fps
        
        print(f"  Camera {cam_id}: {depths.shape[0]} frames, depth shape: {depths.shape[1:]}")
    
    return {
        'depths': depths_dict,
        'images': images_dict,
        'fps': fps_value
    }


def step2_3_render_offset_pointclouds(
    step1_data: dict,
    cam_ids: list,
    data_root: str,
    render_frames: int,
    target_size: Tuple[int, int],
    camera_offset: list,
) -> dict:
    """
    Step 2 & 3: Build point cloud from Step 1 depths and render to offset trajectory.
    
    Returns:
        Dictionary with keys:
            - rendered_images: Dict mapping cam_id to list of (H, W, 3) RGB images
            - fps: FPS of videos
    """
    print("="*60)
    print("Step 2 & 3: Rendering point clouds to offset trajectory (right 1.5m)")
    print("="*60)
    
    depths_dict = step1_data['depths']
    images_dict = step1_data['images']
    fps_value = step1_data['fps']
    
    # Load camera parameters
    extrinsics_dir = os.path.join(data_root, "extrinsics")
    intrinsics_dir = os.path.join(data_root, "intrinsics")
    ego_pose_dir = os.path.join(data_root, "ego_pose")
    
    if not os.path.exists(ego_pose_dir):
        raise FileNotFoundError(f"Ego pose directory not found: {ego_pose_dir}")
    
    # Load ego poses
    pose_files = sorted(
        glob(os.path.join(ego_pose_dir, "*.txt")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )[:render_frames]
    
    # Load intrinsics for source cameras
    orig_size = (1920, 1280)  # Original Waymo size
    K_dict = {}
    for cam_id in cam_ids:
        K_dict[cam_id] = load_intrinsics(extrinsics_dir, cam_id, orig_size, target_size)
    
    # Load camera-to-ego extrinsics
    cam_to_ego_dict = {}
    for cam_id in cam_ids:
        cam_to_ego_dict[cam_id] = np.loadtxt(os.path.join(extrinsics_dir, f"{cam_id}.txt")).reshape(4, 4)
    
    # Render frames sequentially
    H, W = target_size[1], target_size[0]  # height, width
    rendered_images_dict = {cam_id: [] for cam_id in cam_ids}
    
    print("Rendering frames...")
    for frame_idx in tqdm(range(render_frames), desc="Rendering"):
        # Load ego-to-world pose for this frame
        ego_to_world = np.loadtxt(pose_files[frame_idx]).reshape(4, 4)
        
        # Collect depth maps, images, intrinsics, and extrinsics for this frame from source cameras
        frame_depths = []
        frame_images = []
        frame_intrinsics = []
        frame_extrinsics = []
        
        for cam_id in cam_ids:
            depth = depths_dict[cam_id][frame_idx]  # (H, W)
            frame_depths.append(depth)
            
            image = images_dict[cam_id][frame_idx]  # (H, W, 3)
            frame_images.append(image)
            
            K = K_dict[cam_id]  # (3, 3)
            frame_intrinsics.append(K)
            
            # Compute world-to-camera transform (NO offset for source cameras)
            cam_to_ego = cam_to_ego_dict[cam_id]
            world_to_camera = compute_world_to_camera_from_waymo(
                cam_to_ego, ego_to_world,
                convert_coordinates=True,
                camera_offset=None  # No offset for source cameras
            )
            frame_extrinsics.append(world_to_camera)
        
        # Build 3D point cloud from this frame's depth maps
        points_world_frame, colors_frame = depths_to_world_points_with_colors_single_frame(
            frame_depths,
            frame_intrinsics,
            frame_extrinsics,
            frame_images,
            conf_list=None,
            conf_thr=0.0
        )
        
        if points_world_frame.shape[0] == 0:
            for cam_id in cam_ids:
                rendered_images_dict[cam_id].append(np.zeros((H, W, 3), dtype=np.uint8))
        else:
            # Render to each camera with offset trajectory
            for cam_id in cam_ids:
                # Compute world-to-camera transform with offset for rendering
                cam_to_ego = cam_to_ego_dict[cam_id]
                world_to_camera_render = compute_world_to_camera_from_waymo(
                    cam_to_ego, ego_to_world,
                    convert_coordinates=True,
                    camera_offset=camera_offset  # Apply offset for rendering
                )
                
                # Render view
                rgb_image, mask = render_pointcloud_view(
                    points_world_frame, colors_frame, world_to_camera_render, K_dict[cam_id], (H, W)
                )
                
                rendered_images_dict[cam_id].append(rgb_image)
    
    return {
        'rendered_images': rendered_images_dict,
        'fps': fps_value
    }


def step4_infer_rendered_depths(
    step2_3_data: dict,
    cam_ids: list,
    model: VideoDepthAnything,
    render_frames: int,
    input_size: int,
    target_size: Tuple[int, int],
    device: str,
    fp32: bool,
    output_dir: str,
) -> dict:
    """
    Step 4: Load rendered point cloud videos and infer depth maps again.
    
    Returns:
        Dictionary with keys:
            - depths: Dict mapping cam_id to (N, H, W) depth array
            - images: Dict mapping cam_id to (N, H, W, 3) RGB array
            - fps: FPS of videos
    """
    print("="*60)
    print("Step 4: Inferring depth maps from rendered point cloud videos")
    print("="*60)
    
    rendered_images_dict = step2_3_data['rendered_images']
    fps_value = step2_3_data['fps']
    
    # Save rendered images as temporary videos
    temp_video_dir = os.path.join(output_dir, "temp_rendered_videos")
    os.makedirs(temp_video_dir, exist_ok=True)
    
    depths_dict = {}
    images_dict = {}
    
    for cam_id in cam_ids:
        print(f"Processing camera {cam_id}...")
        
        # Convert list of images to numpy array
        frames = np.stack(rendered_images_dict[cam_id], axis=0)  # (N, H, W, 3)
        
        # Save as temporary video
        temp_video_path = os.path.join(temp_video_dir, f"{cam_id}_rendered.mp4")
        save_video(frames, temp_video_path, fps=fps_value)
        
        # Read back and infer depth (ensure 1920x1280 resolution)
        frames_read, _ = read_video_frames(temp_video_path, render_frames, -1, max(target_size))
        
        # Ensure frames are at target resolution (1920x1280)
        if frames_read.shape[1:3] != (target_size[1], target_size[0]):
            resized_frames = []
            for frame in frames_read:
                resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized_frame)
            frames_read = np.stack(resized_frames, axis=0)
        
        # Infer depth
        depths, fps = model.infer_video_depth(frames_read, fps_value, input_size=input_size, device=device, fp32=fp32)
        
        depths_dict[cam_id] = depths
        images_dict[cam_id] = frames_read
        
        print(f"  Camera {cam_id}: {depths.shape[0]} frames, depth shape: {depths.shape[1:]}")
    
    return {
        'depths': depths_dict,
        'images': images_dict,
        'fps': fps_value
    }


def step5_render_front_camera(
    step4_data: dict,
    cam_ids: list,
    front_cam_id: int,
    data_root: str,
    render_frames: int,
    target_size: Tuple[int, int],
    output_dir: str,
    camera_offset: list,
) -> dict:
    """
    Step 5: Build point cloud from Step 4 depths and render to original FRONT camera trajectory.
    
    When building point cloud:
        - FRONT camera uses offset trajectory (camera_offset applied)
        - Other cameras use original trajectory (no offset)
    
    When rendering:
        - FRONT camera uses original trajectory (no offset)
    
    Saves original (non-resized) images only (no videos).
    
    Args:
        camera_offset: Camera offset in Waymo coordinates [x, y, z] to apply when building point cloud for FRONT camera
    
    Returns:
        Dictionary with keys:
            - rgb_images: List of (H, W, 3) RGB images
            - masks: List of (H, W) mask images
            - fps: FPS of videos
    """
    print("="*60)
    print("Step 5: Rendering to original FRONT camera trajectory")
    print("="*60)
    
    depths_dict = step4_data['depths']
    images_dict = step4_data['images']
    fps_value = step4_data['fps']
    
    # Load camera parameters
    extrinsics_dir = os.path.join(data_root, "extrinsics")
    intrinsics_dir = os.path.join(data_root, "intrinsics")
    ego_pose_dir = os.path.join(data_root, "ego_pose")
    
    if not os.path.exists(ego_pose_dir):
        raise FileNotFoundError(f"Ego pose directory not found: {ego_pose_dir}")
    
    # Load ego poses
    pose_files = sorted(
        glob(os.path.join(ego_pose_dir, "*.txt")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )[:render_frames]
    
    # Load intrinsics
    orig_size = (1920, 1280)
    K_dict = {}
    for cam_id in cam_ids:
        K_dict[cam_id] = load_intrinsics(extrinsics_dir, cam_id, orig_size, target_size)
    
    K_front = load_intrinsics(extrinsics_dir, front_cam_id, orig_size, target_size)
    
    # Load camera-to-ego extrinsics
    cam_to_ego_dict = {}
    for cam_id in cam_ids:
        cam_to_ego_dict[cam_id] = np.loadtxt(os.path.join(extrinsics_dir, f"{cam_id}.txt")).reshape(4, 4)
    
    cam_to_ego_front = np.loadtxt(os.path.join(extrinsics_dir, f"{front_cam_id}.txt")).reshape(4, 4)
    
    # Render frames
    H, W = target_size[1], target_size[0]  # height, width
    rgb_images = []
    masks = []
    
    print("Rendering frames...")
    print(f"  Building point cloud: FRONT camera (cam_id={front_cam_id}) uses offset trajectory, other cameras use original trajectory")
    print(f"  Rendering: FRONT camera uses original trajectory (no offset)")
    
    for frame_idx in tqdm(range(render_frames), desc="Rendering"):
        # Load ego-to-world pose for this frame
        ego_to_world = np.loadtxt(pose_files[frame_idx]).reshape(4, 4)
        
        # Collect depth maps, images, intrinsics, and extrinsics for this frame from source cameras
        frame_depths = []
        frame_images = []
        frame_intrinsics = []
        frame_extrinsics = []
        
        for cam_id in cam_ids:
            depth = depths_dict[cam_id][frame_idx]  # (H, W)
            frame_depths.append(depth)
            
            image = images_dict[cam_id][frame_idx]  # (H, W, 3)
            frame_images.append(image)
            
            K = K_dict[cam_id]  # (3, 3)
            frame_intrinsics.append(K)
            
            # Compute world-to-camera transform
            # For FRONT camera: use offset trajectory for building point cloud
            # For other cameras: use original trajectory
            cam_to_ego = cam_to_ego_dict[cam_id]
            # FRONT camera: apply offset for building point cloud
            world_to_camera = compute_world_to_camera_from_waymo(
                cam_to_ego, ego_to_world,
                convert_coordinates=True,
                camera_offset=camera_offset  # Apply offset for FRONT camera
            )
            frame_extrinsics.append(world_to_camera)
        
        # Build 3D point cloud from this frame's depth maps
        points_world_frame, colors_frame = depths_to_world_points_with_colors_single_frame(
            frame_depths,
            frame_intrinsics,
            frame_extrinsics,
            frame_images,
            conf_list=None,
            conf_thr=0.0
        )
        
        if points_world_frame.shape[0] == 0:
            print(f"  Warning: No points found for frame {frame_idx}")
            rgb_images.append(np.zeros((H, W, 3), dtype=np.uint8))
            masks.append(np.zeros((H, W), dtype=np.uint8) * 255)
        else:
            # Compute world-to-camera transform for FRONT camera (original trajectory, no offset)
            world_to_camera_front = compute_world_to_camera_from_waymo(
                cam_to_ego_front, ego_to_world,
                convert_coordinates=True,
                camera_offset=None  # Original trajectory
            )
            
            # Render view
            rgb_image, mask = render_pointcloud_view(
                points_world_frame, colors_frame, world_to_camera_front, K_front, (H, W)
            )
            
            # Ensure mask is uint8 and in correct range [0, 255] before appending
            mask = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
            mask = np.clip(mask, 0, 255).astype(np.uint8)
            
            rgb_images.append(rgb_image)
            masks.append(mask)
    
    # Save original (non-resized) results
    print("Saving original (non-resized) results...")
    occluded_images_dir = os.path.join(output_dir, "occluded_images")
    occluded_masks_dir = os.path.join(output_dir, "occluded_masks")
    os.makedirs(occluded_images_dir, exist_ok=True)
    os.makedirs(occluded_masks_dir, exist_ok=True)
    
    for frame_idx in tqdm(range(len(rgb_images)), desc="Saving original images"):
        rgb_image = rgb_images[frame_idx]
        mask = masks[frame_idx]
        
        rgb_image_path = os.path.join(occluded_images_dir, f"frame_{frame_idx:05d}.jpg")
        mask_image_path = os.path.join(occluded_masks_dir, f"frame_{frame_idx:05d}.png")
        
        # Ensure mask is uint8 and in correct range [0, 255] before saving
        mask_save = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
        mask_save = np.clip(mask_save, 0, 255).astype(np.uint8)
        
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_image_path, rgb_bgr)
        cv2.imwrite(mask_image_path, mask_save)
    
    print(f"✓ Original images saved in:")
    print(f"  RGB images: {occluded_images_dir}")
    print(f"  Mask images: {occluded_masks_dir}")
    
    return {
        'rgb_images': rgb_images,
        'masks': masks,
        'fps': fps_value
    }


def step6_smart_resize(
    step5_data: dict,
    output_dir: str,
    final_size: Tuple[int, int],
) -> dict:
    """
    Step 6: Apply smart resize to Step 5 results (images only, no videos).
    
    Args:
        step5_data: Dictionary with 'rgb_images', 'masks', 'fps'
        output_dir: Output directory
        final_size: Final size (width, height) for resize
    
    Returns:
        Dictionary with keys:
            - rgb_images: Original non-resized RGB images
            - masks: Original non-resized masks
            - resized_rgb_images: Resized RGB images
            - resized_masks: Resized masks
            - fps: FPS of videos
    """
    print("="*60)
    print("Step 6: Applying smart resize to images")
    print("="*60)
    
    rgb_images = step5_data['rgb_images']
    masks = step5_data['masks']
    fps_value = step5_data['fps']
    
    # Create output directories
    rgb_images_dir = os.path.join(output_dir, "resized_occluded_images")
    mask_images_dir = os.path.join(output_dir, "resized_occluded_masks")
    os.makedirs(rgb_images_dir, exist_ok=True)
    os.makedirs(mask_images_dir, exist_ok=True)
    
    # Apply smart resize
    resized_rgb_images = []
    resized_masks = []
    
    print("Applying smart resize...")
    for frame_idx in tqdm(range(len(rgb_images)), desc="Smart resizing"):
        rgb_image = rgb_images[frame_idx]
        mask = masks[frame_idx]
        
        # Apply smart resize
        resized_rgb, resized_mask = smart_resize_drop_black(
            rgb_image, mask, final_size, black_threshold=5
        )
        
        resized_rgb_images.append(resized_rgb)
        resized_masks.append(resized_mask)
        
        # Save images
        rgb_image_path = os.path.join(rgb_images_dir, f"frame_{frame_idx:05d}.jpg")
        mask_image_path = os.path.join(mask_images_dir, f"frame_{frame_idx:05d}.png")
        
        rgb_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_image_path, rgb_bgr)
        cv2.imwrite(mask_image_path, resized_mask)
    
    print(f"✓ Smart resize images saved in:")
    print(f"  RGB images: {rgb_images_dir}")
    print(f"  Mask images: {mask_images_dir}")
    
    return {
        'rgb_images': rgb_images,  # Original non-resized images
        'masks': masks,  # Original non-resized masks
        'resized_rgb_images': resized_rgb_images,  # Resized RGB images
        'resized_masks': resized_masks,  # Resized masks
        'fps': fps_value
    }


def step7_save_segment_videos(
    step6_data: dict,
    output_dir: str,
) -> None:
    """
    Step 7: Save videos for three segments.
    
    Segments:
        - Segment 1: frames 0-48
        - Segment 2: frames 64-112
        - Segment 3: frames 128-176
    
    For each segment, saves:
        - occluded_images(start-end).mp4
        - occluded_masks(start-end).mp4
        - resized_occluded_images(start-end).mp4
        - resized_occluded_masks(start-end).mp4
    
    Args:
        step6_data: Dictionary with 'rgb_images', 'masks', 'resized_rgb_images', 'resized_masks', 'fps'
        output_dir: Output directory
    """
    print("="*60)
    print("Step 7: Saving segment videos")
    print("="*60)
    
    rgb_images = step6_data['rgb_images']
    masks = step6_data['masks']
    resized_rgb_images = step6_data['resized_rgb_images']
    resized_masks = step6_data['resized_masks']
    fps_value = step6_data['fps']
    
    # Define segments: (start_frame, end_frame)
    segments = [
        (0, 48),      # Segment 1: frames 0-48
        (64, 112),    # Segment 2: frames 64-112
        (128, 176),   # Segment 3: frames 128-176
    ]
    
    # Process masks before stacking to ensure correct format
    processed_masks = []
    for idx, mask in enumerate(masks):
        # Ensure each mask is uint8 and in correct range [0, 255]
        mask_proc = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask.copy()
        mask_proc = np.clip(mask_proc, 0, 255).astype(np.uint8)
        processed_masks.append(mask_proc)
    
    mask_frames = np.stack(processed_masks, axis=0)
    rgb_frames = np.stack(rgb_images, axis=0)
    
    # Process resized masks
    processed_resized_masks = []
    for idx, mask in enumerate(resized_masks):
        mask_proc = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask.copy()
        mask_proc = np.clip(mask_proc, 0, 255).astype(np.uint8)
        processed_resized_masks.append(mask_proc)
    
    resized_mask_frames = np.stack(processed_resized_masks, axis=0)
    resized_rgb_frames = np.stack(resized_rgb_images, axis=0)
    
    print("Creating segment videos...")
    for seg_idx, (start_frame, end_frame) in enumerate(segments, 1):
        print(f"\nProcessing Segment {seg_idx}: frames {start_frame}-{end_frame}")
        
        # Extract segment frames (end_frame is inclusive, so use end_frame+1)
        seg_rgb_frames = rgb_frames[start_frame:end_frame+1]
        seg_mask_frames = mask_frames[start_frame:end_frame+1]
        seg_resized_rgb_frames = resized_rgb_frames[start_frame:end_frame+1]
        seg_resized_mask_frames = resized_mask_frames[start_frame:end_frame+1]
        
        # Create video file names
        segment_name = f"{start_frame}-{end_frame}"
        occluded_images_video = os.path.join(output_dir, f"occluded_images({segment_name}).mp4")
        occluded_masks_video = os.path.join(output_dir, f"occluded_masks({segment_name}).mp4")
        resized_occluded_images_video = os.path.join(output_dir, f"resized_occluded_images({segment_name}).mp4")
        resized_occluded_masks_video = os.path.join(output_dir, f"resized_occluded_masks({segment_name}).mp4")
        
        # Save original RGB video
        print(f"  Saving {occluded_images_video}...")
        save_video(seg_rgb_frames, occluded_images_video, fps=fps_value)
        
        # Save original mask video (convert to RGB format)
        print(f"  Saving {occluded_masks_video}...")
        seg_mask_rgb_frames = np.stack([seg_mask_frames, seg_mask_frames, seg_mask_frames], axis=-1)
        seg_mask_rgb_frames = seg_mask_rgb_frames.astype(np.uint8)
        seg_mask_rgb_frames = np.clip(seg_mask_rgb_frames, 0, 255).astype(np.uint8)
        save_video(seg_mask_rgb_frames, occluded_masks_video, fps=fps_value)
        
        # Save resized RGB video
        print(f"  Saving {resized_occluded_images_video}...")
        save_video(seg_resized_rgb_frames, resized_occluded_images_video, fps=fps_value)
        
        # Save resized mask video (convert to RGB format)
        print(f"  Saving {resized_occluded_masks_video}...")
        seg_resized_mask_rgb_frames = np.stack([seg_resized_mask_frames, seg_resized_mask_frames, seg_resized_mask_frames], axis=-1)
        seg_resized_mask_rgb_frames = seg_resized_mask_rgb_frames.astype(np.uint8)
        seg_resized_mask_rgb_frames = np.clip(seg_resized_mask_rgb_frames, 0, 255).astype(np.uint8)
        save_video(seg_resized_mask_rgb_frames, resized_occluded_masks_video, fps=fps_value)
        
        print(f"  ✓ Segment {seg_idx} videos saved")
    
    print(f"\n✓ All segment videos saved in: {output_dir}")


def process_scene(scene_name: str, data_root: str, output_dir: str, model, device: str, render_frames: int, 
                  cam_ids: list, front_cam_id: int, encoder: str, input_size: int, metric: bool, fp32: bool,
                  target_size: tuple, final_size: tuple, camera_offset: list):
    """
    Process a single scene through the entire pipeline.
    
    Args:
        scene_name: Name of the scene (folder name)
        data_root: Root directory containing the scene data
        output_dir: Output directory for this scene
        model: Loaded VideoDepthAnything model
        device: Device string ('cuda', 'npu', or 'cpu')
        render_frames: Number of frames to process
        cam_ids: List of camera IDs
        front_cam_id: FRONT camera ID for final rendering
        encoder: Model encoder type
        input_size: Input size for inference
        metric: Whether to use metric model
        fp32: Whether to use float32 inference
        target_size: Target resolution (width, height) for steps 1-5
        final_size: Final resolution (width, height) for step 6
        camera_offset: Camera offset in Waymo coordinates
    """
    print("\n" + "="*80)
    print(f"Processing scene: {scene_name}")
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    video_folder = os.path.join(data_root, "videos")
    
    if not os.path.exists(video_folder):
        print(f"Warning: Video folder not found: {video_folder}")
        print(f"Skipping scene {scene_name}")
        return
    
    # Create output directory for this scene
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Infer depth maps from original videos
        step1_data = step1_infer_original_depths(
            video_folder=video_folder,
            cam_ids=cam_ids,
            model=model,
            render_frames=render_frames,
            input_size=input_size,
            target_size=target_size,
            device=device,
            fp32=fp32,
        )
        
        # Step 2 & 3: Render point clouds to offset trajectory
        step2_3_data = step2_3_render_offset_pointclouds(
            step1_data=step1_data,
            cam_ids=cam_ids,
            data_root=data_root,
            render_frames=render_frames,
            target_size=target_size,
            camera_offset=camera_offset,
        )
        
        # Step 4: Infer depth maps from rendered videos
        step4_data = step4_infer_rendered_depths(
            step2_3_data=step2_3_data,
            cam_ids=cam_ids,
            model=model,
            render_frames=render_frames,
            input_size=input_size,
            target_size=target_size,
            device=device,
            fp32=fp32,
            output_dir=output_dir,
        )
        
        # Step 5: Render to original FRONT camera trajectory and save original results
        step5_data = step5_render_front_camera(
            step4_data=step4_data,
            cam_ids=cam_ids,
            front_cam_id=front_cam_id,
            data_root=data_root,
            render_frames=render_frames,
            target_size=target_size,
            output_dir=output_dir,
            camera_offset=camera_offset,
        )
        
        # Step 6: Apply smart resize to images (no videos)
        step6_data = step6_smart_resize(
            step5_data=step5_data,
            output_dir=output_dir,
            final_size=final_size,
        )
        
        # Step 7: Save segment videos
        step7_save_segment_videos(
            step6_data=step6_data,
            output_dir=output_dir,
        )
        
        print("\n" + "="*80)
        print(f"✓ Successfully processed scene: {scene_name}")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error processing scene {scene_name}: {e}")
        import traceback
        traceback.print_exc()
        print(f"Skipping scene {scene_name} and continuing with next scene...")


def main():
    """Main pipeline function."""
    # ============================================================================
    # Configuration
    # ============================================================================
    render_frames = 180
    
    # Data paths - processed_data_root contains multiple scene folders
    processed_data_root = '/data/wlh/VDA/RANK19'
    output_base_root = "/data/wlh/VDA/PCP_trainer/RANK19/left@1.0"
    model_path = '/data/wlh/VDA/model/metric_video_depth_anything_vitl.pth'
    input_size = 1190
    # final_size = (1008, 672)  # Step 6 final resolution
    final_size = (1190, 792)  # must be even
    
    # Create base output directory
    os.makedirs(output_base_root, exist_ok=True)
    
    # Camera configuration
    cam_ids = [1, 0, 2]  # Source cameras for point cloud building
    front_cam_id = 0  # FRONT camera for final rendering
    
    # Model configuration
    encoder = 'vitl'
    metric = True
    fp32 = False
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Resolution configuration
    target_size = (1920, 1280)  # Steps 1-4 resolution
    
    # Camera offset (left 1.0 meters in Waymo coordinates: y=1.0)
    camera_offset = [0.0, 1.0, 0.0]  # [x=forward, y=left, z=up] in Waymo coordinates
    
    # ============================================================================
    # Find all scene directories
    # ============================================================================
    print("="*60)
    print("Finding all scene directories...")
    print("="*60)
    
    if not os.path.exists(processed_data_root):
        raise FileNotFoundError(f"Processed data root directory does not exist: {processed_data_root}")
    
    # Find all directories in processed_data_root
    scene_dirs = []
    for item in os.listdir(processed_data_root):
        item_path = os.path.join(processed_data_root, item)
        if os.path.isdir(item_path):
            # Check if it has a videos folder (to ensure it's a valid scene)
            videos_path = os.path.join(item_path, "videos")
            if os.path.exists(videos_path):
                scene_dirs.append(item)
    
    scene_dirs.sort()  # Process in alphabetical order
    
    if len(scene_dirs) == 0:
        print(f"Warning: No valid scene directories found in {processed_data_root}")
        print("A valid scene directory should contain a 'videos' folder.")
        return
    
    print(f"\nFound {len(scene_dirs)} scene(s) to process:")
    for scene_dir in scene_dirs:
        print(f"  - {scene_dir}")
    
    # ============================================================================
    # Initialize model (once for all scenes)
    # ============================================================================
    print("\n" + "="*60)
    print("Initializing VideoDepthAnything model...")
    print("="*60)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    video_depth_anything = VideoDepthAnything(**model_configs[encoder], metric=metric)
    video_depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    
    print("Model initialized successfully!")
    
    # ============================================================================
    # Process each scene
    # ============================================================================
    for scene_idx, scene_name in enumerate(scene_dirs, 1):
        data_root = os.path.join(processed_data_root, scene_name)
        output_dir = os.path.join(output_base_root, scene_name)
        
        # Check if output directory exists and is not empty (scene already processed)
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            # Check if directory is not empty
            print("output_dir_file_num: ", len(os.listdir(output_dir)))
            if len(os.listdir(output_dir)) > 10:
                print("\n" + "="*80)
                print(f"Skipping scene {scene_idx}/{len(scene_dirs)}: {scene_name}")
                print(f"Output directory exists and is not empty: {output_dir}")
                print("Scene already processed, skipping...")
                print("="*80)
                continue
        
        print("\n" + "="*80)
        print(f"Processing scene {scene_idx}/{len(scene_dirs)}: {scene_name}")
        print("="*80)
        
        try:
            process_scene(
                scene_name=scene_name,
                data_root=data_root,
                output_dir=output_dir,
                model=video_depth_anything,
                device=DEVICE,
                render_frames=render_frames,
                cam_ids=cam_ids,
                front_cam_id=front_cam_id,
                encoder=encoder,
                input_size=input_size,
                metric=metric,
                fp32=fp32,
                target_size=target_size,
                final_size=final_size,
                camera_offset=camera_offset,
            )
        except Exception as e:
            print(f"\n✗ Error processing scene {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping scene {scene_name} and continuing with next scene...")
            continue
    
    print("\n" + "="*80)
    print("All scenes processed!")
    print("="*80)
    print(f"Output base directory: {output_base_root}")
    print(f"Total scenes processed: {len(scene_dirs)}")
    print("\nSummary of outputs per scene:")
    print("  1. Original (non-resized) images: occluded_images/, occluded_masks/")
    print("  2. Resized images: resized_occluded_images/, resized_occluded_masks/")
    print("  3. Segment videos (3 segments: 0-48, 64-112, 128-176):")
    print("     - occluded_images(start-end).mp4")
    print("     - occluded_masks(start-end).mp4")
    print("     - resized_occluded_images(start-end).mp4")
    print("     - resized_occluded_masks(start-end).mp4")


if __name__ == "__main__":
    main()

