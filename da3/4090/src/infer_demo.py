import os, torch, cv2, json
import numpy as np
import re
from depth_anything_3.api import DepthAnything3

def extract_frames_from_video(video_path, output_dir, fps=1.0):
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps)) if video_fps > 0 else 1
    
    frames_dir = os.path.join(output_dir, "input_images")
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    frame_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(frames_dir, f"{saved_count:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        frame_count += 1
    
    cap.release()
    return sorted(frame_paths)

def parse_matrix_string(matrix_str):
    """Parse matrix string into 4x4 numpy array."""
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    numbers = [float(n) for n in re.findall(pattern, matrix_str)]
    if len(numbers) != 16:
        raise ValueError(f"Expected 16 numbers, got {len(numbers)}")
    return np.array(numbers).reshape(4, 4)

def load_gt_extrinsics(json_path, cam_id="cam02"):
    """Load ground truth extrinsics from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = sorted([k for k in data.keys() if k.startswith("frame")], 
                    key=lambda x: int(x.replace("frame", "")))
    
    extrinsics_list = []
    for frame_key in frames:
        if cam_id not in data[frame_key]:
            raise ValueError(f"Camera {cam_id} not found in {frame_key}")
        matrix_4x4 = parse_matrix_string(data[frame_key][cam_id])
        extrinsics_list.append(matrix_4x4[:3, :])  # Extract 3x4 part
    
    return np.array(extrinsics_list)

def compute_rotation_error(R_gt, R_pred):
    """Compute rotation error (RotErr) in radians."""
    R_rel = R_gt @ R_pred.T
    trace = np.trace(R_rel)
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
    return np.arccos(cos_theta)

def compute_translation_error(t_gt, t_pred):
    """Compute translation error (TransErr)."""
    return np.linalg.norm(t_gt - t_pred)

def evaluate_pose_estimation(
    video_path, 
    cam_id="cam02", 
    gt_json_path="camera_extrinsics.json", 
    model_dir="/data/wlh/DA3/model", 
    fps=15.0, 
    export_format="glb"
):
    """
    Evaluate camera pose estimation accuracy.
    
    Args:
        video_path: Path to input video file
        cam_id: Camera name in GT JSON file (default: "cam02")
        model_dir: Model directory path
        gt_json_path: Path to ground truth extrinsics JSON file
        fps: Sampling FPS for frame extraction
        export_format: Export format for model inference
    
    Returns:
        tuple: (RotErr_Mean in radians, TransErr_Mean)
    """
    # Find GT JSON file
    possible_paths = [
        gt_json_path,
        os.path.join(os.path.dirname(__file__), gt_json_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", gt_json_path),
    ]
    gt_path = None
    for path in possible_paths:
        if os.path.exists(path):
            gt_path = path
            break
    if gt_path is None:
        raise FileNotFoundError(f"Could not find {gt_json_path}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to(device=device)
    
    # Extract frames and run inference
    images = extract_frames_from_video(video_path, output_dir="output", fps=fps)
    prediction = model.inference(images, export_dir="output", export_format=export_format)
    
    # Load GT and compute errors
    gt_extrinsics = load_gt_extrinsics(gt_path, cam_id=cam_id)
    pred_extrinsics = prediction.extrinsics
    
    N = min(len(gt_extrinsics), len(pred_extrinsics))
    rot_errors, trans_errors = [], []
    
    for i in range(N):
        R_gt, R_pred = gt_extrinsics[i, :3, :3], pred_extrinsics[i, :3, :3]
        t_gt, t_pred = gt_extrinsics[i, :3, 3], pred_extrinsics[i, :3, 3]
        rot_errors.append(compute_rotation_error(R_gt, R_pred))
        trans_errors.append(compute_translation_error(t_gt, t_pred))
    
    rot_err_mean = np.mean(rot_errors)
    trans_err_mean = np.mean(trans_errors)
    
    return rot_err_mean, trans_err_mean

# Main execution
if __name__ == "__main__":
    video_path = "cam02.mp4"
    cam_id = "cam02"
    gt_json_path="camera_extrinsics.json"
    
    rot_err_mean, trans_err_mean = evaluate_pose_estimation(video_path, cam_id, gt_json_path)
    
    print(f"RotErr Mean: {rot_err_mean:.6f} rad")
    print(f"TransErr Mean: {trans_err_mean:.6f}")