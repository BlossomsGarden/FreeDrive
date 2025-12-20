"""Project LiDAR points onto a specified camera image for a single frame."""

from pathlib import Path
import sys
from typing import Iterable, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from waymo_open_dataset.dataset_pb2 import Frame
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils

# Camera name helpers
CAMERA_NAME = {
    0: "unknown",
    1: "front",
    2: "front-left",
    3: "front-right",
    4: "side-left",
    5: "side-right",
}
CAMERA_NAME_TO_ID = {
    "front": dataset_pb2.CameraName.FRONT,
    "front-left": dataset_pb2.CameraName.FRONT_LEFT,
    "front-right": dataset_pb2.CameraName.FRONT_RIGHT,
    "side-left": dataset_pb2.CameraName.SIDE_LEFT,
    "side-right": dataset_pb2.CameraName.SIDE_RIGHT,
}
ID_TO_CAMERA_NAME = {v: k for k, v in CAMERA_NAME_TO_ID.items()}

# Simple color map for bounding boxes (kept minimal)
OBJECT_COLORS = {
    1: (0, 255, 0),  # type VEHICLE
    2: (255, 0, 0),  # PEDESTRIAN
    3: (0, 0, 255),  # SIGN
    4: (255, 255, 0),  # CYCLIST
}


def rgba_func(value: float) -> tuple[float, float, float, float]:
    """Generates a color based on a range value."""
    return cast(tuple[float, float, float, float], plt.get_cmap("jet")(value / 50.0))


def plot_points_on_image(idx: int, projected_points: np.ndarray, camera_image, output_dir: Path) -> None:
    """Draw projected LiDAR points on the camera image and save."""
    image = tf.image.decode_png(camera_image.image)
    image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)

    for point in projected_points:
        x, y = int(point[0]), int(point[1])
        rgba = rgba_func(point[2])
        r, g, b = int(rgba[2] * 255.0), int(rgba[1] * 255.0), int(rgba[0] * 255.0)
        cv2.circle(image, (x, y), 1, (b, g, r), 2)

    name = f"range-image-{idx}-{CAMERA_NAME.get(camera_image.name, camera_image.name)}.png"
    cv2.imwrite(str(output_dir / name), image)


def resolve_camera_id(camera_arg: str) -> int:
    """Return the CameraName enum value from a string or integer."""
    try:
        camera_id = int(camera_arg)
        if camera_id in ID_TO_CAMERA_NAME:
            return camera_id
    except ValueError:
        pass

    key = camera_arg.lower()
    if key not in CAMERA_NAME_TO_ID:
        valid = ", ".join(sorted(CAMERA_NAME_TO_ID))
        raise ValueError(f"Unknown camera '{camera_arg}'. Use one of: {valid} or ids {sorted(ID_TO_CAMERA_NAME)}.")
    return CAMERA_NAME_TO_ID[key]


def load_frame(segment_path: Path, frame_index: int) -> Frame:
    """Load a specific frame from a TFRecord segment."""
    data_set = tf.data.TFRecordDataset(str(segment_path), compression_type="")
    for idx, data in enumerate(data_set):
        if idx == frame_index:
            frame = Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            return frame
    raise IndexError(f"Frame index {frame_index} out of range for {segment_path}.")


def compute_projected_points(points: Iterable[np.ndarray], points_cp: Iterable[np.ndarray], camera_id: int) -> np.ndarray:
    """Filter camera projections for the requested camera and concatenate xy + depth."""
    points_all = np.concatenate(list(points), axis=0)
    points_cp_all = np.concatenate(list(points_cp), axis=0)

    distances = tf.norm(points_all, axis=-1, keepdims=True)
    points_cp_tensor = tf.constant(points_cp_all, dtype=tf.int32)

    mask = tf.equal(points_cp_tensor[..., 0], camera_id)
    selected_cp = tf.cast(tf.gather_nd(points_cp_tensor, tf.where(mask)), tf.float32)
    selected_distances = tf.gather_nd(distances, tf.where(mask))

    return tf.concat([selected_cp[..., 1:3], selected_distances], -1).numpy()


def get_camera_image(frame: Frame, camera_id: int):
    for image in frame.images:
        if image.name == camera_id:
            return image
    raise ValueError(f"Camera {camera_id} not found in frame.")


def project_lidar_to_image(segment_path: Path, frame_index: int, camera_id: int, output_dir: Path, ri_index: int) -> None:
    """Load frame, build point cloud, and overlay LiDAR points on the selected camera image."""
    frame = load_frame(segment_path, frame_index)

    range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    frame.lasers.sort(key=lambda laser: laser.name)
    points, points_cp = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=ri_index
    )

    camera_image = get_camera_image(frame, camera_id)
    projected_points = compute_projected_points(points, points_cp, camera_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_points_on_image(frame_index, projected_points, camera_image, output_dir)
    print(f"Saved projection for frame {frame_index} camera {ID_TO_CAMERA_NAME.get(camera_id, camera_id)} to {output_dir}")


def main(argv: list[str]) -> None:
    # Set your inputs here:
    segment_path = Path("/data/wlh/FreeDrive/data/waymo/raw/individual_files_training_003s_segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord")
    frame_idx = 0
    camera = "front"
    # ri_index 指定使用哪一次激光回波（range image return）来生成点云并投影到图像
    # 0 for first return, 1 for second
    ri_index = 0  
    output_dir = Path("outputs")

    camera_id = resolve_camera_id(camera)
    project_lidar_to_image(segment_path, frame_idx, camera_id, output_dir, ri_index)


if __name__ == "__main__":
    main(sys.argv[1:])

