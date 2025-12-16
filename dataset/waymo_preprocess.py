# Acknowledgement:
#   1. https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/waymo_converter.py
#   2. https://github.com/leolyj/DCA-SRSFE/blob/main/data_preprocess/Waymo/generate_flow.py

"""
conda activate wlh-py
cd /data/wlh/FreeDrive/code/datasets
CUDA_VISIBLE_DEVICES=2 python 
"""
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError("Please run 'pip install waymo-open-dataset-tf-2-11-0==1.6.1' to install the official devkit.")

import json
import os
import sys
import time
from collections.abc import Iterable
from glob import glob
from multiprocessing import Pool
from shutil import get_terminal_size

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from waymo_open_dataset.utils import box_utils, range_image_utils, transform_utils, frame_utils
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops


MOVEABLE_OBJECTS_IDS = [
    cs_pb2.CameraSegmentation.TYPE_CAR,
    cs_pb2.CameraSegmentation.TYPE_TRUCK,
    cs_pb2.CameraSegmentation.TYPE_BUS,
    cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE,
    cs_pb2.CameraSegmentation.TYPE_BICYCLE,
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE,
    cs_pb2.CameraSegmentation.TYPE_TRAILER,
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN,
    cs_pb2.CameraSegmentation.TYPE_CYCLIST,
    cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST,
    cs_pb2.CameraSegmentation.TYPE_BIRD,
    cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL,
    cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT,
]


def get_ground_np(pts):
    """
    Ground removal on a point cloud (NumPy version).
    Modified from https://github.com/tusen-ai/LiDAR_SOT/blob/main/waymo_data/data_preprocessing/ground_removal.py

    Args:
        pts (numpy.ndarray): The input point cloud.

    Returns:
        numpy.ndarray: A boolean array indicating whether each point is ground.
    """
    th_seeds_ = 1.2
    num_lpr_ = 20
    n_iter = 10
    th_dist_ = 0.3
    pts_sort = pts[pts[:, 2].argsort(), :]
    lpr = np.mean(pts_sort[:num_lpr_, 2])
    pts_g = pts_sort[pts_sort[:, 2] < lpr + th_seeds_, :]
    normal_ = np.zeros(3)
    for _ in range(n_iter):
        mean = np.mean(pts_g, axis=0)[:3]
        xx = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 0] - mean[0]))
        xy = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 1] - mean[1]))
        xz = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 2] - mean[2]))
        yy = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 1] - mean[1]))
        yz = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 2] - mean[2]))
        zz = np.mean((pts_g[:, 2] - mean[2]) * (pts_g[:, 2] - mean[2]))
        cov = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float32)
        U, _, _ = np.linalg.svd(cov)
        normal_ = U[:, 2]
        d_ = -normal_.dot(mean)
        th_dist_d_ = th_dist_ - d_
        result = pts[:, :3] @ normal_[..., np.newaxis]
        pts_g = pts[result.squeeze(-1) < th_dist_d_]
    return result < th_dist_d_


class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(
                f'[{" " * self.bar_width}] 0/{self.task_num}, ' "elapsed: 0s, ETA:"
            )
        else:
            self.file.write("completed: 0, elapsed: 0s")
        self.file.flush()
        self.start_time = time.time()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed if elapsed > 0 else float("inf")
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = (
                f"\r[{{}}] {self.completed}/{self.task_num}, "
                f"{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, "
                f"ETA: {eta:5}s"
            )

            bar_width = min(
                self.bar_width,
                int(self.terminal_width - len(msg)) + 2,
                int(self.terminal_width * 0.6),
            )
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = ">" * mark_width + " " * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f"completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,"
                f" {fps:.1f} tasks/s"
            )
        self.file.flush()


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    if initargs is None:
        return Pool(process_num, initializer)
    if not isinstance(initargs, tuple):
        raise TypeError('"initargs" must be a tuple')
    return Pool(process_num, initializer, initargs)


def track_parallel_progress(
    func,
    tasks,
    nproc,
    initializer=None,
    initargs=None,
    bar_width=50,
    chunksize=1,
    skip_first=False,
    keep_order=True,
    file=sys.stdout,
):
    """Track the progress of parallel task execution with a progress bar."""
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError('"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    gen = pool.imap(func, tasks, chunksize) if keep_order else pool.imap_unordered(
        func, tasks, chunksize
    )
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            if len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write("\n")
    pool.close()
    pool.join()
    return results


def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4]
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            dataset_pb2.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()


def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        range_image_origins = tf.zeros_like(range_image_points)
        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
        )
        range_image_origins = (
            tf.einsum("bkr,bijr->bijk", rotation, range_image_origins) + translation
        )
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points)
                + pixel_pose_translation
            )
            range_image_origins = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_origins)
                + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")
            # To vehicle frame corresponding to the given frame_pose
            # [B, 4, 4]
            world_to_vehicle = tf.linalg.inv(frame_pose)
            world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
            world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )
            range_image_origins = (
                tf.einsum(
                    "bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_origins
                )
                + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
            )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        range_image_origins = tf.cast(
            range_image_origins, dtype=range_image_polar_dtype
        )
        return range_image_points, range_image_origins


def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_points: [B, H, W, 3] with {x, y, z} as inner dims in vehicle frame.
      range_image_origins: [B, H, W, 3] with {x, y, z}, the origin of the range image
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar = range_image_utils.compute_range_image_polar(
            range_image, extrinsic, inclination, dtype=dtype
        )
        (
            range_image_points_cartesian,
            range_image_origins_cartesian,
        ) = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
        )
        return range_image_origins_cartesian, range_image_points_cartesian



class WaymoProcessor(object):
    """Process Waymo dataset.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        workers=16,
        num_frames=None,
    ):
        self.filter_no_label_zone_points = True

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)=
        self.save_track_id = False

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split(".")[0]) < 2:
            tf.enable_eager_execution()

        # keep the order defined by the official protocol
        self.cam_list = [
            "_FRONT",
            "_FRONT_LEFT",
            "_FRONT_RIGHT",
            "_SIDE_LEFT",
            "_SIDE_RIGHT",
        ]
        self.lidar_list = ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]
        # Waymo Open Dataset cameras are captured at 10 FPS.
        self.camera_fps = 10

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
        self.num_frames_limit = num_frames if num_frames is None else int(num_frames)

        self.tfrecord_pathnames = sorted(glob(os.path.join(self.load_dir, "*.tfrecord")))
        self.scene_names = [os.path.splitext(os.path.basename(p))[0] for p in self.tfrecord_pathnames]
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        id_list = range(len(self))
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        scene_name = self.scene_names[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type="")
        num_frames_total = sum(1 for _ in dataset)
        limit = (
            num_frames_total
            if self.num_frames_limit is None
            else min(self.num_frames_limit, num_frames_total)
        )
        for frame_idx, data in enumerate(tqdm(
            dataset,
            desc=f"File {file_idx}",
            total=limit,
            dynamic_ncols=True,
        )):
            if frame_idx >= limit:
                break
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            self.save_image(frame, scene_name, frame_idx)
            self.save_calib(frame, scene_name, frame_idx)
            self.save_pose(frame, scene_name, frame_idx)
            self.save_lidar_align(frame, scene_name, frame_idx, ri_index=0)
            self.save_dynamic_mask(frame, scene_name, frame_idx)
            if frame_idx == 0:
                self.save_interested_labels(frame, scene_name)
        self.save_videos(scene_name, limit)

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_interested_labels(self, frame, scene_name):
        """
        Saves the interested labels of a given frame to a JSON file.

        Args:
            frame: A `Frame` object containing the labels to be saved.
            file_idx: An integer representing the index of the file to be saved.

        Returns:
            None
        """
        frame_data = {
            "time_of_day": frame.context.stats.time_of_day,
            "location": frame.context.stats.location,
            "weather": frame.context.stats.weather,
        }
        object_type_name = lambda x: label_pb2.Label.Type.Name(x)
        object_counts = {
            object_type_name(x.type): x.count
            for x in frame.context.stats.camera_object_counts
        }
        frame_data.update(object_counts)
        # write as json
        with open(f"{self.save_dir}/{scene_name}/frame_info.json", "w") as fp:
            json.dump(frame_data, fp)

    def save_image(self, frame, scene_name, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = (
                f"{self.save_dir}/{scene_name}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.jpg"
            )
            with open(img_path, "wb") as fp:
                fp.write(img.image)

    def save_calib(self, frame, scene_name, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        extrinsics = []
        intrinsics = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            intrinsic = list(camera.intrinsic)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            np.savetxt(
                f"{self.save_dir}/{scene_name}/extrinsics/" + f"{str(i)}.txt",
                extrinsics[i],
            )
            np.savetxt(
                f"{self.save_dir}/{scene_name}/intrinsics/" + f"{str(i)}.txt",
                intrinsics[i],
            )


    def save_lidar_align(self, frame, scene_name, frame_idx, ri_index=0):
        """Precompute LiDAR-to-camera projections per frame for faster alignment."""
        (
            range_images,
            camera_projections,
            _,
            range_image_top_pose,
        ) = frame_utils.parse_range_image_and_camera_projection(frame)
        frame.lasers.sort(key=lambda laser: laser.name)
        points_list, points_cp_list = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=ri_index,
        )
        if not points_list or not points_cp_list:
            return

        points_all = np.concatenate(points_list, axis=0)
        points_cp_all = np.concatenate(points_cp_list, axis=0)
        distances = np.linalg.norm(points_all, axis=-1)

        calib_map = {cc.name: cc for cc in frame.context.camera_calibrations}
        for image in frame.images:
            cam_id = image.name
            mask = points_cp_all[:, 0] == cam_id
            if not np.any(mask):
                continue
            cp_sel = points_cp_all[mask]
            depth_sel = distances[mask]
            projected = np.concatenate([cp_sel[:, 1:3], depth_sel[:, None]], axis=-1).astype(np.float32)
            calib = calib_map.get(cam_id, None)
            h = getattr(image, "height", None)
            w = getattr(image, "width", None)
            if calib is not None:
                h = calib.height
                w = calib.width
            hw = np.array([h, w], dtype=np.int32)
            out_path = f"{self.save_dir}/{scene_name}/lidar_align/{str(frame_idx).zfill(3)}_{image.name - 1}.npz"
            np.savez_compressed(out_path, projected=projected, hw=hw)


    def save_pose(self, frame, scene_name, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            f"{self.save_dir}/{scene_name}/ego_pose/"
            + f"{str(frame_idx).zfill(3)}.txt",
            pose,
        )

    def save_dynamic_mask(self, frame, scene_name, frame_idx):
        """Parse and save the segmentation data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            # dynamic_mask
            img_path = (
                f"{self.save_dir}/{scene_name}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.jpg"
            )
            img_shape = np.array(Image.open(img_path))
            dynamic_mask = np.zeros_like(img_shape, dtype=np.float32)[..., 0]

            filter_available = any(
                [label.num_top_lidar_points_in_box > 0 for label in frame.laser_labels]
            )
            calibration = next(
                cc for cc in frame.context.camera_calibrations if cc.name == img.name
            )
            for label in frame.laser_labels:
                # camera_synced_box is not available for the data with flow.
                # box = label.camera_synced_box
                box = label.box
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])
                if not box.ByteSize():
                    continue  # Filter out labels that do not have a camera_synced_box.
                if (filter_available and not label.num_top_lidar_points_in_box) or (
                    not filter_available and not label.num_lidar_points_in_box
                ):
                    continue  # Filter out likely occluded objects.

                # Retrieve upright 3D box corners.
                box_coords = np.array(
                    [
                        [
                            box.center_x,
                            box.center_y,
                            box.center_z,
                            box.length,
                            box.width,
                            box.height,
                            box.heading,
                        ]
                    ]
                )
                corners = box_utils.get_upright_3d_box_corners(box_coords)[
                    0
                ].numpy()  # [8, 3]

                # Project box corners from vehicle coordinates onto the image.
                projected_corners = project_vehicle_to_image(
                    frame.pose, calibration, corners
                )
                u, v, ok = projected_corners.transpose()
                ok = ok.astype(bool)

                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not all(ok):
                    continue
                u = u[ok]
                v = v[ok]

                # Clip box to image bounds.
                u = np.clip(u, 0, calibration.width)
                v = np.clip(v, 0, calibration.height)

                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue

                # Draw projected 2D box onto the image.
                xy = (u.min(), v.min())
                width = u.max() - u.min()
                height = v.max() - v.min()
                # max pooling
                dynamic_mask[
                    int(xy[1]) : int(xy[1] + height),
                    int(xy[0]) : int(xy[0] + width),
                ] = np.maximum(
                    dynamic_mask[
                        int(xy[1]) : int(xy[1] + height),
                        int(xy[0]) : int(xy[0] + width),
                    ],
                    speed,
                )
            # thresholding, use 1.0 m/s to determine whether the pixel is moving
            dynamic_mask = np.clip((dynamic_mask > 1.0) * 255, 0, 255).astype(np.uint8)
            # Pillow >=10 deprecates the mode arg in fromarray; convert afterwards.
            dynamic_mask = Image.fromarray(dynamic_mask).convert("L")
            dynamic_mask_path = (
                f"{self.save_dir}/{scene_name}/dynamic_masks/"
                + f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.png"
            )
            dynamic_mask.save(dynamic_mask_path)

    def save_videos(self, scene_name, num_frames):
        """Create per-camera mp4 videos from saved images."""
        os.makedirs(f"{self.save_dir}/{scene_name}/videos", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for cam_id in range(len(self.cam_list)):
            first_frame_path = (
                f"{self.save_dir}/{scene_name}/images/{str(0).zfill(3)}_{cam_id}.jpg"
            )
            if not os.path.exists(first_frame_path):
                continue
            first_img = cv2.imread(first_frame_path)
            if first_img is None:
                continue
            height, width = first_img.shape[:2]

            out_path = f"{self.save_dir}/{scene_name}/videos/{cam_id}.mp4"
            writer = cv2.VideoWriter(out_path, fourcc, self.camera_fps, (width, height))

            # Write frames in order; skip missing frames silently.
            for idx in range(num_frames):
                img_path = (
                    f"{self.save_dir}/{scene_name}/images/{str(idx).zfill(3)}_{cam_id}.jpg"
                )
                img = cv2.imread(img_path)
                if img is None:
                    continue
                writer.write(img)

            writer.release()

    def create_folder(self):
        """Create folder for data preprocessing."""
        for scene_name in self.scene_names:
            os.makedirs(f"{self.save_dir}/{scene_name}/images", exist_ok=True)
            os.makedirs(
                f"{self.save_dir}/{scene_name}/ego_pose",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{scene_name}/extrinsics",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{scene_name}/intrinsics",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{scene_name}/dynamic_masks",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{scene_name}/videos",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.save_dir}/{scene_name}/lidar_align",
                exist_ok=True,
            )


if __name__ == "__main__":
    processor = WaymoProcessor(
        load_dir="/data/wlh/FreeDrive/data/waymo/raw",
        save_dir="/data/wlh/FreeDrive/data/waymo/processed",
        num_frames=49,
    )
    processor.convert()
