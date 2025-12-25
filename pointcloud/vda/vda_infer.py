"""
conda activate vda
pip install setuptools==80.8
cd /data/wlh/VDA/code
CUDA_VISIBLE_DEVICES=2   python3 run.py


他妈49*1920*1280 才 15396MB显存？？而且这么快？？尼玛骗我呢？你da3搞的什么构式
"""
from types import SimpleNamespace
import numpy as np
import os
import torch
import glob

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

if __name__ == '__main__':
    # 直接在这里配置，避免命令行参数：按需修改即可
    args = SimpleNamespace(
        input_video_folder='/data/wlh/FreeDrive/data/waymo/processed/segment-10061305430875486848_1080_000_1100_000_with_camera_labels/videos',  # 输入视频文件夹路径
        output_dir='./outputs',  # 输出目录
        input_size=518,  # 推理输入尺寸
        max_res=-1,  # 输入视频最长边的最大分辨率，-1表示使用原始分辨率（不resize）
        encoder='vitl',  # 选择模型：vits | vitb | vitl
        max_len=49,  # 最大帧数，-1 表示不限制
        target_fps=-1,  # 目标帧率，-1 使用原始帧率
        metric=True,  # 是否使用 metric 模型
        # metric=False,
        fp32=True,  # 使用 float32 推理（默认 float16）
        grayscale=False,  # 深度可视化是否仅灰度（否则彩色）
        save_npz=True,  # 是否保存深度为 npz
        save_exr=False,  # 是否保存深度为 exr
        # focal_length_x=470.4,  # x 方向焦距
        # focal_length_y=470.4,  # y 方向焦距
        cam_ids=[1, 0, 2],  # 指定要处理的摄像头ID列表，例如 [1, 0, 2] 会读取 1.mp4, 0.mp4, 2.mp4
    )


    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    video_depth_anything.load_state_dict(torch.load(f'../model/metric_video_depth_anything_vitl.pth', map_location='cpu'), strict=True)
    # video_depth_anything.load_state_dict(torch.load(f'../model/relative_video_depth_anything_vitl.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # 根据 cam_ids 配置读取视频文件
    video_folder = args.input_video_folder
    os.makedirs(args.output_dir, exist_ok=True)
    # 如果指定了 cam_ids，按照列表读取对应的视频文件
    mp4_files = []
    for cam_id in args.cam_ids:
        video_path = os.path.join(video_folder, f'{cam_id}.mp4')
        if os.path.exists(video_path):
            mp4_files.append((cam_id, video_path))
        else:
            assert False, f'警告: 未找到摄像头 {cam_id} 的视频文件: {video_path}'

    print(f'找到 {len(mp4_files)} 个视频文件，开始处理...')
    
    # 循环处理所有找到的mp4文件
    for idx, (cam_id, input_video) in enumerate(mp4_files, 1):
        print(f'[{idx}/{len(mp4_files)}] 正在处理摄像头 {cam_id} 的视频: {os.path.basename(input_video)}')
        frames, target_fps = read_video_frames(input_video, args.max_len, args.target_fps, args.max_res)
        depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

        # 使用摄像头ID作为输出文件名
        output_name = str(cam_id)
        processed_video_path = os.path.join(args.output_dir, output_name+'_src.mp4')
        depth_vis_path = os.path.join(args.output_dir, output_name+'_vis.mp4')
        save_video(frames, processed_video_path, fps=fps)
        save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

        if args.save_npz:
            depth_npz_path = os.path.join(args.output_dir, output_name+'_depths.npz')
            np.savez_compressed(depth_npz_path, depths=depths)
        if args.save_exr:
            depth_exr_dir = os.path.join(args.output_dir, output_name+'_depths_exr')
            os.makedirs(depth_exr_dir, exist_ok=True)
            import OpenEXR
            import Imath
            for i, depth in enumerate(depths):
                output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
                header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                exr_file = OpenEXR.OutputFile(output_exr, header)
                exr_file.writePixels({"Z": depth.tobytes()})
                exr_file.close()

        # if args.metric:
        #     import open3d as o3d

        #     width, height = depths[0].shape[-1], depths[0].shape[-2]
        #     x, y = np.meshgrid(np.arange(width), np.arange(height))
        #     x = (x - width / 2) / args.focal_length_x
        #     y = (y - height / 2) / args.focal_length_y

        #     for i, (color_image, depth) in enumerate(zip(frames, depths)):
        #         z = np.array(depth)
        #         points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        #         colors = np.array(color_image).reshape(-1, 3) / 255.0

        #         pcd = o3d.geometry.PointCloud()
        #         pcd.points = o3d.utility.Vector3dVector(points)
        #         pcd.colors = o3d.utility.Vector3dVector(colors)
        #         o3d.io.write_point_cloud(os.path.join(args.output_dir, 'point' + str(i).zfill(4) + '.ply'), pcd)
        
        print(f'摄像头 {cam_id} 的视频处理完成\n')
    
    print('所有视频处理完成！')
