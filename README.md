# FreeDrive

## Overview

```
FreeDrive
├── code
│   ├── datasets
│   │   ├── lidar_to_2d_image.py        # tools for lidar projector
│   │   └── waymo_preprocess.py         # waymo dataset processor
│   ├── pointcloud
│   │   ├── depth_anything_3            # da3 official code
│   │   ├── align_depth_with_lidar.py   # TODO: More specific alignment by SAM 3/Per-frame npz like Gen3C
│   │   ├── render_from_npz.py          # export depth result to point cloud
│   │   └── da3_infer.py                # main func
│   └── video_painter_infer
│       ├── diffusers
│       ├── example                     # 示例输入输出与传参格式
│       ├── inpaint_custom.py           
│       ├── inpaint_custom.sh           # 启动脚本
│       └── requirements.txt            # 4090 推理依赖安装
└── data
    └── waymo
        ├── raw
        │   └── segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord
        └── processed
```

## Data Preprocess

依次处理文件夹下所有tfrecord，取出特定帧
```
conda activate wlh-py
cd /data/wlh/FreeDrive/code/datasets
python waymo_preprocess.py
```

将 LiDAR 投影到 2D 摄像机照片上
```
python lidar_to_2d_image.py
```

## Point Cloud

借 depth anything 3 的深度估计和点云导出 utils，先推理、再对齐、最后导出为 .glb 可以直接双击查看
```
conda activate da3
cd /data/wlh/DA3/code/Depth-Anything-3-main/src
CUDA_VISIBLE_DEVICES=3 python infer_demo.py
```

## Point Cloud Painter

借 VideoPainter 做的 pipeline。使用时应注意 sh 文件中的各文件、模型路径与prompt。
```
requirements.txt
conda activate videopainter
cd /data/wlh/FreeDrive/code/video_painter_infer
bash inpaint_custom.sh
```


## TODO

- [x] [2025/11/26] DA3 inference with provided camera
- [x] [2025/12/12] WOD tfrecord to images/cam extrinsics scripts
- [x] [2025/12/15] Reload scene from *.npz, get dense point cloud seq, unseen patch seq with given camera
- [x] [2025/12/16] Test VideoPainter with custom mask/rgb video.
- [ ] Construct fine-tune dataset
- [ ] FID/NTA IoU scripts
- [ ] Add reference frame/bbox branch and fine-tune
- [ ] Possibility to align real-world metric
- [ ] Possibility to change env settings
- [ ] Possibility to explore per-frame point cloud and align
- [ ] Possibility to support SIDE_LEFT/SIDE_RIGHT
- [ ] Possibility to DIY cam extrinsics like Gen3C


## Great Thanks

[Open3D PointCloud Viewer](https://salzi.blog/2022/05/14/waymo-open-dataset-open3d-point-cloud-viewer/) for his lidar projector.