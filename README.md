# FreeDrive

## Overview

```
FreeDrive
├── code
│   └── datasets
│   │   ├── lidar_to_2d_image.py        # tools for lidar projector
│   │   └── waymo_preprocess.py         # waymo dataset processor
│   └── pointcloud
│       ├── depth_anything_3            # da3 official code
│       ├── align_depth_with_lidar.py   # TODO: More specific alignment by SAM 3
│       ├── custom_data_to_glb.py       # export depth result to point cloud
│       └── infer_demo.py               # main func
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


## TODO

- [x] DA3 inference with provided camera
- [x] WOD tfrecord to images/cam extrinsics scripts
- [ ] Reload scene from *.npz, get dense point cloud seq, unseen patch seq with given camera
- [ ] Support SIDE_LEFT/SIDE_RIGHT
- [ ] Test VideoPainter
- [ ] FID/NTA IoU
- [ ] Add Image Branch and fine-tune
- [ ] Specified alignment with SAM 3
- [ ] Dynamic Scene
- [ ] DIY cam extrinsics like Gen3C


## Great Thanks

[Open3D PointCloud Viewer](https://salzi.blog/2022/05/14/waymo-open-dataset-open3d-point-cloud-viewer/) for his lidar projector.