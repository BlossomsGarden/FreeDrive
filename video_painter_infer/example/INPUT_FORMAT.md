# VideoPainter V2V 输入格式说明

本文档说明使用 VideoPainter 预训练模型进行 V2V (Video-to-Video) 视频修复所需的输入信息及其格式要求。

## 一、必要输入信息 (Required)

### 1. RGB 视频 (rgb_video_path)
- **格式**: 视频文件
- **支持格式**: `.mp4`, `.avi`, `.mov` 等常见视频格式
- **要求**: 
  - RGB 彩色视频
  - 包含需要修复的区域（缺失或需要替换的区域）
  - 视频帧数应与 mask 帧数匹配

### 2. Mask 视频/文件 (mask_video_path)
- **格式**: 支持三种格式
  - **视频文件**: `.mp4`, `.avi` 等
    - 每个帧应为灰度图或 RGB 图
    - 白色区域（>128）表示需要修复的区域
    - 黑色区域（<128）表示保留区域
  - **Numpy 数组文件**: `.npz`, `.npy`
    - Shape: `(T, H, W)` 或 `(T, H, W, C)`
    - 值 > 0.5 表示 mask 区域
    - 值 <= 0.5 表示非 mask 区域
  - **图像序列目录**: 包含 mask 图像的文件夹
    - 图像格式: `.png`, `.jpg`, `.jpeg`
    - 按文件名排序
    - 白色区域表示 mask 区域

### 3. 视频描述文本 (prompt)
- **格式**: 字符串
- **要求**: 
  - 描述视频内容的文本提示词
  - 例如: "A person walking in a park", "A dog running on the beach"
  - 用于指导模型生成修复内容

### 4. 预训练模型路径 (model_path)
- **格式**: 目录路径
- **要求**: 
  - CogVideoX 基础模型路径
  - 例如: `../ckpt/CogVideoX-5b-I2V`

### 5. 输出路径 (output_path)
- **格式**: 文件路径
- **要求**: 
  - 输出视频的保存路径
  - 例如: `./output/inpainted_video.mp4`

## 二、推荐输入信息 (Recommended)

### 1. Inpainting 分支模型 (inpainting_branch)
- **格式**: 目录路径
- **说明**: 
  - 用于视频修复的专用分支模型
  - 强烈推荐使用，可显著提升修复效果
  - 例如: `../ckpt/VideoPainter/checkpoints/branch`

## 三、可选输入信息 (Optional)

### 1. ID 适配器权重 (id_adapter_resample_learnable_path)
- **格式**: 目录路径
- **说明**: 
  - ID 池重采样可学习权重
  - 用于提升身份一致性
  - 例如: `../ckpt/VideoPainterID/checkpoints`

### 2. 视频参数
- **fps**: 视频帧率（整数，0 表示自动检测）
- **num_frames**: 要生成的帧数（整数，None 表示使用所有帧）
- **start_frame**: 起始帧索引（整数，默认 0）
- **end_frame**: 结束帧索引（整数，None 表示使用所有帧）
- **down_sample_fps**: 处理时的下采样帧率（整数，默认 8）

### 3. Mask 参数
- **mask_background**: 布尔值，True 表示 mask 背景，False 表示 mask 前景（默认 False）
- **dilate_size**: Mask 膨胀大小（整数，0 表示不膨胀，默认 0）

### 4. 生成参数
- **num_inference_steps**: 推理步数（整数，默认 50）
- **guidance_scale**: 引导强度（浮点数，默认 6.0）
- **num_videos_per_prompt**: 每个 prompt 生成的视频数（整数，默认 1）
- **dtype**: 数据类型（"float16" 或 "bfloat16"，默认 "bfloat16"）
- **seed**: 随机种子（整数，默认 42）

### 5. 修复参数
- **first_frame_gt**: 布尔值，使用第一帧作为 ground truth（默认 False）
- **replace_gt**: 布尔值，替换 ground truth 帧（默认 False）
- **mask_add**: 布尔值，将 mask 添加到输入（默认 False）
- **prev_clip_weight**: 前一 clip 的权重（浮点数，默认 0.0）
- **overlap_frames**: 重叠帧数（整数，默认 0）
- **long_video**: 布尔值，启用长视频模式（默认 False）

## 四、输入格式示例

### 示例 1: 使用视频文件作为 mask
```bash
python infer/inpaint_custom.py \
    --rgb_video_path "./input/rgb_video.mp4" \
    --mask_video_path "./input/mask_video.mp4" \
    --prompt "A person walking in a park" \
    --model_path "../ckpt/CogVideoX-5b-I2V" \
    --output_path "./output/result.mp4" \
    --inpainting_branch "../ckpt/VideoPainter/checkpoints/branch"
```

### 示例 2: 使用 Numpy 数组作为 mask
```bash
python infer/inpaint_custom.py \
    --rgb_video_path "./input/rgb_video.mp4" \
    --mask_video_path "./input/masks.npz" \
    --prompt "A dog running on the beach" \
    --model_path "../ckpt/CogVideoX-5b-I2V" \
    --output_path "./output/result.mp4" \
    --inpainting_branch "../ckpt/VideoPainter/checkpoints/branch" \
    --id_adapter_resample_learnable_path "../ckpt/VideoPainterID/checkpoints"
```

### 示例 3: 使用图像序列目录作为 mask
```bash
python infer/inpaint_custom.py \
    --rgb_video_path "./input/rgb_video.mp4" \
    --mask_video_path "./input/mask_frames/" \
    --prompt "A car driving on the road" \
    --model_path "../ckpt/CogVideoX-5b-I2V" \
    --output_path "./output/result.mp4" \
    --inpainting_branch "../ckpt/VideoPainter/checkpoints/branch" \
    --num_frames 49 \
    --fps 30
```

## 五、Metadata 文件格式

### JSON 格式示例
参考 `infer/metadata_example.json`:
```json
{
    "rgb_video_path": "./input/rgb_video.mp4",
    "mask_video_path": "./input/mask_video.mp4",
    "prompt": "A person walking in a park",
    "output_path": "./output/inpainted_video.mp4",
    "model_path": "../ckpt/CogVideoX-5b-I2V",
    "inpainting_branch": "../ckpt/VideoPainter/checkpoints/branch",
    "fps": 30,
    "num_frames": 49
}
```

### CSV 格式示例
参考 `infer/metadata_example.csv`:
```csv
rgb_video_path,mask_video_path,prompt,output_path,model_path,inpainting_branch,fps,num_frames
./input/video1.mp4,./input/mask1.mp4,"A person walking in a park",./output/result1.mp4,../ckpt/CogVideoX-5b-I2V,../ckpt/VideoPainter/checkpoints/branch,30,49
./input/video2.mp4,./input/mask2.npz,"A dog running on the beach",./output/result2.mp4,../ckpt/CogVideoX-5b-I2V,../ckpt/VideoPainter/checkpoints/branch,25,49
```

## 六、Mask 格式详细说明

### 视频文件格式
- 每个帧可以是灰度图或 RGB 图
- 像素值 > 128 的区域表示需要修复的区域（mask 区域）
- 像素值 <= 128 的区域表示保留区域

### Numpy 数组格式
- **Shape**: `(T, H, W)` 或 `(T, H, W, C)`
  - `T`: 帧数
  - `H`: 高度
  - `W`: 宽度
  - `C`: 通道数（可选）
- **数据类型**: `float` 或 `uint8`
- **值范围**: 
  - 对于 float: 0.0-1.0，> 0.5 表示 mask 区域
  - 对于 uint8: 0-255，> 128 表示 mask 区域
- **保存格式**: 
  - `.npz`: 使用 `np.savez()` 保存，键名可以是 `'arr_0'` 或 `'masks'`
  - `.npy`: 使用 `np.save()` 保存

### 图像序列格式
- 目录中包含按顺序命名的图像文件
- 文件名应可排序（如 `000.png`, `001.png`, ... 或 `frame_000.png`, `frame_001.png`, ...）
- 图像格式: `.png`, `.jpg`, `.jpeg`
- 白色区域表示 mask 区域，黑色区域表示保留区域

## 七、注意事项

1. **帧数匹配**: RGB 视频和 mask 的帧数应该匹配。如果不匹配，将使用较小的帧数。

2. **分辨率**: RGB 视频和 mask 的分辨率应该匹配。如果不匹配，mask 会被调整到视频分辨率。

3. **Mask 方向**: 
   - `mask_background=False` (默认): 白色区域（255）表示需要修复的前景区域
   - `mask_background=True`: 白色区域（255）表示需要保留的前景区域，黑色区域（0）表示需要修复的背景区域

4. **Prompt 质量**: 
   - 视频 prompt 应该描述整个视频的内容
   - 建议使用简洁、准确的描述

5. **模型路径**: 确保所有模型路径正确，模型文件完整。

6. **GPU 内存**: 根据视频长度和分辨率，可能需要较大的 GPU 内存。可以使用 `--down_sample_fps` 和 `--num_frames` 来减少内存使用。

7. **V2V vs I2V**: 
   - V2V (Video-to-Video): 输入是完整的视频和 mask，直接进行视频修复，不需要首帧图像修复
   - I2V (Image-to-Video): 输入是单张图像和 mask，需要首帧图像修复模型

## 八、快速开始

1. 准备 RGB 视频和 mask 文件
2. 修改 `infer/inpaint_custom.sh` 中的路径和参数
3. 运行: `bash infer/inpaint_custom.sh`

或直接使用 Python 命令:
```bash
python infer/inpaint_custom.py \
    --rgb_video_path "your_video.mp4" \
    --mask_video_path "your_mask.mp4" \
    --prompt "your prompt" \
    --model_path "path/to/model" \
    --output_path "output.mp4"
```

## 九、批量处理

可以使用 metadata 文件进行批量处理。参考 `metadata_example.json` 或 `metadata_example.csv` 创建 metadata 文件，然后编写脚本读取并批量调用 `inpaint_custom.py`。
