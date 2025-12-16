# VideoPainter V2V 必要输入信息清单

## 一、核心必要输入 (Core Required Inputs)

| 输入项 | 格式 | 说明 | 示例 |
|--------|------|------|------|
| **rgb_video_path** | 文件路径 | RGB 彩色视频文件 | `./input/video.mp4` |
| **mask_video_path** | 文件/目录路径 | Mask 文件（视频/numpy/图像序列） | `./input/mask.mp4` 或 `./input/masks.npz` |
| **prompt** | 字符串 | 视频内容描述文本 | `"A person walking in a park"` |
| **model_path** | 目录路径 | CogVideoX 基础模型路径 | `../ckpt/CogVideoX-5b-I2V` |
| **output_path** | 文件路径 | 输出视频保存路径 | `./output/result.mp4` |

## 二、强烈推荐输入 (Highly Recommended)

| 输入项 | 格式 | 说明 | 示例 |
|--------|------|------|------|
| **inpainting_branch** | 目录路径 | 视频修复专用分支模型 | `../ckpt/VideoPainter/checkpoints/branch` |

## 三、可选但有用的输入 (Optional but Useful)

| 输入项 | 格式 | 说明 | 默认值 |
|--------|------|------|--------|
| **id_adapter_resample_learnable_path** | 目录路径 | ID 适配器权重路径 | None |
| **fps** | 整数 | 视频帧率（0=自动检测） | 自动检测 |
| **num_frames** | 整数 | 生成帧数（None=全部） | 全部帧 |
| **mask_background** | 布尔值 | 是否 mask 背景 | False |
| **dilate_size** | 整数 | Mask 膨胀大小 | 0 |

## 四、输入格式详细说明

### 1. RGB Video (rgb_video_path)
- **支持格式**: `.mp4`, `.avi`, `.mov` 等
- **要求**: RGB 彩色视频，包含需要修复的区域

### 2. Mask (mask_video_path) - 三种格式任选其一

#### 格式 A: 视频文件
- **支持格式**: `.mp4`, `.avi` 等
- **像素值**: 
  - 白色区域 (>128) = 需要修复的区域
  - 黑色区域 (<=128) = 保留区域

#### 格式 B: Numpy 数组文件
- **支持格式**: `.npz`, `.npy`
- **Shape**: `(T, H, W)` 或 `(T, H, W, C)`
- **值**: > 0.5 表示 mask 区域

#### 格式 C: 图像序列目录
- **内容**: 包含按顺序命名的 mask 图像
- **格式**: `.png`, `.jpg`, `.jpeg`
- **命名**: 可排序（如 `000.png`, `001.png`）

### 3. Prompt (prompt)
- **类型**: 字符串
- **内容**: 描述视频内容的文本
- **示例**: 
  - `"A person walking in a park"`
  - `"A dog running on the beach"`
  - `"A car driving on the road"`

### 4. Model Paths

#### model_path (必需)
- CogVideoX 基础模型
- 包含完整的 transformer、VAE、text_encoder 等

#### inpainting_branch (推荐)
- 视频修复专用分支模型
- 显著提升修复效果

## 五、Metadata 文件示例

### JSON 格式 (metadata_example.json)
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

### CSV 格式 (metadata_example.csv)
```csv
rgb_video_path,mask_video_path,prompt,output_path,model_path,inpainting_branch,fps,num_frames
./input/video1.mp4,./input/mask1.mp4,"A person walking in a park",./output/result1.mp4,../ckpt/CogVideoX-5b-I2V,../ckpt/VideoPainter/checkpoints/branch,30,49
./input/video2.mp4,./input/mask2.npz,"A dog running on the beach",./output/result2.mp4,../ckpt/CogVideoX-5b-I2V,../ckpt/VideoPainter/checkpoints/branch,25,49
```

## 六、最小配置示例

```bash
python infer/inpaint_custom.py \
    --rgb_video_path "video.mp4" \
    --mask_video_path "mask.mp4" \
    --prompt "A person walking" \
    --model_path "../ckpt/CogVideoX-5b-I2V" \
    --output_path "output.mp4"
```

## 七、完整配置示例

```bash
python infer/inpaint_custom.py \
    --rgb_video_path "video.mp4" \
    --mask_video_path "mask.mp4" \
    --prompt "A person walking in a park" \
    --model_path "../ckpt/CogVideoX-5b-I2V" \
    --output_path "output.mp4" \
    --inpainting_branch "../ckpt/VideoPainter/checkpoints/branch" \
    --id_adapter_resample_learnable_path "../ckpt/VideoPainterID/checkpoints" \
    --num_frames 49 \
    --fps 30 \
    --mask_background \
    --dilate_size 32 \
    --first_frame_gt \
    --replace_gt
```

## 八、快速检查清单

使用前请确认：

- [ ] RGB 视频文件存在且可读
- [ ] Mask 文件/目录存在且格式正确
- [ ] Prompt 文本已准备好
- [ ] 模型路径正确且模型文件完整
- [ ] 输出目录有写入权限
- [ ] RGB 视频和 mask 帧数匹配（或可接受自动裁剪）
- [ ] GPU 内存充足（建议 >= 24GB）

## 九、常见问题

**Q: RGB 视频和 mask 帧数不匹配怎么办？**
A: 程序会自动使用较小的帧数，并给出警告。

**Q: Mask 格式如何选择？**
A: 
- 如果已有视频格式的 mask，直接用视频文件
- 如果使用 Python 生成，推荐 numpy 数组（.npz）
- 如果是一帧一帧的图像，用图像序列目录

**Q: 必须使用 inpainting_branch 吗？**
A: 不是必须的，但强烈推荐。不使用会影响修复效果。

**Q: V2V 和 I2V 有什么区别？**
A: 
- V2V (Video-to-Video): 输入是完整的视频和 mask，直接进行视频修复
- I2V (Image-to-Video): 输入是单张图像和 mask，生成视频序列

**Q: 可以使用 metadata 文件批量处理吗？**
A: 可以。参考 `metadata_example.json` 或 `metadata_example.csv` 创建 metadata 文件，然后编写脚本读取并批量调用。
