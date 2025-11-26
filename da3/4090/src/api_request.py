import requests

# 参数配置
video_path = "cam02.mp4"
cam_id = "cam02"
gt_json_path = "camera_extrinsics.json"
api_url = "http://localhost:8888/evaluate_pose"

# 发送请求
payload = {
    "video_path": video_path,
    "cam_id": cam_id,
    "gt_json_path": gt_json_path
}

response = requests.post(api_url, json=payload, timeout=300)
result = response.json()

# 打印结果
print(f"RotErr Mean: {result['rot_err_mean']:.6f} rad")
print(f"TransErr Mean: {result['trans_err_mean']:.6f}")

