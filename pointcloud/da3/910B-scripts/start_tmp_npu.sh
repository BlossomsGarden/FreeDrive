#!/bin/bash
# 启动脚本：在8张NPU卡上同时启动tmp_npu的8个进程（RANK0-7）

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "=========================================="
echo "Starting tmp_npu processes (RANK0-7)"
echo "=========================================="
echo "Time: $(date)"
echo ""

# 启动 RANK0-7，设备号 0-7
ASCEND_RT_VISIBLE_DEVICES=0  nohup  python occluded_pipeline_RANK0_tmp_npu.py > RANK0.out 2>&1 &
echo "Started RANK0 on device 0 (PID: $!)"

ASCEND_RT_VISIBLE_DEVICES=1  nohup python occluded_pipeline_RANK1_tmp_npu.py > RANK1.out 2>&1 &
echo "Started RANK1 on device 1 (PID: $!)"

ASCEND_RT_VISIBLE_DEVICES=2  nohup python occluded_pipeline_RANK2_tmp_npu.py > RANK2.out 2>&1 &
echo "Started RANK2 on device 2 (PID: $!)"

ASCEND_RT_VISIBLE_DEVICES=3  nohup python occluded_pipeline_RANK3_tmp_npu.py > RANK3.out 2>&1 &
echo "Started RANK3 on device 3 (PID: $!)"

ASCEND_RT_VISIBLE_DEVICES=4  nohup python occluded_pipeline_RANK4_tmp_npu.py > RANK4.out 2>&1 &
echo "Started RANK4 on device 4 (PID: $!)"

ASCEND_RT_VISIBLE_DEVICES=5  nohup python occluded_pipeline_RANK5_tmp_npu.py > RANK5.out 2>&1 &
echo "Started RANK5 on device 5 (PID: $!)"

ASCEND_RT_VISIBLE_DEVICES=6  nohup python occluded_pipeline_RANK6_tmp_npu.py > RANK6.out 2>&1 &
echo "Started RANK6 on device 6 (PID: $!)"

ASCEND_RT_VISIBLE_DEVICES=7  nohup python occluded_pipeline_RANK7_tmp_npu.py > RANK7.out 2>&1 &
echo "Started RANK7 on device 7 (PID: $!)"

echo ""
echo "=========================================="
echo "All 8 tmp_npu processes started!"
echo "Check logs: RANK0.out ~ RANK7.out"
echo "Check processes: ps aux | grep occluded_pipeline"
echo "=========================================="

