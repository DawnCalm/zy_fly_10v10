# zy_fly_10v10

卓翼杯 2026 10v10 控制器。当前比赛版本只保留两种可解释方案：

- `classic`：匈牙利一对一分配 + 三维提前量 + 近距末制导；
- `los_pn`：在 classic 上叠加提前启用、受加速度限制的 LOS-rate PN，
  仅用于 High，随预计剩余时间渐进介入，近距离平滑退出。

控制器只读取公开 ROS 话题，不修改比赛平台文件，也不包含固定 seed、
预设位置或轨迹。

## 当前默认值

| 难度 | 速度 | 末制导距离 / 增益 | deadline |
| --- | ---: | ---: | --- |
| low | 20 m/s | 60 m / 0.7 | auto 启用 |
| mid | 25 m/s | 60 m / 0.7 | auto 启用 |
| high | 30 m/s | 15 m / 2.0 | auto 禁用 |

High LOS-PN：`N=3`，最大加速度 `5 m/s²`，`tgo=6 s` 开始介入、
`tgo=4 s` 完全介入，响应前视 `2.5 s`，距目标 5 m 内衰减。

当前正式平台同 seed `20260723` A/B：classic `3/10`，LOS-PN `5/10`。
另 5 个随机 High seed 的 LOS-PN 成绩为 `8、7、7、5、7`，平均
`6.8/10`。LOS-PN 是当前 High 首选方案，但仍保持显式选择。

## 验证

```bash
cd /root/gpufree-data/fly/zy_fly_10v10
/opt/conda/envs/demo/bin/python -m unittest discover -s tests -v
/opt/conda/envs/demo/bin/python -m compileall -q \
  zhuoyi_mappo ros_controller.py
bash -n run_platform.sh run_ros.sh
git diff --check
```

## 真实平台

终端一启动固定 seed High：

```bash
bash run_platform.sh --scene 10v10 --diff high --seed 20260723
```

终端二运行经典基线：

```bash
bash run_ros.sh \
  --mode classic --difficulty high --guidance classic \
  --target-deadline-mode disabled --auto-arm \
  --log artifacts/ros/real_classic_high_seed20260723.jsonl
```

每次对照都应完整重启平台。LOS-PN：

```bash
bash run_ros.sh \
  --mode classic --difficulty high --guidance los_pn \
  --target-deadline-mode disabled --auto-arm \
  --log artifacts/ros/real_los_pn_high_seed20260723.jsonl
```

控制模式默认持续到所有目标结束；`observe` 模式默认 20 秒。启动脚本默认
使用正式平台 `/home/ubuntu/zhuoyi_cup/run.sh`，只有显式设置
`ZHUOYI_PLATFORM_RUN` 才会切换。

## 目录

- `ros_controller.py`：ROS 接入、状态保护和控制循环；
- `zhuoyi_mappo/assignment.py`：分配和经典提前量；
- `zhuoyi_mappo/guidance.py`：LOS 运动学与受限 PN；
- `zhuoyi_mappo/runtime_core.py`：两种制导的共享运行核心；
- `zhuoyi_mappo/tracking.py`：Alpha-Beta 跟踪和异常状态保护；
- `docs/CURRENT_STATE.md`：当前结论与下一步；
- `docs/HISTORY.md`：被拒绝方法的简要结果。

训练网络、IMM/APN、GRU 及其中间模型已从比赛分支移除；原因和大致结果
仅保留在历史摘要中。
