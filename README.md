# zy_fly_10v10

卓翼杯 2026 10v10 控制器。当前比赛版本保留三种可解释方案：

更新时间：2026-08-12。

- `classic`：匈牙利一对一分配 + 三维提前量 + 近距末制导；
- `los_pn`：在 classic 上叠加提前启用、受加速度限制的 LOS-rate PN，
  随预计剩余时间渐进介入，近距离平滑退出；
- `los_pn_kf`：High 使用的自适应 LOS Kalman 融合，在受限 PN 前降低
  LOS-rate 估计滞后，并对相对运动学估计的修正做硬限制。

控制器只读取公开 ROS 话题，不修改比赛平台文件，也不包含固定 seed、
预设位置或轨迹。

## 当前 benchmark

| 难度 | 制导 | 速度 | 末制导距离 / 增益 | deadline |
| --- | --- | ---: | ---: | --- |
| low | LOS-PN | 20 m/s | 60 m / 0.7 | auto 启用 |
| mid | LOS-PN | 25 m/s | 60 m / 0.7 | auto 启用 |
| high | LOS-PN + adaptive LOS KF | 30 m/s | 15 m / 2.0 | auto 禁用 |

LOS-PN：`N=3`，最大加速度 `5 m/s²`，`tgo=6 s` 开始介入、
`tgo=4 s` 完全介入，响应前视 `2.5 s`，距目标 5 m 内衰减。

命令行默认按难度选择 High=`los_pn_kf`、Low/Mid=`los_pn`；原运动学
LOS-PN 和 classic 均可显式回退。

## 真实平台成绩

所有对比均使用同 seed、完整重启平台的 A/B。不同表行使用的 seed 组
不同，均值不能跨行直接合并。

| 难度 / 阶段 | 候选成绩 | 对照成绩 | 候选均值 | 结论 |
| --- | ---: | ---: | ---: | --- |
| Low，3 seed | LOS-PN `30/30` | classic `30/30` | `10.0/10` | 3/3 更快，合计快 `18.348 s` |
| Mid，3 seed | LOS-PN `30/30` | classic `29/30` | `10.0/10` | 3/3 胜出 |
| High，5 seed | LOS-PN `34/50` | classic `18/50` | `6.8/10` | 逐局 `8、7、7、5、7`，5/5 改善 |
| High KF，另5 seed | `los_pn_kf` `33/50` | LOS-PN `30/50` | `6.6/10` | 2胜3平0负，升级为当前 High benchmark |

因此常说的“High 平均约 6.5 架”对应当前 `los_pn_kf` 的精确结果
`6.6/10`；原始 LOS-PN 在另一组5个 seed 上的精确均值是 `6.8/10`。
固定 seed `20260723` 的历史 A/B 为 classic `3/10`、LOS-PN `5/10`。

## 主要尝试结果

| 方法 | 真实平台结果 | 决策 |
| --- | --- | --- |
| PN 上限 `5 -> 7.5 m/s²` | `33/50 -> 28/50`，LOS-rate 与跟随误差变差 | 删除，保持 `5 m/s²` |
| 原始 LOS 方向直接差分 | `29/50 -> 27/50`，PN 跳变 P95 增加约 `80%` | 删除 |
| 自适应 LOS Kalman | `30/50 -> 33/50`，末段 LOS-rate P95 降低 `7.7%` | 成为 High benchmark |
| 末端最小闭合速度 `5 m/s` | `24/50 -> 29/50`，但出现单 seed `-3` | 仅保留显式候选 |
| LOS-rate 门控闭合下限 | benchmark `28/50`，候选 `27/50`，尾追更久 | 删除 |
| 预测碰撞走廊门控 | benchmark `33/50`，候选 `23/50` | 删除 |
| MAPPO 受限速度残差 v2 | 离线 `+0.275/局`，实机 `13/20 -> 12/20` | 删除 |
| 无安全区 MAPPO 横向加速度残差 | 30 个冻结验证 seed：LOS-PN `182/300`，第20/30次检查点均 `181/300` | 删除，不进入实机 |
| 响应模型单步横向反馈 | 预测误差下降，但实机 `8/10 -> 5/10` | 删除控制路径，保留模型 |
| 多步预测最近交会距离 v2 | 已完成3个可比 seed：`11/30 -> 14/30`，逐局 `-1、+1、+3`；另1局候选 `5/10` 尚缺对照 | 实验中，不替换 benchmark |
| 稳健响应感知多步 CPA v3 | 5 个全新 High seed：`26/50 -> 26/50`，逐局 `+3、+1、0、-2、-2` | 扩样本后收益归零且连续回归，删除 v3 |
| 靶机航向前方斜侧占位 | 3 个 High 同 seed 配对：benchmark `13/30`、候选 `11/30`，逐局 `+1、-1、-2`；只改善约 100 m 的前向位置，30 m 时优势消失 | 删除 |
| 外部 v14 高空控制，禁用碰撞包 | seed `20260723` 真实物理命中 `0/10` | 不采用；其文档10/10依赖内部碰撞包 |

飞控响应模型使用 `54,058` 条真实时序数据训练。在独立日志中，15 m 内
约2秒速度预测误差从 `5.07` 降至 `1.90 m/s`，说明系统辨识有效；但
预测准确不等于控制目标正确。多步 CPA v3 虽在首轮3对中改善，但补足5对
后与 benchmark 同为 `26/50`，并在最后两个种子各回归2架，因此 v3 控制
代码已删除。更完整的逐 seed 结果和失败诊断见 `docs/CURRENT_STATE.md` 与
`docs/HISTORY.md`。

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

终端二运行当前 benchmark：

```bash
bash run_ros.sh \
  --mode classic --difficulty high --guidance los_pn_kf \
  --target-deadline-mode disabled --auto-arm \
  --log artifacts/ros/real_los_pn_kf_high_seed20260723.jsonl
```

每次对照都应完整重启平台。classic 回退：

```bash
bash run_ros.sh \
  --mode classic --difficulty high --guidance classic \
  --target-deadline-mode disabled --auto-arm \
  --log artifacts/ros/real_classic_high_seed20260723.jsonl
```

控制模式默认持续到所有目标结束；`observe` 模式默认 20 秒。启动脚本默认
使用正式平台 `/home/ubuntu/zhuoyi_cup/run.sh`，只有显式设置
`ZHUOYI_PLATFORM_RUN` 才会切换。

## 目录

- `ros_controller.py`：ROS 接入、状态保护和控制循环；
- `zhuoyi_mappo/assignment.py`：分配和经典提前量；
- `zhuoyi_mappo/guidance.py`：LOS 运动学与受限 PN；
- `zhuoyi_mappo/runtime_core.py`：三种制导的共享运行核心；
- `zhuoyi_mappo/los_kalman.py`：High 自适应 LOS Kalman 融合；
- `zhuoyi_mappo/tracking.py`：Alpha-Beta 跟踪和异常状态保护；
- `docs/CURRENT_STATE.md`：当前结论与下一步；
- `docs/HISTORY.md`：被拒绝方法的简要结果。

训练网络、IMM/APN、GRU 及其中间模型已从比赛分支移除；原因和大致结果
仅保留在历史摘要中。
