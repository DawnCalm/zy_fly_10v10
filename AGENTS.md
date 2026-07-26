# zy_fly_10v10 项目约束

本文件适用于仓库内全部目录。开始工作前，同时阅读：

- `docs/CURRENT_STATE.md`
- `docs/EXPERIMENTS.md`
- `docs/NEXT_STEPS.md`

## 项目目标

卓翼杯 2026 10v10 赛道以三局总拦截数为第一排名指标，总用时为
第二指标。任何优化都必须先保证命中数，再优化时间。

`main` 保存当前可运行、已验证的经典制导与 MAPPO residual 版本。
`high-v2-predictive-guidance` 用于下一代 High 预测制导实验。

## 比赛合规与安全边界

- 只使用比赛公开 ROS 接口。
- 禁止预设位置点、轨迹点或针对固定 seed 的硬编码。
- 不修改 `/home/ubuntu/zhuoyi_cup` 中的正式比赛文件。
- 所有目标位置、速度和预测都必须从本局实时雷达数据计算。
- low/mid 当前已达到 10/10；未经回归验证，不改变其默认实机参数。
- 新算法必须保留纯经典方案作为即时回退。
- 未通过离线配对验证的策略不得直接作为比赛默认方案。

## 环境与常用命令

工作目录：

```bash
cd /root/gpufree-data/fly/zy_fly_10v10
```

Python：

```bash
/opt/conda/envs/demo/bin/python
```

测试：

```bash
/opt/conda/envs/demo/bin/python -m unittest discover -s tests -v
/opt/conda/envs/demo/bin/python -m compileall -q \
  zhuoyi_mappo train.py evaluate.py ros_controller.py
bash -n run_platform.sh run_ros.sh
git diff --check
```

启动仿真和控制器的完整命令见 `README.md`。

## 实现原则

- 训练环境与 ROS 运行时应复用相同的观测、动作坐标和门控逻辑。
- 修改观测维度、动作语义或配置字段时，同步修改训练、评估、运行时和测试。
- 优先增加可解释、可独立 A/B 的模块，不同时替换预测、分配、制导和学习器。
- High-v2 按顺序验证：目标预测、APN、可达性分配、MPC、R-MAPPO。
- 目标预测优先采用物理模型或物理模型加小型 GRU 残差；数据充分前不直接使用大型 Transformer。
- 全局学习策略优先输出分配边权或战术参数，不直接端到端输出十架飞机的联合速度。
- 保持对智能体/目标编号的置换鲁棒性，避免网络记忆固定编号。

## 实验规则

- 对比方案必须使用相同难度、相同 seed 和相同平台参数。
- 每次真实仿真之间停止并重新启动平台，不能在同一局切换算法。
- 日志文件名必须包含算法、难度、关键参数和 seed。
- High 诊断日志推荐 `--log-interval 0.1`。
- 记录命中数、最后命中时间、逐目标最近距离、闭合速度、分配切换和指令/实际速度。
- 不以训练回报选择模型；使用未参与训练的固定验证 seed 选检查点。
- 更新 `docs/EXPERIMENTS.md` 后再宣称某个参数或模型更优。

## Git 与工作区

- 保留用户已有的未提交修改，不执行破坏性恢复命令。
- 每个实验阶段使用独立、可描述的提交。
- 不提交全部中间检查点；只提交明确选出的模型、配置、指标和可复查结果。
- 修改长期决策时同步更新 `docs/CURRENT_STATE.md` 和
  `docs/NEXT_STEPS.md`。
