# zy_fly_10v10：卓翼杯 10v10 经典制导 + MAPPO

本目录是独立参赛算法工程，不直接读取或修改比赛运行包内的比赛文件，
也不加载去年的模型权重。

`run_platform.sh` 默认调用镜像正式比赛包
`/home/ubuntu/zhuoyi_cup/run.sh`。如需测试其他副本，必须显式设置
`ZHUOYI_PLATFORM_RUN`，避免因附近存在开发副本而静默切换。参赛控制器
`ros_controller.py` 只依赖公开 ROS 接口，与运行包路径无关。

长期开发上下文：

- `AGENTS.md`：项目约束、验证命令和实验规则；
- `docs/CURRENT_STATE.md`：当前经典/MAPPO 状态和已知瓶颈；
- `docs/EXPERIMENTS.md`：真实与离线实验记录；
- `docs/NEXT_STEPS.md`：High-v2 分阶段实现路线。

当前实现包含快速运动学训练环境：

- 10 架拦截机与 10 架靶机；
- 匈牙利算法按预计拦截时间进行一对一目标分配；
- 经典三维提前量制导提供基础速度；
- 共享 Actor 输出制导坐标系内受限的前向/侧向/垂向速度残差；
- 集中式 Critic 读取完整 10v10 态势并从零训练；
- low/mid/high 三种随机化靶机运动；
- High 中残差在 250 m 内渐进启用、80 m 内完全启用；
- 训练 minibatch 始终保留同一时刻的联合智能体状态。

训练场景只使用随机生成的相对几何关系，不含比赛预设位置点。High
专项配置已经依据真实 ROS 日志标定速度范围、起动时间和飞控响应，
并在每回合随机化响应时间、靶机相位和碰撞半径。

## 环境

当前服务器已存在同时包含 PyTorch、SciPy 和 ROS Python 包的 Conda
环境。建议明确 source ROS，并显式选择该解释器：

```bash
source /opt/ros/noetic/setup.bash
source /opt/rostrans/sdk/x86_64-u20.04-ros1-noetic/setup.bash
PYTHON=/opt/conda/envs/demo/bin/python
```

## 自动测试

```bash
cd /root/gpufree-data/fly/zy_fly_10v10
$PYTHON -m unittest discover -s tests -v
```

## 从零训练

短程烟测：

```bash
$PYTHON train.py \
  --updates 2 \
  --num-envs 2 \
  --rollout-steps 32 \
  --update-epochs 1 \
  --hidden-dim 64 \
  --output artifacts/smoke
```

High 专项正式训练（`high-real` 是当前默认 profile）：

```bash
$PYTHON train.py \
  --profile high-real \
  --updates 100 \
  --num-envs 8 \
  --rollout-steps 256 \
  --device cuda \
  --output artifacts/mappo_high_real_v1
```

从检查点继续训练：

```bash
$PYTHON train.py \
  --profile high-real \
  --updates 50 \
  --num-envs 8 \
  --rollout-steps 256 \
  --device cuda \
  --resume artifacts/mappo_high_real_v1/checkpoint_00050.pt \
  --output artifacts/mappo_high_real_v1
```

`--updates` 表示本次命令继续运行多少次更新，日志和检查点中的 update
编号会接着已有模型递增。

训练会生成：

- `config.json`：环境和训练参数；
- `metrics.jsonl`：每次更新的损失和完赛统计；
- `latest.pt`：最新模型；
- `checkpoint_*.pt`：周期性检查点。

## 评估

使用检查点内完全相同的环境配置和随机种子做配对评估：

```bash
$PYTHON evaluate.py \
  --checkpoint artifacts/mappo_high_real_v1/best_offline.pt \
  --compare-classic \
  --difficulties high \
  --action-scales 0.5 \
  --episodes 50 \
  --seed 50000 \
  --json-output artifacts/evaluations/high_final.json
```

先比较平均拦截数和满拦截率，只有拦截数稳定后再比较回合步数。
评估 JSON 同时保存逐种子结果、均值、标准误和改善/持平/变差局数。

## 当前阶段结果

已在比赛镜像的真实 10v10 ROS/PX4 平台上使用固定 seed `20260723`
完成经典方案 A/B。下表为各难度当前最佳官方裁判结果：

| 难度 | 经典方案参数 | 命中 | 最后命中时间 |
| --- | --- | ---: | ---: |
| low | 20 m/s，末制导 60 m / 0.7 | 10/10 | 78.100 s |
| mid | 25 m/s | 10/10 | 98.255 s |
| high | 30 m/s，末制导 15 m / 2.0 | 4/10 | 87.552 s |
| 合计 | 纯经典，无 MAPPO 残差 | 24/30 | — |

High 对照实验表明：原参数为 3/10，缩短预测时域为 2/10，提高到
35 m/s 为 2/10，末制导增益提高到 3.0 为 3/10。因此默认配置保留
当前最佳的 30 m/s、15 m / 2.0。

High 专项 MAPPO 已从零训练到 100 个更新。固定验证选择
`checkpoint_00080.pt`，并复制为 `best_offline.pt`；部署候选使用
`--residual-scale 0.5`，因此最大残差速度为 4.5 m/s。三组未参与训练的
验证种子共 100 局结果如下：

| 方案 | 总命中 | 平均命中 |
| --- | ---: | ---: |
| 相同配置的纯经典制导 | 491/1000 | 4.91/10 |
| MAPPO checkpoint 80，残差 0.5 倍 | 505/1000 | 5.05/10 |

离线平均提升为 `+0.14` 架/局。最后一组全新 50 局为 4.96 对 5.04，
配对标准误约 0.14，因此方向为正但尚不能视为显著、稳定的实机收益。
该模型是受控实机 A/B 候选，尚未替代比赛用经典方案。详细结果见
`artifacts/evaluations/mappo_high80_scale05_seed50000_n50.json`。

## High-v2：IMM、GRU 与 3D APN/ZEM

当前分支新增了默认关闭的 High-v2 实验链：

```text
10 Hz 雷达 -> CV/CA/CT-IMM -> 3D APN/ZEM -> 原速度/加速度安全限制
```

真实 High 日志回放命令：

```bash
$PYTHON analyze_prediction.py \
  --output artifacts/prediction/imm_high_seed20260723.json
```

小型 GRU 只学习 IMM 的局部坐标残差。CPU 冒烟/小规模训练：

```bash
$PYTHON train_gru_predictor.py \
  --output artifacts/gru_predictor_cpu_v1 \
  --train-episodes 4 \
  --validation-episodes 2 \
  --epochs 10 \
  --device cpu

$PYTHON analyze_prediction.py \
  --warmup-seconds 4 \
  --gru-checkpoint artifacts/gru_predictor_cpu_v1/best.pt \
  --output artifacts/prediction/imm_gru_high_seed20260723.json
```

当前 GRU 在唯一真实 seed 上与纯 IMM 基本持平，所以不进入真实控制
默认路径。带发布指令限幅和飞控滞后的 APN 离线配对命令：

```bash
$PYTHON evaluate_guidance.py \
  --episodes 10 \
  --seed 73000 \
  --modes classic apn \
  --output artifacts/evaluations/apn_limited_seed73000_n10.json
```

该组结果为 classic 0.7、IMM+APN 3.3，9 局改善、1 局持平、0 局变差。
绝对分数不直接代表实机得分，只说明候选达到真实 A/B 门槛。

## 真实 ROS 接入

`ros_controller.py` 只使用比赛公开 ROS 接口，提供三种模式：

- `observe`：只订阅和记录，不发布控制、不解锁；
- `classic`：匈牙利分配和经典提前量制导；
- `mappo`：在经典制导上叠加训练后的受限 MAPPO 残差。

当前镜像的全局 `python3` 是缺少 PyYAML 的 Conda 3.13，而比赛二进制
扩展按 Python 3.10 编译；RflySim 同时拒绝 root 身份。包装脚本选择
`demo` Python 3.10，并让平台以镜像配置好的 `ubuntu` 非 root 用户运行：

```bash
cd /root/gpufree-data/fly/zy_fly_10v10
bash run_platform.sh --scene 10v10 --diff low --seed 20260723
```

启动比赛平台后先做只读验证：

```bash
cd /root/gpufree-data/fly/zy_fly_10v10
bash run_ros.sh \
  --mode observe \
  --difficulty low \
  --duration 20 \
  --log artifacts/ros/observe_low.jsonl
```

确认 10 架拦截机与 10 个目标均有数据后，再运行经典基线：

```bash
bash run_ros.sh \
  --mode classic \
  --difficulty low \
  --auto-arm \
  --duration 180 \
  --max-speed 20 \
  --max-acceleration 5 \
  --log artifacts/ros/classic_low.jsonl
```

真实基线稳定后，High 候选使用同一 seed 做配对 A/B：

```bash
bash run_ros.sh \
  --mode mappo \
  --checkpoint artifacts/mappo_high_real_v1/best_offline.pt \
  --difficulty high \
  --residual-scale 0.5 \
  --auto-arm \
  --duration 180 \
  --max-speed 30 \
  --log artifacts/ros/mappo_high_seedNAME.jsonl
```

High-v2 IMM+APN 使用独立显式开关，先保持旧 MAPPO 关闭：

```bash
bash run_ros.sh \
  --mode classic \
  --guidance apn \
  --difficulty high \
  --auto-arm \
  --duration 180 \
  --max-speed 30 \
  --max-acceleration 5 \
  --terminal-distance 15 \
  --terminal-gain 2 \
  --log-interval 0.1 \
  --log artifacts/ros/apn_high_seedNAME.jsonl
```

`--guidance apn` 当前只允许 High + classic 模式。没有该参数时仍走原
Alpha-Beta + classic 路径；预测或制导实验不会影响 low/mid 默认行为。

接入节点会在线估计雷达目标速度、将本机局部 ENU 转为世界 ENU，
持续以 20 Hz 发布速度设定点，并限制速度和加速度。安全区中心由
本局实时收到的拦截机出生点计算，不含预设位置。

## 后续工作

1. 对同一 High seed 先跑当前 classic，再重新启动平台跑 IMM+APN；
2. 先完成 1 个开发 seed，检查 10 Hz 超时、指令方向和最近距离；
3. 安全后扩展到 3--5 个实机 seed，再决定 APN 参数；
4. APN 真实收益确认后实现可达性分配，再训练新底座上的 R-MAPPO；
5. 任一候选退化或控制异常时，立即回退无 `--guidance` 的 classic。
