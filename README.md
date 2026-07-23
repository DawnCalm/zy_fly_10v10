# zy_fly_10v10：卓翼杯 10v10 经典制导 + MAPPO

本目录是独立参赛算法工程，不读取或修改 `/home/ubuntu/zhuoyi_cup`
内的比赛文件，也不加载去年的模型权重。

当前阶段实现的是快速运动学训练环境：

- 10 架拦截机与 10 架靶机；
- 匈牙利算法按预计拦截时间进行一对一目标分配；
- 经典三维提前量制导提供基础速度；
- 共享 Actor 输出受限的三维速度残差；
- 集中式 Critic 读取完整 10v10 态势并从零训练；
- low/mid/high 三种随机化靶机运动；
- 训练 minibatch 始终保留同一时刻的联合智能体状态。

训练场景只使用随机生成的相对几何关系，不含比赛预设位置点。速度、
飞控响应时间和碰撞半径目前是可配置初值，接入真实 ROS 平台后需要
用实测数据重新标定。

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

正式训练初始命令：

```bash
$PYTHON train.py \
  --updates 300 \
  --num-envs 8 \
  --rollout-steps 256 \
  --output artifacts/mappo_fresh
```

从检查点继续训练：

```bash
$PYTHON train.py \
  --updates 100 \
  --num-envs 8 \
  --rollout-steps 256 \
  --resume artifacts/mappo_fresh/latest.pt \
  --output artifacts/mappo_fresh
```

`--updates` 表示本次命令继续运行多少次更新，日志和检查点中的 update
编号会接着已有模型递增。

训练会生成：

- `config.json`：环境和训练参数；
- `metrics.jsonl`：每次更新的损失和完赛统计；
- `latest.pt`：最新模型；
- `checkpoint_*.pt`：周期性检查点。

## 评估

先测不加 MAPPO 残差的经典基线：

```bash
$PYTHON evaluate.py --episodes 10
```

再用同样的随机种子评估训练模型：

```bash
$PYTHON evaluate.py \
  --checkpoint artifacts/mappo_fresh/latest.pt \
  --episodes 10
```

先比较平均拦截数和满拦截率，只有拦截数稳定后再比较回合步数。

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

MAPPO 已能从零训练并接入真实 ROS 控制器，但现有 checkpoint 在代理
环境中没有稳定超过经典基线，尚未用于正式平台成绩。下一步是用
`artifacts/ros/` 中的真实飞行日志继续标定 high 环境，再训练经典
制导上的受限残差策略。

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

真实基线稳定后才加载 MAPPO：

```bash
bash run_ros.sh \
  --mode mappo \
  --checkpoint artifacts/gpu_probe/latest.pt \
  --difficulty low \
  --auto-arm \
  --duration 180 \
  --log artifacts/ros/mappo_low.jsonl
```

接入节点会在线估计雷达目标速度、将本机局部 ENU 转为世界 ENU，
持续以 20 Hz 发布速度设定点，并限制速度和加速度。安全区中心由
本局实时收到的拦截机出生点计算，不含预设位置。

## 后续工作

1. 根据真实 high 日志继续标定目标机动与飞控响应；
2. 重新训练 MAPPO 残差策略，先在固定 seed 离线 A/B；
3. 只有 MAPPO 稳定超过各难度经典基线后，才进入真实平台测试；
4. 增加多 seed 的 low/mid/high 回归测试，避免对单一场景过拟合。
