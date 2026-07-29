# 项目约束

本仓库用于卓翼杯 2026 10v10。排名首先看三局总命中数，其次看用时。

- 只使用公开 ROS 接口，不修改 `/home/ubuntu/zhuoyi_cup`。
- 禁止固定 seed、预设位置或轨迹硬编码。
- low/mid benchmark 为 `los_pn`，High benchmark 为 `los_pn_kf`；
  保留 `los_pn` 和 `classic` 显式回退。
- low/mid 已达到 10/10，未经回归不改速度、末制导和 deadline 参数。
- High 新方案必须同难度、同 seed、完整重启做 A/B。
- 未通过真实平台配对的方案不能设为默认。
- 只保留能影响比赛决策的代码、代表性日志和摘要。

工作前阅读 `docs/CURRENT_STATE.md`；历史失败方法见 `docs/HISTORY.md`。

验证命令：

```bash
/opt/conda/envs/demo/bin/python -m unittest discover -s tests -v
/opt/conda/envs/demo/bin/python -m compileall -q \
  zhuoyi_mappo ros_controller.py
bash -n run_platform.sh run_ros.sh
git diff --check
```
