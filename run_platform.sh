#!/bin/bash
set -euo pipefail

PLATFORM_ARGS=("$@")
set --
set +u
source /opt/ros/noetic/setup.bash
source /opt/rostrans/sdk/x86_64-u20.04-ros1-noetic/setup.bash
set -u
set -- "${PLATFORM_ARGS[@]}"

# 比赛的 Python 扩展按 Python 3.10 编译，demo Conda 同时具备 PyYAML。
# RflySim3D 拒绝 root 身份；云镜像为 ubuntu 用户授予了平台运行路径
# 的写权限，因此平台进程以 ubuntu 启动，参赛控制器仍在独立终端运行。
export PATH="/opt/conda/envs/demo/bin:$PATH"
RFLY_SDK=/home/ubuntu/PX4PSP/RflySimAPIs/RflySimSDK
export PYTHONPATH="$RFLY_SDK/ctrl:$RFLY_SDK/ue:$RFLY_SDK:${PYTHONPATH:-}"
RUNTIME_DIR=/tmp/zhuoyi-runtime-ubuntu
mkdir -p "$RUNTIME_DIR"
chown ubuntu:ubuntu "$RUNTIME_DIR"
chmod 700 "$RUNTIME_DIR"

# 早期以 root 运行 PX4 的失败尝试可能会留下 root 所有的 lock/socket，
# 之后 ubuntu 身份的正式启动既不能覆盖，也不会自动报到平台日志里。
# 仅在没有存活 PX4 时清理本平台固定的 1..20 号临时 IPC 文件。
if ! pgrep -x px4 >/dev/null 2>&1; then
  for instance_id in $(seq 1 20); do
    rm -f "/tmp/px4_lock-${instance_id}" "/tmp/px4-sock-${instance_id}"
  done
fi

exec sudo -E -u ubuntu env \
  HOME=/home/ubuntu \
  XDG_RUNTIME_DIR="$RUNTIME_DIR" \
  DISPLAY="${DISPLAY:-:20}" \
  PATH="$PATH" \
  PYTHONPATH="${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH:-}" \
  ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}" \
  ROS_DISTRO="${ROS_DISTRO:-noetic}" \
  ROS_VERSION="${ROS_VERSION:-1}" \
  CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}" \
  PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}" \
  /home/ubuntu/zhuoyi_cup/run.sh "$@"
