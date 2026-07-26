#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
CONTROLLER_ARGS=("$@")
set --
set +u
source /opt/ros/noetic/setup.bash
source /opt/rostrans/sdk/x86_64-u20.04-ros1-noetic/setup.bash
set -u
export ROS_LOG_DIR="$SCRIPT_DIR/artifacts/ros/ros_logs"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$ROS_LOG_DIR"
set -- "${CONTROLLER_ARGS[@]}"
exec /opt/conda/envs/demo/bin/python -u "$SCRIPT_DIR/ros_controller.py" "$@"
