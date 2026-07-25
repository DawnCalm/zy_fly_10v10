#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rospy
import torch
from geometry_msgs.msg import PointStamped
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry

from zhuoyi_mappo.config import EnvConfig
from zhuoyi_mappo.residual import residual_action_to_world
from zhuoyi_mappo.runtime_core import GuidanceResult, build_guidance_inputs
from zhuoyi_mappo.tracking import (
    AlphaBetaTrack,
    TargetRetirementDetector,
    VelocityLimiter,
)
from zhuoyi_mappo.trainer import load_policy


FRAME_LOCAL_NED = 1
VELOCITY_ONLY_TYPEMASK = (
    1 << 0
    | 1 << 1
    | 1 << 2
    | 1 << 6
    | 1 << 7
    | 1 << 8
    | 1 << 11
)


def mavros_ns(uav_id: int) -> str:
    return "/mavros" if int(uav_id) == 1 else f"/mavros{int(uav_id)}"


def clip_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norms, 1.0e-8))
    return value * scale


class RosStateCache:
    def __init__(self, count: int, target_timeout: float, odom_timeout: float):
        self.count = int(count)
        self.target_timeout = float(target_timeout)
        self.odom_timeout = float(odom_timeout)
        self.lock = threading.RLock()

        self.origins = np.zeros((count, 3), dtype=np.float32)
        self.origin_seen = np.zeros(count, dtype=bool)
        self.local_pos = np.zeros((count, 3), dtype=np.float32)
        self.agent_vel = np.zeros((count, 3), dtype=np.float32)
        self.odom_seen = np.zeros(count, dtype=bool)
        self.odom_time = np.full(count, -np.inf, dtype=np.float64)
        self.armed = np.zeros(count, dtype=bool)
        self.modes = [""] * count
        self.state_seen = np.zeros(count, dtype=bool)
        self.tracks = [AlphaBetaTrack() for _ in range(count)]
        self.target_ever_seen = np.zeros(count, dtype=bool)

    def origin_callback(self, message: PointStamped, index: int) -> None:
        with self.lock:
            self.origins[index] = (
                message.point.x,
                message.point.y,
                message.point.z,
            )
            self.origin_seen[index] = True

    def odom_callback(self, message: Odometry, index: int) -> None:
        now = time.monotonic()
        with self.lock:
            position = message.pose.pose.position
            velocity = message.twist.twist.linear
            self.local_pos[index] = (position.x, position.y, position.z)
            self.agent_vel[index] = (velocity.x, velocity.y, velocity.z)
            self.odom_seen[index] = True
            self.odom_time[index] = now

    def state_callback(self, message: State, index: int) -> None:
        with self.lock:
            self.armed[index] = bool(message.armed)
            self.modes[index] = str(message.mode)
            self.state_seen[index] = True

    def target_callback(self, message: PointStamped, index: int) -> None:
        now = time.monotonic()
        measurement = np.array(
            (message.point.x, message.point.y, message.point.z),
            dtype=np.float64,
        )
        with self.lock:
            self.tracks[index].update(measurement, now)
            self.target_ever_seen[index] = True

    def ready_counts(self) -> Tuple[int, int, int, int]:
        with self.lock:
            return (
                int(self.origin_seen.sum()),
                int(self.odom_seen.sum()),
                int(self.state_seen.sum()),
                int(self.target_ever_seen.sum()),
            )

    def snapshot(self) -> Dict[str, object]:
        now = time.monotonic()
        with self.lock:
            agent_pos = self.origins + self.local_pos
            agent_active = (
                self.origin_seen
                & self.odom_seen
                & ((now - self.odom_time) <= self.odom_timeout)
            )
            target_pos = np.zeros((self.count, 3), dtype=np.float32)
            target_vel = np.zeros((self.count, 3), dtype=np.float32)
            target_age = np.full(self.count, np.inf, dtype=np.float32)
            for target_id, track in enumerate(self.tracks):
                target_pos[target_id], target_vel[target_id] = track.predict(now)
                target_age[target_id] = track.age(now)
            target_active = self.target_ever_seen & (
                target_age <= self.target_timeout
            )
            return {
                "monotonic": now,
                "origins": self.origins.copy(),
                "origin_seen": self.origin_seen.copy(),
                "local_pos": self.local_pos.copy(),
                "agent_pos": agent_pos.copy(),
                "agent_vel": self.agent_vel.copy(),
                "agent_active": agent_active.copy(),
                "armed": self.armed.copy(),
                "modes": list(self.modes),
                "target_pos": target_pos,
                "target_vel": target_vel,
                "target_active": target_active.copy(),
                "target_age": target_age,
                "target_ever_seen": self.target_ever_seen.copy(),
            }


class JsonlRecorder:
    def __init__(self, path: Optional[Path]):
        self.path = path
        self.file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", encoding="utf-8")

    def write(self, payload: Dict[str, object]) -> None:
        if self.file is None:
            return
        self.file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


class ZhuoyiRosController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.count = 10
        self.cache = RosStateCache(
            self.count, args.target_timeout, args.odom_timeout
        )
        self.publishers: Dict[int, rospy.Publisher] = {}
        self.command_lock = threading.RLock()
        self.commands = np.zeros((self.count, 3), dtype=np.float32)
        self.timer: Optional[rospy.Timer] = None
        self.control_started_at: Optional[float] = None
        self.previous_assignment = np.full(self.count, -1, dtype=np.int64)
        self.last_control_time: Optional[float] = None
        self.last_target_count = 0
        self.no_target_since: Optional[float] = None
        self.recorder = JsonlRecorder(args.log)
        self.retirement_detector = TargetRetirementDetector(
            self.count,
            moving_speed=args.escape_moving_speed,
            stationary_speed=args.escape_stationary_speed,
            stationary_grace=args.escape_stationary_grace,
        )

        self.policy = None
        self.device = torch.device("cpu")
        if args.mode == "mappo":
            self.policy, self.config, _, self.device = load_policy(
                args.checkpoint, args.device
            )
        else:
            self.config = EnvConfig()
        if args.max_speed is not None:
            self.config.interceptor_max_speed = float(args.max_speed)
        if args.terminal_distance is not None:
            if args.difficulty == "high":
                self.config.terminal_guidance_distance_high = float(
                    args.terminal_distance
                )
            else:
                self.config.terminal_guidance_distance = float(
                    args.terminal_distance
                )
        if args.terminal_gain is not None:
            if args.difficulty == "high":
                self.config.terminal_guidance_gain_high = float(
                    args.terminal_gain
                )
            else:
                self.config.terminal_guidance_gain = float(args.terminal_gain)
        self.limiter = VelocityLimiter(
            self.count,
            self.config.interceptor_max_speed,
            args.max_acceleration,
        )

    def setup_topics(self) -> None:
        for uav_id in range(1, self.count + 1):
            index = uav_id - 1
            namespace = mavros_ns(uav_id)
            rospy.Subscriber(
                f"/interceptor{uav_id}/origin",
                PointStamped,
                self.cache.origin_callback,
                callback_args=index,
                queue_size=5,
            )
            rospy.Subscriber(
                f"{namespace}/local_position/odom",
                Odometry,
                self.cache.odom_callback,
                callback_args=index,
                queue_size=10,
            )
            rospy.Subscriber(
                f"{namespace}/state",
                State,
                self.cache.state_callback,
                callback_args=index,
                queue_size=5,
            )
            rospy.Subscriber(
                f"/radar/target{uav_id}/position",
                PointStamped,
                self.cache.target_callback,
                callback_args=index,
                queue_size=10,
            )
            if self.args.mode != "observe":
                self.publishers[index] = rospy.Publisher(
                    f"{namespace}/setpoint_raw/local",
                    PositionTarget,
                    queue_size=10,
                )

    def wait_for_topics(self) -> bool:
        deadline = time.monotonic() + self.args.ready_timeout
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            origins, odoms, states, targets = self.cache.ready_counts()
            if origins == self.count and odoms == self.count and targets == self.count:
                rospy.loginfo(
                    "公开话题已就绪 origin=%d odom=%d state=%d target=%d",
                    origins,
                    odoms,
                    states,
                    targets,
                )
                return True
            rospy.loginfo_throttle(
                2.0,
                "等待公开话题 origin=%d/10 odom=%d/10 state=%d/10 target=%d/10",
                origins,
                odoms,
                states,
                targets,
            )
            rate.sleep()
        counts = self.cache.ready_counts()
        rospy.logwarn(
            "等待话题超时 origin=%d odom=%d state=%d target=%d",
            *counts,
        )
        return counts[0] == self.count and counts[1] == self.count

    @staticmethod
    def _velocity_message(velocity: np.ndarray) -> PositionTarget:
        message = PositionTarget()
        message.header.stamp = rospy.Time.now()
        message.coordinate_frame = FRAME_LOCAL_NED
        message.type_mask = VELOCITY_ONLY_TYPEMASK
        message.velocity.x = float(velocity[0])
        message.velocity.y = float(velocity[1])
        message.velocity.z = float(velocity[2])
        if float(np.linalg.norm(velocity[:2])) > 0.2:
            # API_README 约定：ENU 中 yaw=0 指东，逆时针为正。
            message.yaw = float(math.atan2(velocity[1], velocity[0]))
        else:
            message.yaw = 0.0
        return message

    def _publish_timer(self, _event: rospy.timer.TimerEvent) -> None:
        with self.command_lock:
            commands = self.commands.copy()
        for index, publisher in self.publishers.items():
            publisher.publish(self._velocity_message(commands[index]))

    def start_publisher(self) -> None:
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.args.publish_rate),
            self._publish_timer,
        )

    def arm_all(self) -> None:
        def arm_one(uav_id: int) -> Tuple[int, bool, str]:
            namespace = mavros_ns(uav_id)
            try:
                arm = rospy.ServiceProxy(
                    f"{namespace}/cmd/arming", CommandBool, persistent=False
                )
                mode = rospy.ServiceProxy(
                    f"{namespace}/set_mode", SetMode, persistent=False
                )
                arm_response = arm(True)
                mode_response = mode(custom_mode="OFFBOARD")
                ok = bool(getattr(arm_response, "success", False)) and bool(
                    getattr(mode_response, "mode_sent", False)
                )
                return uav_id, ok, ""
            except Exception as exc:  # ROS 服务不可用时平台可能外部解锁
                return uav_id, False, f"{type(exc).__name__}: {exc}"

        for retry in range(self.args.arm_retries):
            with ThreadPoolExecutor(max_workers=self.count) as executor:
                results = list(executor.map(arm_one, range(1, self.count + 1)))
            snapshot = self.cache.snapshot()
            armed_count = int(np.asarray(snapshot["armed"], dtype=bool).sum())
            offboard_count = sum(
                mode == "OFFBOARD" for mode in snapshot["modes"]
            )
            errors = [result[2] for result in results if result[2]]
            rospy.loginfo(
                "并发解锁 retry=%d armed=%d/10 offboard=%d/10 service_ok=%d/10",
                retry + 1,
                armed_count,
                offboard_count,
                sum(result[1] for result in results),
            )
            if armed_count == self.count and offboard_count == self.count:
                return
            if errors:
                rospy.logwarn_throttle(2.0, "解锁服务示例错误: %s", errors[0])
            rospy.sleep(0.4)

    def _policy_residual(self, guidance: GuidanceResult) -> np.ndarray:
        if self.policy is None:
            return np.zeros((self.count, 3), dtype=np.float32)
        with torch.no_grad():
            action, _, _ = self.policy.act(
                torch.as_tensor(
                    guidance.observation[None],
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.as_tensor(
                    guidance.global_state[None],
                    dtype=torch.float32,
                    device=self.device,
                ),
                deterministic=True,
            )
        return action[0].cpu().numpy()

    def _desired_command(
        self,
        snapshot: Dict[str, object],
        guidance: GuidanceResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        residual_action = self._policy_residual(guidance)
        residual_action *= float(self.args.residual_scale)
        residual_action[guidance.assignment < 0] = 0.0
        agent_pos = np.asarray(snapshot["agent_pos"], dtype=np.float32)
        target_pos = np.asarray(snapshot["target_pos"], dtype=np.float32)
        relative_position = np.zeros((self.count, 3), dtype=np.float32)
        for agent_id, target_id in enumerate(guidance.assignment):
            if target_id >= 0:
                relative_position[agent_id] = (
                    target_pos[target_id] - agent_pos[agent_id]
                )
        activation_distance, full_distance = (
            self.config.residual_gate_distances(self.args.difficulty)
        )
        residual_velocity, _ = residual_action_to_world(
            residual_action,
            guidance.guide_velocity,
            relative_position,
            self.config.interceptor_max_speed,
            self.config.residual_fraction(self.args.difficulty),
            activation_distance,
            full_distance,
        )
        desired = guidance.guide_velocity + residual_velocity
        desired = clip_norm(desired, self.config.interceptor_max_speed)
        local_pos = np.asarray(snapshot["local_pos"], dtype=np.float32)
        active = np.asarray(snapshot["agent_active"], dtype=bool)

        # 起飞阶段只使用速度指令，并逐渐放开水平速度。
        for agent_id in range(self.count):
            if not active[agent_id]:
                desired[agent_id] = 0.0
                continue
            altitude = float(local_pos[agent_id, 2])
            if altitude < self.args.takeoff_altitude:
                horizontal_scale = float(np.clip(altitude / 10.0, 0.0, 1.0))
                desired[agent_id, :2] *= horizontal_scale
                desired[agent_id, 2] = max(
                    float(desired[agent_id, 2]), self.args.climb_speed
                )
        return desired, residual_action

    def _record(
        self,
        snapshot: Dict[str, object],
        guidance: Optional[GuidanceResult],
        desired: Optional[np.ndarray],
        residual: Optional[np.ndarray],
    ) -> None:
        payload: Dict[str, object] = {
            "wall_time": time.time(),
            "mode": self.args.mode,
            "difficulty": self.args.difficulty,
            "origins": np.asarray(snapshot["origins"]).round(4).tolist(),
            "local_pos": np.asarray(snapshot["local_pos"]).round(4).tolist(),
            "agent_pos": np.asarray(snapshot["agent_pos"]).round(4).tolist(),
            "agent_vel": np.asarray(snapshot["agent_vel"]).round(4).tolist(),
            "agent_active": np.asarray(snapshot["agent_active"]).astype(int).tolist(),
            "armed": np.asarray(snapshot["armed"]).astype(int).tolist(),
            "modes": snapshot["modes"],
            "target_pos": np.asarray(snapshot["target_pos"]).round(4).tolist(),
            "target_vel": np.asarray(snapshot["target_vel"]).round(4).tolist(),
            "target_active": np.asarray(snapshot["target_active"]).astype(int).tolist(),
            "target_age": np.asarray(snapshot["target_age"]).round(4).tolist(),
        }
        if guidance is not None:
            payload["assignment"] = guidance.assignment.tolist()
            payload["intercept_time"] = guidance.intercept_time.round(4).tolist()
            payload["guide_velocity"] = guidance.guide_velocity.round(4).tolist()
        if desired is not None:
            payload["desired_velocity"] = desired.round(4).tolist()
        if residual is not None:
            payload["residual_action"] = residual.round(4).tolist()
        if "target_active_control" in snapshot:
            payload["target_active_control"] = (
                np.asarray(snapshot["target_active_control"])
                .astype(int)
                .tolist()
            )
        payload["target_retired"] = self.retirement_detector.retired.astype(
            int
        ).tolist()
        with self.command_lock:
            payload["published_velocity"] = self.commands.round(4).tolist()
        self.recorder.write(payload)

    def observe_loop(self) -> None:
        started = time.monotonic()
        rate = rospy.Rate(self.args.control_rate)
        last_record = -np.inf
        while not rospy.is_shutdown():
            now = time.monotonic()
            snapshot = self.cache.snapshot()
            if now - last_record >= self.args.log_interval:
                target_vel = np.linalg.norm(
                    np.asarray(snapshot["target_vel"])[:, :2], axis=1
                )
                rospy.loginfo(
                    "observe agent=%d target=%d armed=%d target_speed=[%.2f, %.2f]",
                    int(np.asarray(snapshot["agent_active"]).sum()),
                    int(np.asarray(snapshot["target_active"]).sum()),
                    int(np.asarray(snapshot["armed"]).sum()),
                    float(target_vel.min()),
                    float(target_vel.max()),
                )
                self._record(snapshot, None, None, None)
                last_record = now
            if self.args.duration > 0 and now - started >= self.args.duration:
                return
            rate.sleep()

    def control_loop(self) -> None:
        self.start_publisher()
        rospy.loginfo("预热 10 路设定点流 %.1f 秒", self.args.prewarm)
        rospy.sleep(self.args.prewarm)
        if self.args.auto_arm:
            self.arm_all()
        else:
            rospy.logwarn("未启用 --auto-arm，仅发布控制设定点")

        self.control_started_at = time.monotonic()
        self.last_control_time = self.control_started_at
        rate = rospy.Rate(self.args.control_rate)
        last_record = -np.inf
        while not rospy.is_shutdown():
            now = time.monotonic()
            snapshot = self.cache.snapshot()
            target_speed = np.linalg.norm(
                np.asarray(snapshot["target_vel"], dtype=np.float32)[:, :2],
                axis=1,
            )
            retired_before = self.retirement_detector.retired.copy()
            retired = self.retirement_detector.update(
                target_speed,
                np.asarray(snapshot["target_active"], dtype=bool),
                now,
            )
            newly_retired = np.flatnonzero(retired & ~retired_before)
            if len(newly_retired):
                rospy.loginfo(
                    "目标在线退役(曾运动后持续静止): %s",
                    ",".join(str(int(index + 1)) for index in newly_retired),
                )
            effective_target_active = (
                np.asarray(snapshot["target_active"], dtype=bool) & ~retired
            )
            snapshot["target_active_control"] = effective_target_active.copy()
            origins = np.asarray(snapshot["origins"], dtype=np.float32)
            origin_mask = np.asarray(snapshot["origin_seen"], dtype=bool)
            if origin_mask.any():
                safe_center = origins[origin_mask].mean(axis=0)
            else:
                safe_center = np.zeros(3, dtype=np.float32)
            elapsed = now - self.control_started_at
            guidance = build_guidance_inputs(
                self.config,
                np.asarray(snapshot["agent_pos"]),
                np.asarray(snapshot["agent_vel"]),
                np.asarray(snapshot["agent_active"]),
                np.asarray(snapshot["target_pos"]),
                np.asarray(snapshot["target_vel"]),
                effective_target_active,
                safe_center,
                self.previous_assignment,
                elapsed,
                self.args.difficulty,
            )
            self.previous_assignment[:] = guidance.assignment
            desired, residual = self._desired_command(snapshot, guidance)
            dt = max(1.0e-3, now - self.last_control_time)
            self.last_control_time = now
            limited = self.limiter.update(
                desired, dt, np.asarray(snapshot["agent_active"])
            )
            with self.command_lock:
                self.commands[:] = limited

            target_count = int(effective_target_active.sum())
            if target_count > 0:
                self.last_target_count = target_count
                self.no_target_since = None
            elif np.asarray(snapshot["target_ever_seen"]).all():
                if self.no_target_since is None:
                    self.no_target_since = now
                elif now - self.no_target_since >= self.args.target_gone_grace:
                    rospy.loginfo("全部目标话题已超时，结束控制循环")
                    self._record(snapshot, guidance, desired, residual)
                    return

            if now - last_record >= self.args.log_interval:
                actual_speed_all = np.linalg.norm(
                    np.asarray(snapshot["agent_vel"]), axis=1
                )
                active_agents = np.asarray(snapshot["agent_active"], dtype=bool)
                actual_speed = actual_speed_all[active_agents]
                command_speed = np.linalg.norm(limited, axis=1)
                rospy.loginfo(
                    "control target=%d agent=%d armed=%d cmd_max=%.1f actual_max=%.1f",
                    target_count,
                    int(np.asarray(snapshot["agent_active"]).sum()),
                    int(np.asarray(snapshot["armed"]).sum()),
                    float(command_speed.max()),
                    float(actual_speed.max()) if len(actual_speed) else 0.0,
                )
                self._record(snapshot, guidance, desired, residual)
                last_record = now
            if self.args.duration > 0 and elapsed >= self.args.duration:
                rospy.loginfo("达到运行时长 %.1f 秒", self.args.duration)
                return
            rate.sleep()

    def shutdown(self) -> None:
        if self.args.mode != "observe" and self.publishers:
            with self.command_lock:
                self.commands[:] = 0.0
            rate = rospy.Rate(self.args.publish_rate)
            for _ in range(5):
                self._publish_timer(None)
                rate.sleep()
        if self.timer is not None:
            self.timer.shutdown()
        self.recorder.close()

    def run(self) -> int:
        self.setup_topics()
        ready = self.wait_for_topics()
        if not ready:
            rospy.logerr("origin/odom 未全部就绪，拒绝进入控制")
            return 2
        try:
            if self.args.mode == "observe":
                self.observe_loop()
            else:
                self.control_loop()
        finally:
            self.shutdown()
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="卓翼杯 10v10 真实 ROS 接入节点")
    parser.add_argument(
        "--mode", choices=("observe", "classic", "mappo"), default="observe"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--difficulty", choices=("low", "mid", "high"), default="low")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--auto-arm", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--target-timeout", type=float, default=0.8)
    parser.add_argument("--odom-timeout", type=float, default=1.0)
    parser.add_argument("--target-gone-grace", type=float, default=2.0)
    parser.add_argument("--escape-moving-speed", type=float, default=5.0)
    parser.add_argument("--escape-stationary-speed", type=float, default=0.5)
    parser.add_argument("--escape-stationary-grace", type=float, default=1.5)
    parser.add_argument("--publish-rate", type=float, default=20.0)
    parser.add_argument("--control-rate", type=float, default=10.0)
    parser.add_argument("--log-interval", type=float, default=0.5)
    parser.add_argument("--prewarm", type=float, default=2.0)
    parser.add_argument("--arm-retries", type=int, default=5)
    parser.add_argument("--max-speed", type=float)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="MAPPO 残差额外缩放；0 等价于相同配置下的纯经典制导",
    )
    parser.add_argument("--max-acceleration", type=float, default=5.0)
    parser.add_argument("--takeoff-altitude", type=float, default=30.0)
    parser.add_argument("--climb-speed", type=float, default=3.0)
    parser.add_argument(
        "--terminal-distance",
        type=float,
        help="覆盖当前难度的末制导启用距离，便于同场景 A/B 测试",
    )
    parser.add_argument(
        "--terminal-gain",
        type=float,
        help="覆盖当前难度的末制导位置闭合增益，便于同场景 A/B 测试",
    )
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(rospy.myargv()[1:])
    if args.mode == "mappo" and args.checkpoint is None:
        parser.error("--mode mappo 必须提供 --checkpoint")
    if args.residual_scale < 0.0:
        parser.error("--residual-scale 必须大于等于 0")
    return args


def main() -> int:
    args = parse_args()
    rospy.init_node("zy_fly_10v10_controller", anonymous=False)
    controller = ZhuoyiRosController(args)
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
