#!/usr/bin/env python3
"""Replay the median recorded drill_side action trajectory in ROS 2."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINT_NAMES = [
    "crane_plus_joint1",
    "crane_plus_joint2",
    "crane_plus_joint3",
    "crane_plus_joint4",
]
GRIPPER_JOINT_NAME = "crane_plus_joint_hand"


def load_reference(dataset_root, phase_duration):
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    fps = int(json.loads(info_path.read_text(encoding="utf-8"))["fps"])
    episode_files = sorted((root / "data" / "chunk-000").glob("episode_*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found under {root}")

    sample_count = round(phase_duration * fps) + 1
    target_phase = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    episode_actions = []
    initial_states = []
    for path in episode_files:
        frame = pd.read_parquet(path)
        states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        if states.ndim != 2 or states.shape[1] < 6 or actions.shape[1] != 5:
            raise ValueError(
                f"{path}: expected state shape (N, >=6) and action shape (N, 5), "
                f"got {states.shape} and {actions.shape}"
            )
        phase = states[:, 5]
        if (
            not np.isfinite(states).all()
            or not np.isfinite(actions).all()
            or np.any(np.diff(phase) < 0)
            or phase[0] > 1e-4
            or phase[-1] < 1.0 - 1e-4
        ):
            raise ValueError(f"{path}: invalid phase or non-finite trajectory data")
        resampled = np.stack(
            [
                np.interp(target_phase, phase, actions[:, joint])
                for joint in range(actions.shape[1])
            ],
            axis=1,
        )
        episode_actions.append(resampled)
        initial_states.append(states[0, :5])

    reference = np.median(np.stack(episode_actions), axis=0).astype(np.float32)
    initial_state = np.median(np.stack(initial_states), axis=0).astype(np.float32)
    return fps, reference, initial_state


class PhaseReplayNode(Node):
    def __init__(
        self, fps, reference, initial_state, phase_duration, start_tolerance, dry_run
    ):
        super().__init__("crane_plus_phase_replay")
        self.fps = fps
        self.reference = reference
        self.initial_state = initial_state
        self.phase_duration = phase_duration
        self.start_tolerance = start_tolerance
        self.dry_run = dry_run
        self.latest_state = None
        self.started_at = None
        self.finish_hold_started_at = None
        self.frame_index = 0
        self.timer = None
        self.arm_pub = self.create_publisher(
            JointTrajectory, "/crane_plus_arm_controller/joint_trajectory", 10
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory, "/crane_plus_gripper_controller/joint_trajectory", 10
        )
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

    def joint_state_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        names = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
        if all(name in name_to_pos for name in names):
            self.latest_state = np.asarray(
                [name_to_pos[name] for name in names], dtype=np.float32
            )

    def start_replay(self):
        if self.latest_state is None:
            raise RuntimeError("No complete /joint_states message received")
        error = np.max(np.abs(self.latest_state - self.initial_state))
        if error > self.start_tolerance:
            raise RuntimeError(
                "Gazebo pose is too far from the recorded start pose: "
                f"max joint error={error:.3f} rad, "
                f"allowed={self.start_tolerance:.3f} rad. "
                "Reset Gazebo to the start pose before replay."
            )
        self.started_at = time.monotonic()
        self.timer = self.create_timer(1.0 / self.fps, self.publish_action)
        self.get_logger().info(
            f"Starting median demonstration replay: {len(self.reference)} frames "
            f"at {self.fps} Hz"
        )

    def publish_action(self):
        elapsed = time.monotonic() - self.started_at
        phase = min(elapsed / self.phase_duration, 1.0)
        index = min(round(phase * (len(self.reference) - 1)), len(self.reference) - 1)
        action = self.reference[index]
        if not self.dry_run:
            self.arm_pub.publish(self.make_trajectory(ARM_JOINT_NAMES, action[:4]))
            self.gripper_pub.publish(
                self.make_trajectory([GRIPPER_JOINT_NAME], action[4:5])
            )
        if self.frame_index % self.fps == 0 or phase >= 1.0:
            self.get_logger().info(
                f"phase={phase:.3f} state={np.round(self.latest_state, 3)} "
                f"action={np.round(action, 3)}"
            )
        self.frame_index += 1
        if phase >= 1.0:
            if self.finish_hold_started_at is None:
                self.finish_hold_started_at = time.monotonic()
            elif time.monotonic() - self.finish_hold_started_at >= 2.0:
                self.timer.cancel()
                self.get_logger().info("Phase replay completed")
                rclpy.shutdown()

    @staticmethod
    def make_trajectory(names, positions):
        trajectory = JointTrajectory()
        trajectory.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = [float(position) for position in positions]
        point.time_from_start.nanosec = 100_000_000
        trajectory.points.append(point)
        return trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/workspace/ros2_ws/dataset/drill_side_lerobot_phase_state",
    )
    parser.add_argument("--phase-duration", type=float, default=57.025)
    parser.add_argument("--start-tolerance", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.phase_duration <= 0 or args.start_tolerance < 0:
        parser.error("--phase-duration must be positive and --start-tolerance non-negative")

    fps, reference, initial_state = load_reference(
        args.dataset_root, args.phase_duration
    )
    rclpy.init()
    node = PhaseReplayNode(
        fps,
        reference,
        initial_state,
        args.phase_duration,
        args.start_tolerance,
        args.dry_run,
    )
    try:
        for _ in range(10):
            if node.latest_state is not None:
                break
            rclpy.spin_once(node, timeout_sec=1.0)
        node.start_replay()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Phase replay interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
