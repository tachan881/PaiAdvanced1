#!/usr/bin/env python3
"""CRANE+ V2 のアームへ関節角の軌道を送る最小サンプル。

使い方:
  source /opt/ros/humble/setup.bash
  source /workspace/ros2_ws/install/local_setup.bash
  python3 /workspace/ros2_ws/send_joint_trajectory.py
  python3 /workspace/ros2_ws/send_joint_trajectory.py --joints 0.3 0.4 0.6 0.3 --time 4
"""
import argparse
import math
import sys

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

ACTION_NAME = '/crane_plus_arm_controller/follow_joint_trajectory'
JOINT_NAMES = [
    'crane_plus_joint1',
    'crane_plus_joint2',
    'crane_plus_joint3',
    'crane_plus_joint4',
]

# URDFの関節リミット [rad]（crane_plus.urdf.xacro と同じ値）
_HOME = math.radians(150.0)
JOINT_LIMITS = [
    (math.radians(0.0) - _HOME, math.radians(300.0) - _HOME),
    (math.radians(45.45) - _HOME, math.radians(252.20) - _HOME),
    (math.radians(3.52) - _HOME, math.radians(290.62) - _HOME),
    (math.radians(44.57) - _HOME, math.radians(251.32) - _HOME),
]

HOME_POSE = [0.0, 0.0, 0.0, 0.0]
SAMPLE_POSES = [
    [0.0, -0.8, -0.8, -0.8],
    [0.0, -0.8, -0.8, -0.8],
    [0.0, -0.8, -0.8, -0.8],
    HOME_POSE,
]

# 前方へ少し傾けるサンプル姿勢。実機やテーブルとの干渉は確認してから変更すること
# SAMPLE_POSES = [
#     [0.0, 0.5, 0.8, 0.3],
#     [0.5, 0.5, 0.8, 0.3],
#     [-0.5, 0.5, 0.8, 0.3],
#     HOME_POSE,
# ]


def check_limits(positions):
    for name, value, (low, high) in zip(JOINT_NAMES, positions, JOINT_LIMITS):
        if not low <= value <= high:
            raise ValueError(
                f'{name}={value:.3f} rad は範囲外です（{low:.3f} 〜 {high:.3f}）')


class TrajectorySender(Node):
    def __init__(self):
        super().__init__('send_joint_trajectory')
        self._client = ActionClient(self, FollowJointTrajectory, ACTION_NAME)

    def send(self, poses, seconds_per_pose):
        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f'{ACTION_NAME} が見つかりません。シミュレーションを起動してください')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        for i, pose in enumerate(poses, start=1):
            check_limits(pose)
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in pose]
            point.velocities = [0.0] * len(JOINT_NAMES)
            total = seconds_per_pose * i
            point.time_from_start = Duration(
                sec=int(total), nanosec=int((total % 1) * 1e9))
            goal.trajectory.points.append(point)

        self.get_logger().info(f'{len(poses)} 点の軌道を送信します')
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('ゴールが拒否されました')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        ok = result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        if ok:
            self.get_logger().info('動作完了')
        else:
            self.get_logger().error(
                f'失敗: error_code={result.error_code} {result.error_string}')
        return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--joints', type=float, nargs=4, metavar=('J1', 'J2', 'J3', 'J4'),
                        help='指定した1姿勢[rad]だけ動かす。省略時はサンプル姿勢を順に実行')
    parser.add_argument('--time', type=float, default=3.0,
                        help='1姿勢あたりの所要時間[s]（既定: 3.0）')
    args = parser.parse_args()

    poses = [args.joints] if args.joints else SAMPLE_POSES

    rclpy.init()
    node = TrajectorySender()
    try:
        ok = node.send(poses, args.time)
    except ValueError as e:
        node.get_logger().error(str(e))
        ok = False
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
