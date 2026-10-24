#!/usr/bin/env python3
"""ドリル先端を base_link 座標の XYZ[m] へ MoveIt で動かす。

使い方（シミュレーション起動中に）:
  source /opt/ros/humble/setup.bash
  source /workspace/ros2_ws/install/local_setup.bash
  python3 /workspace/ros2_ws/move_to_xyz.py 0.15 0.0 0.05
  python3 /workspace/ros2_ws/move_to_xyz.py 0.20 0.05 0.03 --pitch-deg 150 --plan-only

流れ: 4関節の数値IK(crane_kinematics.py) → MoveIt /move_action に関節ゴールを渡す
      → 衝突を避ける経路計画と実行 → TFでドリル先端の到達位置を検証
"""
import argparse
import math
import sys

import numpy as np
import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

import crane_kinematics as kin

JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
BASE_FRAME = 'base_link'
LINK4_FRAME = 'crane_plus_link4'
DRILL_FRAME = 'drill_body'


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class XYZMover(Node):
    def __init__(self, drill_length):
        super().__init__('move_to_xyz')
        self._half = drill_length / 2.0
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._client = ActionClient(self, MoveGroup, '/move_action')

    def _lookup(self, target, source, timeout=10.0):
        end = self.get_clock().now() + Duration(seconds=timeout)
        while self.get_clock().now() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._tf.can_transform(target, source, rclpy.time.Time()):
                return self._tf.lookup_transform(target, source, rclpy.time.Time())
        raise RuntimeError(f'TF {target} -> {source} が取得できません')

    def tip_in_link4(self):
        """URDFのドリル取り付け位置から、link4座標でのドリル先端位置を求める。"""
        t = self._lookup(LINK4_FRAME, DRILL_FRAME).transform
        rot = quat_to_matrix(t.rotation)
        if not np.allclose(rot, np.eye(3), atol=1e-3):
            self.get_logger().warn('ドリルがlink4と異なる向きで取り付けられています。IKの前提(軸=link4のz)とずれます')
        return np.array([t.translation.x, t.translation.y, t.translation.z]) + rot @ [0, 0, self._half]

    def tip_in_base(self):
        t = self._lookup(BASE_FRAME, DRILL_FRAME).transform
        rot = quat_to_matrix(t.rotation)
        return np.array([t.translation.x, t.translation.y, t.translation.z]) + rot @ [0, 0, self._half]

    def move_to(self, xyz, pitch, plan_only=False, speed=0.3):
        tip_offset = self.tip_in_link4()
        q = kin.solve_ik(xyz, pitch=pitch, tip_in_link4=tip_offset)
        if q is None:
            self.get_logger().error(
                f'XYZ={list(xyz)} はこの姿勢(pitch={math.degrees(pitch):.0f}°)では届きません。'
                'XYZかpitchを変えてください')
            return False
        self.get_logger().info('IK解: ' + ', '.join(f'{v:.3f}' for v in q) + ' [rad]')

        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('/move_action が見つかりません。シミュレーションを起動してください')
            return False

        request = MotionPlanRequest()
        request.group_name = 'arm'
        request.num_planning_attempts = 10
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = speed
        request.max_acceleration_scaling_factor = speed
        constraints = Constraints()
        for name, value in zip(JOINT_NAMES, q):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        request.goal_constraints.append(constraints)

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only = plan_only

        send = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        handle = send.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('MoveItにゴールが拒否されました')
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        code = result_future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'MoveIt失敗: error_code={code}')
            return False

        if plan_only:
            self.get_logger().info('計画のみ成功（実行はしていません）')
            return True
        # 実行直後はTFが更新されきっていないことがあるため少し待つ
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
        tip = self.tip_in_base()
        err = np.linalg.norm(tip - np.asarray(xyz))
        self.get_logger().info(
            f'到達位置: ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) m  目標との差: {err * 1000:.1f} mm')
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('xyz', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='ドリル先端の目標位置[m]（base_link座標）')
    parser.add_argument('--pitch-deg', type=float, default=150.0,
                        help='ドリル軸の傾き[deg]。180=真下、90=水平（既定: 150）')
    parser.add_argument('--drill-length', type=float, default=0.09,
                        help='URDFのドリル円柱の長さ[m]（既定: 0.09）')
    parser.add_argument('--speed', type=float, default=0.3, help='速度・加速度のスケール(0-1)')
    parser.add_argument('--plan-only', action='store_true', help='計画だけ行い、実行しない')
    args = parser.parse_args()

    rclpy.init()
    node = XYZMover(args.drill_length)
    try:
        ok = node.move_to(args.xyz, math.radians(args.pitch_deg), args.plan_only, args.speed)
    except RuntimeError as e:
        node.get_logger().error(str(e))
        ok = False
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
