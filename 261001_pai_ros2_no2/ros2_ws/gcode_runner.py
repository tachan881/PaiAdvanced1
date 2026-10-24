#!/usr/bin/env python3
"""Gコード(.nc)を解釈して CRANE+ V2 のドリル先端を動かす。

使い方（シミュレーション起動中に）:
  source /opt/ros/humble/setup.bash
  source /workspace/ros2_ws/install/local_setup.bash
  cd /workspace/ros2_ws
  python3 gcode_runner.py sample_drill.nc --dry-run   # 到達可否と所要時間だけ確認
  python3 gcode_runner.py sample_drill.nc             # 実行

座標: Gコードの X Y Z[mm] は「ワーク座標」。--origin で、そのワーク原点が
base_link座標のどこか[m]を指定する（既定はワーク上面の中心）。

処理の流れ:
  1. 全移動を細かい点列に分割し、各点をIKで解く（1点でも届かなければ何も動かさず終了）
  2. MoveItの衝突回避で、ワーク原点の真上(--safe-z)へ移動
  3. 点列を1本の関節軌道として FollowJointTrajectory で実行
     （G00は --rapid-speed、G01は F の速さ。関節速度が上限を超える区間は自動で遅くする）
"""
import argparse
import math
import sys

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint

import crane_kinematics as kin
import gcode_parser
from move_to_xyz import JOINT_NAMES, XYZMover

ARM_ACTION = '/crane_plus_arm_controller/follow_joint_trajectory'
MAX_JOINT_JUMP = 0.35   # 隣り合う点の間の関節変化の上限[rad]。超えたらIK解が飛んだとみなす
MAX_JOINT_SPEED = 1.0   # 実行時に許す関節速度[rad/s]（URDF上限は2.0）


def build_path(moves, start_mm, step_feed, step_rapid):
    """Moveのリストを [(x,y,z[mm], rapid, feed[mm/min], 元の行番号), ...] に分割する。"""
    path = []
    cur = np.asarray(start_mm, dtype=float)
    for m in moves:
        goal = np.array([m.x, m.y, m.z])
        dist = np.linalg.norm(goal - cur)
        if dist < 1e-9:
            continue
        step = step_rapid if m.rapid else step_feed
        n = max(1, int(math.ceil(dist / step)))
        for i in range(1, n + 1):
            p = cur + (goal - cur) * (i / n)
            path.append((p[0], p[1], p[2], m.rapid, m.feed, m.line))
        cur = goal
    return path


def solve_path(path, origin_m, pitch, tip_in_link4, start_q):
    """点列全体をIKで解く。届かない点があれば (None, エラー文) を返す。"""
    qs = []
    seed = start_q
    for x, y, z, _rapid, _feed, line in path:
        xyz = origin_m + np.array([x, y, z]) / 1000.0
        q = kin.solve_ik(xyz, pitch=pitch, tip_in_link4=tip_in_link4, seed=seed, n_seeds=10)
        if q is None:
            return None, (f'{line}行目付近の点 work=({x:.1f}, {y:.1f}, {z:.1f}) mm '
                          f'(base_link=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m) は届きません')
        if seed is not None and np.max(np.abs(q - seed)) > MAX_JOINT_JUMP:
            return None, (f'{line}行目付近で関節が {np.max(np.abs(q - seed)):.2f} rad 飛びます'
                          '（IK解の切り替わり）。点を近づけるか pitch を変えてください')
        qs.append(q)
        seed = q
    return qs, None


def timestamps(path, qs, start_q, start_mm, rapid_speed, feed_scale):
    """各点の time_from_start[s] を返す。工具速度と関節速度の両方の制約を満たす。"""
    times = []
    t = 0.0
    prev_q = start_q
    prev_p = np.asarray(start_mm, dtype=float)
    for (x, y, z, rapid, feed, _line), q in zip(path, qs):
        p = np.array([x, y, z])
        speed = rapid_speed if rapid else max(feed * feed_scale / 60.0, 0.1)  # mm/s
        dt = np.linalg.norm(p - prev_p) / speed
        dt = max(dt, float(np.max(np.abs(q - prev_q))) / MAX_JOINT_SPEED, 0.02)
        t += dt
        times.append(t)
        prev_q, prev_p = q, p
    return times


def execute(node, qs, times):
    client = ActionClient(node, FollowJointTrajectory, ARM_ACTION)
    if not client.wait_for_server(timeout_sec=10.0):
        node.get_logger().error(f'{ARM_ACTION} が見つかりません')
        return False
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINT_NAMES
    for q, t in zip(qs, times):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start = DurationMsg(sec=int(t), nanosec=int((t % 1.0) * 1e9))
        goal.trajectory.points.append(pt)

    send = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send)
    handle = send.result()
    if handle is None or not handle.accepted:
        node.get_logger().error('軌道ゴールが拒否されました')
        return False
    result = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result)
    code = result.result().result.error_code
    if code != FollowJointTrajectory.Result.SUCCESSFUL:
        node.get_logger().error(f'実行失敗: error_code={code}')
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('program', help='Gコードファイル(.nc)')
    ap.add_argument('--origin', type=float, nargs=3, default=[0.14, 0.0, 0.035],
                    metavar=('X', 'Y', 'Z'),
                    help='ワーク原点のbase_link座標[m]（既定: 0.14 0 0.035 = ワーク上面中心）')
    ap.add_argument('--pitch-deg', type=float, default=150.0, help='ドリル軸の傾き[deg]（既定150）')
    ap.add_argument('--drill-length', type=float, default=0.09, help='ドリル円柱の長さ[m]')
    ap.add_argument('--safe-z', type=float, default=20.0,
                    help='開始前にワーク原点の上空へ移動する高さ[mm]（既定20）')
    ap.add_argument('--rapid-speed', type=float, default=30.0, help='G00の工具速度[mm/s]')
    ap.add_argument('--feed-scale', type=float, default=1.0, help='G01の送り速度の倍率')
    ap.add_argument('--step', type=float, default=2.0, help='G01の分割間隔[mm]')
    ap.add_argument('--dry-run', action='store_true', help='IKと所要時間の確認のみ（動かさない）')
    args = ap.parse_args()

    with open(args.program, encoding='utf-8') as f:
        try:
            items = gcode_parser.parse(f.read())
        except gcode_parser.GCodeError as e:
            print(f'Gコードエラー: {e}')
            return 1
    moves = [i for i in items if isinstance(i, gcode_parser.Move)]
    events = [i for i in items if isinstance(i, gcode_parser.Event)]
    if not moves:
        print('移動指令がありません')
        return 1
    print(f'{len(moves)} 個の移動、{len(events)} 個の補助指令を読み込みました')

    origin = np.array(args.origin)
    pitch = math.radians(args.pitch_deg)
    start_mm = np.array([0.0, 0.0, args.safe_z])

    rclpy.init()
    node = XYZMover(args.drill_length)
    ok = False
    try:
        tip = node.tip_in_link4()
        path = build_path(moves, start_mm, args.step, 5.0)
        # 開始点(ワーク原点の上空)のIKを起点にして全体を解く
        start_xyz = origin + start_mm / 1000.0
        start_q = kin.solve_ik(start_xyz, pitch=pitch, tip_in_link4=tip, n_seeds=30)
        if start_q is None:
            node.get_logger().error('開始点（ワーク原点の上空）に届きません。--origin / --safe-z を見直してください')
            return 1
        node.get_logger().info(f'{len(path)} 点のIKを解いています...')
        qs, err = solve_path(path, origin, pitch, tip, start_q)
        if qs is None:
            node.get_logger().error(err)
            return 1
        times = timestamps(path, qs, start_q, start_mm, args.rapid_speed, args.feed_scale)
        node.get_logger().info(f'全点到達可能。所要時間 約 {times[-1]:.1f} 秒')
        for e in events:
            node.get_logger().info(f'(補助指令 {e.text}: {e.line}行目 / 回転はシミュレーション未対応のため記録のみ)')
        if args.dry_run:
            node.get_logger().info('--dry-run のため実行しません')
            return 0

        node.get_logger().info('開始点の上空へ移動します（MoveIt・衝突回避）')
        if not node.move_to(start_xyz, pitch):
            return 1
        node.get_logger().info('Gコードを実行します')
        ok = execute(node, qs, times)
        if ok:
            tip_now = node.tip_in_base()
            last = origin + np.array(path[-1][:3]) / 1000.0
            node.get_logger().info(
                f'完了。最終位置との差 {np.linalg.norm(tip_now - last) * 1000:.1f} mm')
    except (RuntimeError, gcode_parser.GCodeError) as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
