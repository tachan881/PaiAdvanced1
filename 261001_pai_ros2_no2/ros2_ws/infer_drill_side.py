#!/usr/bin/env python3
"""infer.py variant matching the joint order of drill_side_lerobot_reference (joint1..4, hand)."""
import argparse
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot.common.policies.act.modeling_act import ACTPolicy

ARM_JOINT_NAMES = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'


class InferenceNode(Node):
    def __init__(self, model_path, dry_run, phase_duration, traj_time=0.1, n_action_steps=None):
        super().__init__('crane_plus_act_infer')
        self.dry_run = dry_run
        self.phase_duration = phase_duration
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.traj_time = traj_time
        self.policy = ACTPolicy.from_pretrained(model_path)
        if n_action_steps is not None:
            self.policy.config.n_action_steps = n_action_steps
        self.policy.eval()
        self.policy.to(torch.device('cpu'))
        self.policy.reset()
        self.latest_joint_state = None
        self.start_time = None
        self.create_timer(0.05, self.publish_action)  # 20Hz, same as the dataset

    def joint_state_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        pos = [name_to_pos[j] for j in ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME] if j in name_to_pos]
        if len(pos) == 5:
            self.latest_joint_state = np.array(pos, dtype=np.float32)

    def publish_action(self):
        if self.latest_joint_state is None:
            return
        if self.start_time is None:
            self.start_time = time.monotonic()
        phase = np.float32(
            min((time.monotonic() - self.start_time) / self.phase_duration, 1.0)
        )
        # 6-dim models (phase_state) take progress; 5-dim models (e.g. drill_side_nc_act) do not
        if self.policy.config.robot_state_feature.shape[0] == 6:
            policy_state = np.append(self.latest_joint_state, phase)
        else:
            policy_state = self.latest_joint_state
        state = torch.tensor(policy_state, dtype=torch.float32).unsqueeze(0)
        environment_state = state
        obs = {
            'observation.state': state,
            'observation.environment_state': environment_state,
        }
        with torch.no_grad():
            action = self.policy.select_action(obs).cpu().numpy().flatten()
        print('phase :', round(float(phase), 3))
        print('state :', np.round(self.latest_joint_state, 3))
        print('action:', np.round(action, 3))
        if self.dry_run:
            return
        self.arm_pub.publish(self._traj(ARM_JOINT_NAMES, action[:4]))
        self.gripper_pub.publish(self._traj([GRIPPER_JOINT_NAME], action[4:5]))

    def _traj(self, names, positions):
        jt = JointTrajectory()
        jt.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.time_from_start.nanosec = int(self.traj_time * 1e9)
        jt.points.append(pt)
        return jt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument(
        '--phase-duration',
        type=float,
        default=57.025,
        help='Seconds for normalized progress to advance from 0 to 1',
    )
    parser.add_argument('--traj-time', type=float, default=0.1,
                        help='time_from_start [s] of each published trajectory point')
    parser.add_argument('--n-action-steps', type=int, default=None,
                        help='override the policy n_action_steps (default: value in the checkpoint)')
    parser.add_argument('--dry-run', action='store_true', help='print actions without publishing')
    args = parser.parse_args()
    if args.phase_duration <= 0:
        parser.error('--phase-duration must be positive')
    rclpy.init()
    node = InferenceNode(args.model_path, args.dry_run, args.phase_duration,
                         args.traj_time, args.n_action_steps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
