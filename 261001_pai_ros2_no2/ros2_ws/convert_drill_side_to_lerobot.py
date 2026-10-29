#!/usr/bin/env python3
"""Convert drill_side ROS 2 bags to a state-only LeRobot dataset."""

import argparse
from bisect import bisect_right
import re
from pathlib import Path

import numpy as np
import torch
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


JOINT_NAMES = (
    "crane_plus_joint1",
    "crane_plus_joint2",
    "crane_plus_joint3",
    "crane_plus_joint4",
    "crane_plus_joint_hand",
)
STATE_TOPIC = "/joint_states"
ARM_TOPIC = "/crane_plus_arm_controller/controller_state"
GRIPPER_TOPIC = "/crane_plus_gripper_controller/controller_state"
REQUIRED_TOPICS = (STATE_TOPIC, ARM_TOPIC, GRIPPER_TOPIC)


def make_typestore():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    message_file = Path(
        "/opt/ros/humble/share/control_msgs/msg/JointTrajectoryControllerState.msg"
    )
    if not message_file.is_file():
        raise FileNotFoundError(
            f"ROS message definition not found: {message_file}. "
            "Run the converter in the pai_ros2 container."
        )
    typestore.register(
        get_types_from_msg(
            message_file.read_text(encoding="utf-8"),
            "control_msgs/msg/JointTrajectoryControllerState",
        )
    )
    return typestore


def episode_sort_key(path):
    match = re.fullmatch(r"ep(\d+)", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def _ordered_values(names, values, expected_names, topic, timestamp):
    if len(names) != len(values):
        raise ValueError(
            f"{topic} at {timestamp}: {len(names)} joint names but "
            f"{len(values)} positions"
        )
    by_name = dict(zip(names, values))
    missing = [name for name in expected_names if name not in by_name]
    if missing:
        raise ValueError(f"{topic} at {timestamp}: missing joints {missing}")
    return np.asarray([by_name[name] for name in expected_names], dtype=np.float32)


def read_episode(episode_dir):
    records = {topic: [] for topic in REQUIRED_TOPICS}
    typestore = make_typestore()
    with Reader(episode_dir) as reader:
        connections = [c for c in reader.connections if c.topic in records]
        found_topics = {c.topic for c in connections}
        missing_topics = set(REQUIRED_TOPICS) - found_topics
        if missing_topics:
            raise ValueError(f"{episode_dir}: missing topics {sorted(missing_topics)}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            if connection.topic == STATE_TOPIC:
                values = _ordered_values(
                    msg.name, msg.position, JOINT_NAMES, connection.topic, timestamp
                )
            elif connection.topic == ARM_TOPIC:
                values = _ordered_values(
                    msg.joint_names,
                    msg.reference.positions,
                    JOINT_NAMES[:4],
                    connection.topic,
                    timestamp,
                )
            else:
                values = _ordered_values(
                    msg.joint_names,
                    msg.reference.positions,
                    JOINT_NAMES[4:],
                    connection.topic,
                    timestamp,
                )
            records[connection.topic].append((timestamp, values))

    for topic, entries in records.items():
        if not entries:
            raise ValueError(f"{episode_dir}: topic {topic} has no messages")
        entries.sort(key=lambda entry: entry[0])

    return records


def sample_hold(records, target_times):
    times = [timestamp for timestamp, _ in records]
    values = np.stack([value for _, value in records])
    indices = [max(0, bisect_right(times, t) - 1) for t in target_times]
    return values[indices]


def resample_episode(records, fps):
    start = max(records[topic][0][0] for topic in REQUIRED_TOPICS)
    end = min(records[topic][-1][0] for topic in REQUIRED_TOPICS)
    if end <= start:
        raise ValueError(f"Topics have no common time interval ({start} to {end})")

    duration_frames = (end - start) * fps
    frame_count = max(
        1, (duration_frames + 1_000_000_000 - 1) // 1_000_000_000
    )
    if frame_count < 1:
        raise ValueError("No samples fall within the common time interval")
    target_times = start + (
        np.arange(frame_count, dtype=np.int64) * 1_000_000_000 // fps
    )

    state = sample_hold(records[STATE_TOPIC], target_times)
    arm = sample_hold(records[ARM_TOPIC], target_times)
    gripper = sample_hold(records[GRIPPER_TOPIC], target_times)
    action = np.concatenate((arm, gripper), axis=1)
    if state.shape[1] != len(JOINT_NAMES) or action.shape[1] != len(JOINT_NAMES):
        raise ValueError(
            f"Expected 5 state/action values; got {state.shape[1]} and "
            f"{action.shape[1]}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State/action data contains NaN or infinite values")
    return state, action


def trim_initial_stationary(
    state,
    action,
    fps,
    movement_threshold,
    confirm_frames,
    preroll_frames,
):
    baseline_frames = min(len(action), fps)
    baseline = np.median(action[:baseline_frames], axis=0)
    movement = np.linalg.norm(action - baseline, axis=1) > movement_threshold
    movement_start = next(
        (
            index
            for index in range(len(movement) - confirm_frames + 1)
            if movement[index : index + confirm_frames].all()
        ),
        None,
    )
    if movement_start is None:
        raise ValueError(
            "No sustained movement detected in action data "
            f"(threshold={movement_threshold}, frames={confirm_frames})"
        )

    start = max(0, movement_start - preroll_frames)
    return state[start:], action[start:], start


def trim_terminal_stationary(
    state,
    action,
    fps,
    movement_threshold,
    confirm_frames,
    postroll_frames,
):
    baseline_frames = min(len(action), fps)
    baseline = np.median(action[-baseline_frames:], axis=0)
    movement = np.linalg.norm(action - baseline, axis=1) > movement_threshold
    movement_end = next(
        (
            index + confirm_frames - 1
            for index in range(len(movement) - confirm_frames, -1, -1)
            if movement[index : index + confirm_frames].all()
        ),
        None,
    )
    if movement_end is None:
        raise ValueError(
            "No sustained movement detected before the terminal baseline "
            f"(threshold={movement_threshold}, frames={confirm_frames})"
        )

    end = min(len(action), movement_end + postroll_frames + 1)
    return state[:end], action[:end], len(action) - end


def main():
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=workspace / "dataset" / "drill_side",
        help="Directory containing one rosbag2 folder per episode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace / "dataset" / "drill_side_lerobot_reference",
        help="New output directory (must not already exist)",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--task", default="drill_side")
    parser.add_argument("--repo-id", default="crane_plus_drill_side")
    parser.add_argument(
        "--trim-initial-stationary",
        action="store_true",
        help="Remove the initial stationary segment based on reference action movement",
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=0.02,
        help="L2 action displacement in radians used to detect movement",
    )
    parser.add_argument(
        "--movement-confirm-frames",
        type=int,
        default=3,
        help="Consecutive frames above threshold required to detect movement",
    )
    parser.add_argument(
        "--movement-preroll-frames",
        type=int,
        default=3,
        help="Frames retained immediately before detected movement",
    )
    parser.add_argument(
        "--trim-terminal-stationary",
        action="store_true",
        help="Remove the final stationary segment based on reference action movement",
    )
    parser.add_argument(
        "--movement-postroll-frames",
        type=int,
        default=3,
        help="Frames retained immediately after detected movement",
    )
    parser.add_argument(
        "--include-phase",
        action="store_true",
        help="Append normalized episode progress to observation.environment_state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report all episodes without creating a dataset",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be a positive integer")
    if args.movement_threshold <= 0:
        parser.error("--movement-threshold must be positive")
    if args.movement_confirm_frames <= 0:
        parser.error("--movement-confirm-frames must be positive")
    if args.movement_preroll_frames < 0:
        parser.error("--movement-preroll-frames cannot be negative")
    if args.movement_postroll_frames < 0:
        parser.error("--movement-postroll-frames cannot be negative")
    if args.include_phase and not (
        args.trim_initial_stationary and args.trim_terminal_stationary
    ):
        parser.error(
            "--include-phase requires trimming both initial and terminal "
            "stationary segments"
        )
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    episode_dirs = sorted(
        {path.parent for path in input_dir.rglob("*.db3")}, key=episode_sort_key
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No .db3 rosbag files found under {input_dir}")
    if not args.dry_run and output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new output path; existing data will not be overwritten."
        )

    episodes = []
    total_frames = 0
    for episode_dir in episode_dirs:
        records = read_episode(episode_dir)
        state, action = resample_episode(records, args.fps)
        removed_start_frames = 0
        removed_end_frames = 0
        if args.trim_initial_stationary:
            state, action, removed_start_frames = trim_initial_stationary(
                state,
                action,
                args.fps,
                args.movement_threshold,
                args.movement_confirm_frames,
                args.movement_preroll_frames,
            )
        if args.trim_terminal_stationary:
            state, action, removed_end_frames = trim_terminal_stationary(
                state,
                action,
                args.fps,
                args.movement_threshold,
                args.movement_confirm_frames,
                args.movement_postroll_frames,
            )
        episodes.append((episode_dir, state, action))
        total_frames += len(state)
        print(
            f"{episode_dir.name}: {len(state)} frames, "
            f"state/action shape {state.shape}/{action.shape}, "
            f"removed {removed_start_frames} initial and "
            f"{removed_end_frames} terminal frames"
        )

    print(
        f"Validated {len(episodes)} episodes ({total_frames} frames total) "
        f"at {args.fps} fps."
    )
    if args.dry_run:
        return

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    feature_names = list(JOINT_NAMES)
    state_names = feature_names + (["progress"] if args.include_phase else [])
    environment_names = feature_names + (["progress"] if args.include_phase else [])
    state_shape = (len(state_names),)
    environment_shape = (len(environment_names),)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": state_shape,
            "names": state_names,
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": environment_shape,
            "names": environment_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (5,),
            "names": feature_names,
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=output_dir,
        robot_type="crane_plus",
        features=features,
        use_videos=False,
    )
    for index, (episode_dir, state, action) in enumerate(episodes):
        frame_count = len(state)
        for frame_index, (observation, command) in enumerate(zip(state, action)):
            policy_state = observation
            environment_state = observation
            if args.include_phase:
                progress = frame_index / max(frame_count - 1, 1)
                policy_state = np.append(observation, np.float32(progress))
                environment_state = np.append(observation, np.float32(progress))
            dataset.add_frame(
                {
                    "observation.state": torch.from_numpy(policy_state.copy()),
                    "observation.environment_state": torch.from_numpy(
                        environment_state.copy()
                    ),
                    "action": torch.from_numpy(command.copy()),
                    "task": args.task,
                }
            )
        dataset.save_episode()
        print(f"Saved episode {index} from {episode_dir.name}")

    print(f"LeRobot dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
