import numpy as np
from scipy.spatial.transform import Rotation as R


def extract_features(joints, positions, rotations):
    """
    Extract feature matrix from positions and rotations for motion matching.

    Feature vector:
        Root future positions (x, z) at 10, 20, 30 frames ahead
        Root forward directions (x, z) at 10, 20, 30 frames ahead
        Left foot position (x, y, z), velocity (x, y, z)
        Right foot position (x, y, z), velocity (x, y, z)

    Feature matrix: (n_frames, 24)
    """

    ROOTIDX = 0
    n_frames = positions.shape[0]
    n_features = 24
    frames = np.arange(n_frames)
    features = np.zeros((n_frames, n_features), dtype=np.float32)

    # Global to local translation and y-rotation
    local_T = positions[:, ROOTIDX]
    local_R = R.from_euler("xyz", rotations[:, ROOTIDX])
    local_R = R.from_euler("y", local_R.as_euler("yxz")[:, 0])

    # Root trajectories and forward directions
    for i, offset in enumerate([10, 20, 30]):
        next_frames = frames + offset
        next_frames = np.clip(next_frames, 0, n_frames - 1)

        next_positions = positions[next_frames, ROOTIDX]
        next_positions -= local_T
        next_positions = local_R.inv().apply(next_positions)
        features[:, i * 2 : i * 2 + 2] = next_positions[:, [0, 2]]

        next_rotations = R.from_euler("xyz", rotations[next_frames, ROOTIDX])
        next_rotations = local_R.inv() * next_rotations
        # CHECKME: Is this correct?
        forward_directions = next_rotations.apply(np.array([0.0, 0.0, 1.0]))
        features[:, i * 2 + 6 : i * 2 + 8] = forward_directions[:, [0, 2]]

    # Foot positions and velocities
    for joint_name in ["LeftFoot", "RightFoot"]:
        joint_idx = joints.index(joint_name)
        position = positions[:, joint_idx]
        local_position = local_R.inv().apply(position - local_T)
        velocity = np.gradient(positions[:, joint_idx], axis=0)
        local_velocity = local_R.inv().apply(velocity)
        if joint_name == "LeftFoot":
            features[:, 12:18] = np.hstack((local_position, local_velocity))
        else:  # RIGHTFOOT
            features[:, 18:24] = np.hstack((local_position, local_velocity))

    return features
