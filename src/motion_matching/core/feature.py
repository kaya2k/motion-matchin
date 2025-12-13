import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_matching.utils import extract_y_rotation


FEATURE_DIM = 27
OFFSETS = [10, 20, 30]


def extract_features(joints, positions, rotations):
    """
    Extract feature matrix from positions and rotations for motion matching.

    Feature vector:
        Root future positions and directions (x, z) at 10, 20, 30 frames ahead
        Root velocity (x, y, z)
        Left foot position (x, y, z), velocity (x, y, z)
        Right foot position (x, y, z), velocity (x, y, z)
    """

    ROOTIDX = 0
    n_frames = positions.shape[0]
    features = []
    velocities = np.gradient(positions, axis=0)

    for frame in range(n_frames):
        feature = []

        root_T = positions[frame, ROOTIDX]
        root_y_rotation = extract_y_rotation(rotations[frame, ROOTIDX])
        root_R_inv = R.from_euler("y", root_y_rotation).inv()

        # Root trajectories and forward directions
        for offset in OFFSETS:
            next_frame = min(frame + offset, n_frames - 1)

            position = positions[next_frame, ROOTIDX]
            position = root_R_inv.apply(position - root_T)
            feature.extend(position[[0, 2]])

            y_rotation = extract_y_rotation(rotations[next_frame, ROOTIDX])
            y_rotation = root_R_inv * R.from_euler("y", y_rotation)
            forward = y_rotation.apply(np.array([-1.0, 0.0, 0.0]))
            feature.extend(forward[[0, 2]])

        # Root velocity
        root_velocity = velocities[frame, ROOTIDX]
        root_velocity = root_R_inv.apply(root_velocity)
        feature.extend(root_velocity)

        # Foot positions and velocities
        for joint_name in ["LeftFoot", "RightFoot"]:
            joint_idx = joints.index(joint_name)
            position = positions[frame, joint_idx]
            position = root_R_inv.apply(position - root_T)
            velocity = velocities[frame, joint_idx]
            velocity = root_R_inv.apply(velocity)
            feature.extend(position)
            feature.extend(velocity)

        features.append(feature)

    return np.array(features)
