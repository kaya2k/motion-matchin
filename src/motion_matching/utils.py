import numpy as np
from scipy.spatial.transform import Rotation as R


def extract_y_rotation(rotations):
    """Extract y-axis rotation from full 3D rotations."""

    rotations = rotations.reshape(-1, 3)
    r = R.from_euler("xyz", rotations)
    z_axis = r.apply(np.array([0.0, 0.0, 1.0]))
    y_rotations = np.arctan2(z_axis[:, 0], z_axis[:, 2])
    return y_rotations.squeeze()


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""

    return (angle + np.pi) % (2 * np.pi) - np.pi


def spring_model(velocity, target, dt, smoothness=60.0):
    """Spring model to smoothly interpolate position."""

    # <Game Programming Gems 4> Chapter 1.10
    omega = 2.0 / smoothness
    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = -target
    temp = (velocity + omega * change) * dt
    new_pos = target + (change + temp) * exp
    return new_pos
