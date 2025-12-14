import numpy as np
from scipy.spatial.transform import Rotation as R


class InertialRotationBlender:
    """Class to blend rotations using inertia."""

    def __init__(self, blending_time):
        self.blending_time = blending_time
        self.offset = None
        self.velocity = None

    def reset(self, org_rotations, org_drotations, new_rotations, new_drotations):
        org_R = R.from_euler("xyz", org_rotations)
        new_R = R.from_euler("xyz", new_rotations)
        offset_R = org_R * new_R.inv()
        self.offset = offset_R.as_rotvec()
        self.velocity = org_drotations - new_drotations

    def update(self, dt):
        k = 2.0 / self.blending_time
        e = np.exp(-k * dt)
        offset, velocity = self.offset, self.velocity
        self.offset = e * (offset + (velocity + k * offset) * dt)
        self.velocity = e * (velocity - k * (velocity + k * offset) * dt)
