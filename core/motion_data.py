import numpy as np
from .bvh import BVH


class MotionData:
    """Class to handle motion capture data."""

    n_frames: int
    frame_time: float
    n_joints: int
    joints: list[str]
    edge_list: list[tuple[int, int]]
    positions: np.ndarray
    rotations: np.ndarray

    def __init__(self):
        pass

    def load_from_bvh(self, bvh_filename):
        bvh = BVH(bvh_filename)
        self.n_frames = bvh.n_frames
        self.frame_time = bvh.frame_time
        n_joints, joints, edge_list, positions, rotations = bvh.get_motion_data()
        self.joints = joints
        self.n_joints = n_joints
        self.edge_list = edge_list
        self.positions = positions
        self.rotations = rotations
