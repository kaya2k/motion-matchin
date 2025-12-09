import rerun as rr
import numpy as np

YUP_TO_ZUP = np.array(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]
)


class MotionVisualizer:
    """Class to visualize motion using Rerun."""

    def __init__(self):
        rr.init("MotionVisualizer", spawn=True)
        rr.log("world", rr.Transform3D(mat3x3=YUP_TO_ZUP))

    def update(self, n_joints, edge_list, positions, rotations):
        rr.log("world/joints", rr.Points3D(positions, radii=4))
