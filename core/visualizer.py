import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R

XYZ_TO_ZXY = [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
]


class MotionVisualizer:
    """Class to visualize motion using Rerun."""

    def __init__(self):
        rr.init("MotionVisualizer", spawn=True)
        rr.log("world", rr.Transform3D(mat3x3=XYZ_TO_ZXY))
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[100, 0, 0], [0, 100, 0], [0, 0, 100]],
                colors=[[255, 0, 0], [255, 0, 0], [255, 0, 0]],
            ),
        )

    def update(self, joints, edge_list, positions, rotations):
        # Log root position
        root_position = positions[0].copy()
        root_position[1] = 0
        rr.log(
            "world/root",
            rr.Points3D(positions=[root_position], radii=4.0, colors=[[255, 0, 0]]),
        )

        # Log bones
        for i, j in edge_list:
            center = (positions[i] + positions[j]) / 2.0
            length = np.linalg.norm(positions[i] - positions[j])
            size = 8.0
            half_sizes = np.array([length / 2.0, size / 2.0, size / 2.0])
            quat = R.from_euler("xyz", rotations[i]).as_quat()
            rr.log(
                f"world/bones/{joints[i]}-{joints[j]}",
                rr.Boxes3D(
                    centers=[center],
                    half_sizes=[half_sizes],
                    rotations=[quat],
                    colors=[[255, 255, 255]],
                    fill_mode="solid",
                ),
            )
