import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R
from .utils import extract_y_rotation


class MotionVisualizer:
    """Class to visualize motion using Rerun."""

    def __init__(self):
        self.input_direction = np.array([0.0, 0.0, 0.0])

        XYZ_TO_ZXY = [0, 0, 1, 1, 0, 0, 0, 1, 0]
        rr.init("MotionVisualizer", spawn=True)
        rr.log("world", rr.Transform3D(mat3x3=XYZ_TO_ZXY))
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[100, 0, 0], [0, 100, 0], [0, 0, 100]],
                colors=[[255, 255, 255]],
            ),
        )

    def update(
        self,
        joints,
        edges,
        positions,
        rotations,
        future_positions,
        future_directions,
    ):
        self.log_root(positions, rotations)
        self.log_bones(joints, edges, positions, rotations)
        self.log_input_direction(positions)
        self.log_future(future_positions, future_directions)

    def log_root(self, positions, rotations):
        root_position = positions[0].copy()
        root_position[1] = 0
        root_y_rotation = extract_y_rotation(rotations[0])
        rr.log(
            "world/root",
            rr.Transform3D(
                translation=root_position,
                quaternion=R.from_euler("y", root_y_rotation).as_quat(),
            ),
        )

    def log_bones(self, joints, edges, positions, rotations):
        for i, j in edges:
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

    def log_input_direction(self, positions):
        root_position = positions[0].copy()
        root_position[1] = 0
        rr.log(
            "world/input_direction",
            rr.Arrows3D(
                origins=[root_position],
                vectors=[self.input_direction * 100.0],
                colors=[[255, 0, 0]],
            ),
        )

    def log_future(self, future_positions, future_directions):
        origins = future_positions.copy()
        vectors = future_directions * 20.0
        rr.log(
            "world/root/future_positions",
            rr.Points3D(positions=origins, radii=4.0, colors=[[255, 255, 255]]),
        )
        rr.log(
            "world/root/future_directions",
            rr.Arrows3D(origins=origins, vectors=vectors, colors=[[255, 255, 255]]),
        )
