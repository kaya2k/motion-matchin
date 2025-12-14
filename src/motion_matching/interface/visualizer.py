import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_matching.utils import extract_y_rotation


class MotionVisualizer:
    """Class to visualize motion using Rerun."""

    def __init__(self):
        XYZ_TO_ZXY = [0, 0, 0.01, 0.01, 0, 0, 0, 0.01, 0]
        rr.init("motion_matching", spawn=True)
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
        is_toe_contact,
        input_direction,
    ):
        self.log_local(positions, rotations)
        self.log_bones(joints, edges, positions, rotations, is_toe_contact)
        self.log_input_direction(positions, input_direction)
        self.log_future(future_positions, future_directions)

    def log_local(self, positions, rotations):
        root_position = positions[0].copy()
        root_position[1] = 0
        root_y_rotation = extract_y_rotation(rotations[0])
        rr.log(
            "world/local",
            rr.Transform3D(
                translation=root_position,
                quaternion=R.from_euler("y", root_y_rotation).as_quat(),
            ),
        )
        rr.log(
            "world/local/position",
            rr.Points3D(
                positions=np.array([0.0, 0.0, 0.0]), radii=4.0, colors=[[128, 128, 128]]
            ),
        )

    def log_bones(self, joints, edges, positions, rotations, is_toe_contact):
        for i, j in edges:
            center = (positions[i] + positions[j]) / 2.0
            length = np.linalg.norm(positions[i] - positions[j])
            half_sizes = np.array([length / 2.0, 4.0, 4.0])
            quat = R.from_euler("xyz", rotations[i]).as_quat()
            if joints[j] == "LeftToe" and is_toe_contact[0]:
                color = [255, 0, 0]
            elif joints[j] == "RightToe" and is_toe_contact[1]:
                color = [255, 0, 0]
            else:
                color = [255, 255, 255]
            rr.log(
                f"world/bones/{joints[i]}-{joints[j]}",
                rr.Boxes3D(
                    centers=[center],
                    half_sizes=[half_sizes],
                    rotations=[quat],
                    colors=[color],
                    fill_mode="solid",
                ),
            )

    def log_input_direction(self, positions, input_direction):
        origin = positions[0].copy()
        origin[1] = 0.0
        rr.log(
            "world/input_direction",
            rr.Arrows3D(
                origins=[origin],
                vectors=[input_direction * 100.0],
                colors=[[128, 128, 128]],
            ),
        )

    def log_future(self, future_positions, future_directions):
        origins = future_positions.copy()
        vectors = future_directions * 20.0
        rr.log(
            "world/local/future_positions",
            rr.Points3D(positions=origins, radii=4.0, colors=[[128, 0, 0]]),
        )
        rr.log(
            "world/local/future_directions",
            rr.Arrows3D(origins=origins, vectors=vectors, colors=[[128, 0, 0]]),
        )
