import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_matching.utils import extract_y_rotation


class MotionVisualizer:
    """Class to visualize motion using Rerun."""

    def __init__(self):
        XYZ_TO_ZXY = [0, 0, 1, 1, 0, 0, 0, 1, 0]
        self.frame = 0

        rr.init("motion_matching", spawn=True)
        rr.set_time("frame", sequence=self.frame)
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
        edges,
        positions,
        rotations,
        future_positions,
        future_directions,
        input_direction,
    ):
        self.frame += 1
        rr.set_time("frame", sequence=self.frame)
        self.log_bones(edges, positions, rotations)
        self.log_local(positions, rotations)
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

    def log_bones(self, edges, positions, rotations):
        centers = [[positions[0][0], 100.0, positions[0][2]]]
        half_sizes = [np.array([100.0, 150.0, 100.0])]
        quats = [R.from_euler("y", 0.0).as_quat()]
        colors = [[255, 255, 255, 0]]
        for i, j in edges:
            center = (positions[i] + positions[j]) / 2.0
            length = np.linalg.norm(positions[i] - positions[j])
            half_size = np.array([length / 2.0, 4.0, 4.0])
            quat = R.from_euler("xyz", rotations[i]).as_quat()
            centers.append(center)
            half_sizes.append(half_size)
            quats.append(quat)
            colors.append([255, 255, 255, 255])

        rr.log(
            f"world/bones",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                rotations=quats,
                colors=colors,
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
