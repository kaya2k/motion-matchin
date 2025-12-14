import numpy as np
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from motion_matching.bvh import BVH, BVHNode
from motion_matching.utils import extract_y_rotation


class PoseSet:
    """Class representing a 3D poses with rotations"""

    def __init__(self, bvh: BVH):
        self.n_frames = bvh.n_frames
        self.xz_translations, self.y_positions = self.extract_root_poses(bvh)
        self.dy_rotations = self.extract_dy_rotations(bvh)
        self.rotations = self.extract_rotations(bvh)
        self.normalize()

    def extract_root_poses(self, bvh: BVH):
        root_positions = np.array(bvh.root.channel_values)[:, :3]
        translations = np.diff(root_positions, axis=0)
        translations = np.insert(translations, 0, np.array([0.0, 0.0, 0.0]), axis=0)
        translations[:, 1] = 0.0  # Ignore y translation
        return translations, root_positions[:, 1]

    def extract_dy_rotations(self, bvh: BVH):
        root_rotation_degrees_zyx = np.array(bvh.root.channel_values)[:, -3:]
        root_rotation_radians_xyz = np.deg2rad(root_rotation_degrees_zyx[:, ::-1])
        root_y_rotations = extract_y_rotation(root_rotation_radians_xyz)
        dy_rotations = np.diff(root_y_rotations, axis=0)
        dy_rotations = np.insert(dy_rotations, 0, 0.0)
        return dy_rotations

    def extract_rotations(self, bvh: BVH):
        rotations = []
        self.collect_rotations(bvh.root, rotations)
        rotations = np.array(rotations)
        rotations = rearrange(rotations, "joints frames xyz -> frames joints xyz")
        return rotations

    def collect_rotations(self, node: BVHNode, rotations):
        rotation_degrees_zyx = np.array(node.channel_values)[:, -3:]
        rotation_radians_xyz = np.deg2rad(rotation_degrees_zyx[:, ::-1])
        rotations.append(rotation_radians_xyz)
        for child_node in node.children:
            self.collect_rotations(child_node, rotations)

    def normalize(self):
        root_rotations = self.rotations[:, 0]
        root_R = R.from_euler("xyz", root_rotations)
        root_y_rotations = extract_y_rotation(root_rotations)
        root_y_R = R.from_euler("y", root_y_rotations)
        self.xz_translations = root_y_R.inv().apply(self.xz_translations)
        self.rotations[:, 0] = (root_y_R.inv() * root_R).as_euler("xyz")
