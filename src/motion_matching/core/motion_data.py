import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_matching.bvh import BVH
from motion_matching.core.feature import extract_features
from motion_matching.utils import extract_y_rotation


class MotionData:
    """Class to handle motion data for motion matching."""

    n_frames: int
    frame_time: float
    joints: list[str]
    edges: list[tuple[int, int]]
    positions: np.ndarray
    rotations: np.ndarray
    features: np.ndarray
    translations: np.ndarray
    dy_rotations: np.ndarray

    def __init__(self, bvh_filename):
        self.load_data(bvh_filename)
        self.calculate_delta()

    def load_data(self, bvh_filename):
        POSITIONS_DIR = "./data/positions"
        ROTATIONS_DIR = "./data/rotations"
        FEATURES_DIR = "./data/features"
        name = os.path.basename(bvh_filename).replace(".bvh", "")
        positions_filename = os.path.join(POSITIONS_DIR, f"{name}_positions.npy")
        rotations_filename = os.path.join(ROTATIONS_DIR, f"{name}_rotations.npy")
        features_filename = os.path.join(FEATURES_DIR, f"{name}_features.npy")

        for dir in [POSITIONS_DIR, ROTATIONS_DIR, FEATURES_DIR]:
            os.makedirs(dir, exist_ok=True)

        # Load BVH file
        bvh = BVH(bvh_filename)
        self.n_frames = bvh.n_frames
        self.frame_time = bvh.frame_time
        self.joints = bvh.joints
        self.edges = bvh.edges

        # Load or compute positions and rotations
        if os.path.exists(positions_filename) and os.path.exists(rotations_filename):
            self.positions = np.load(positions_filename)
            self.rotations = np.load(rotations_filename)
        else:
            self.positions, self.rotations = bvh.calculate_positions_rotations()
            np.save(positions_filename, self.positions)
            np.save(rotations_filename, self.rotations)

        # Load or compute features
        if os.path.exists(features_filename):
            self.features = np.load(features_filename)
        else:
            self.features = extract_features(
                self.joints, self.positions, self.rotations
            )
            np.save(features_filename, self.features)

    def calculate_delta(self):
        y_rotations = extract_y_rotation(self.rotations[:, 0])
        root_R = R.from_euler("y", y_rotations)
        self.translations = np.zeros((self.n_frames, 3), dtype=np.float32)
        self.translations[1:] = self.positions[1:, 0] - self.positions[:-1, 0]
        self.translations[1:] = root_R[:-1].inv().apply(self.translations[1:])
        self.dy_rotations = np.zeros(self.n_frames, dtype=np.float32)
        self.dy_rotations[1:] = y_rotations[1:] - y_rotations[:-1]
