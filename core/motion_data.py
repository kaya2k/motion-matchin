import os
import numpy as np
from .bvh import BVH
from .feature import extract_features


class MotionData:
    """Class to handle motion data for motion matching."""

    n_frames: int
    frame_time: float
    joints: list[str]
    edges: list[tuple[int, int]]
    positions: np.ndarray
    rotations: np.ndarray
    features: np.ndarray

    def __init__(self, bvh_filename):
        name = os.path.basename(bvh_filename).replace(".bvh", "")
        print(f"Loading MotionData from {name}")

        POSITIONS_DIR = "./data/positions"
        ROTATIONS_DIR = "./data/rotations"
        FEATURES_DIR = "./data/features"
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
