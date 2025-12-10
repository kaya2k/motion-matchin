import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from .motion_data import MotionData
from .feature import FEATURE_DIM, OFFSETS
from .utils import (
    project_to_xz,
    extract_y_rotation,
    wrap_angle,
    spring_model,
)


class MotionMatchingController:
    """Class to control motion matching."""

    def __init__(self):
        self.motion_dataset = []
        self.feature_nns = []
        self.feature_mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.feature_std = np.ones(FEATURE_DIM, dtype=np.float32)
        self.preprocess()

        self.SEARCH_INTERVAL = 5
        self.LOCK_FOOT = False

        self.frame_after_search = 0
        self.input_direction = np.array([0.0, 0.0, 0.0])
        self.future_positions = np.zeros((len(OFFSETS), 3), dtype=np.float32)
        self.future_directions = np.zeros((len(OFFSETS), 3), dtype=np.float32)

        self.origin_data_idx = 0
        self.target_data_idx = 0
        self.origin_frame = 0
        self.target_frame = 0

        y = self.motion_dataset[0].positions[0, 0][1]
        self.root_position = np.array([0.0, y, 0.0])
        self.root_y_rotation = np.pi / 2.0
        self.root_xz_velocity = np.array([0.0, 0.0, 0.0])

    def preprocess(self):
        BVH_DIR = "./data/bvh"
        for filename in sorted(os.listdir(BVH_DIR)):
            if filename.endswith(".bvh"):
                print(f"[PREPROCESS] Loading {filename}")
                motion_data = MotionData(os.path.join(BVH_DIR, filename))
                self.motion_dataset.append(motion_data)

        # Compute feature mean and std
        all_features = []
        for motion_data in self.motion_dataset:
            all_features.append(motion_data.features)
        all_features = np.concatenate(all_features, axis=0)
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0)

        for motion_data in self.motion_dataset:
            features = motion_data.features
            features = self.normalize(features)
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(features)
            self.feature_nns.append(nn)

    def update(self):
        self.origin_frame += 1
        self.target_frame += 1
        self.frame_after_search += 1

        if self.frame_after_search == self.SEARCH_INTERVAL:
            self.frame_after_search = 0
            self.search()

        motion_data = self.motion_dataset[self.origin_data_idx]

        root_R = R.from_euler("y", self.root_y_rotation)
        translation = motion_data.translations[self.origin_frame]
        translation = root_R.apply(translation)

        self.root_position += translation
        self.root_y_rotation += motion_data.dy_rotations[self.origin_frame]
        self.root_y_rotation = wrap_angle(self.root_y_rotation)
        self.root_xz_velocity = project_to_xz(translation)

        self.update_future()

    def update_future(self):
        root_R = R.from_euler("y", self.root_y_rotation)
        local_root_velocity = root_R.inv().apply(self.root_xz_velocity)
        local_input_direction = root_R.inv().apply(self.input_direction)

        DISTANCE = 8 * 60
        target_position = local_input_direction * DISTANCE

        # Calculate expected future root positions and directions
        for i, offset in enumerate(OFFSETS):
            position0 = spring_model(local_root_velocity, target_position, offset)
            position1 = spring_model(local_root_velocity, target_position, offset + 1)
            direction = position1 - position0
            if np.linalg.norm(direction) > 1e-6:
                direction /= np.linalg.norm(direction)
            self.future_positions[i] = position0
            self.future_directions[i] = direction

    def search(self):
        time_start = time.perf_counter()

        self.origin_data_idx = self.target_data_idx
        self.origin_frame = self.target_frame

        feature = self.get_current_feature()
        feature = self.normalize(feature)

        min_distance = float("inf")
        best_data_idx = -1
        best_frame = -1

        for data_idx in range(len(self.motion_dataset)):
            nn = self.feature_nns[data_idx]
            distances, indices = nn.kneighbors([feature], n_neighbors=1)
            distance = distances[0][0]
            frame = indices[0][0]

            if distance < min_distance:
                min_distance = distance
                best_data_idx = data_idx
                best_frame = frame

        self.target_data_idx = best_data_idx
        self.target_frame = best_frame

        time_end = time.perf_counter()
        print(
            (
                f"\r[SEARCH] data {best_data_idx:1d} "
                f"frame {best_frame:4d} "
                f"distance {min_distance:.4f} "
                f"time {time_end - time_start:.4f}s"
            ),
            end="",
        )

    def get_current_feature(self):
        feature = np.zeros(FEATURE_DIM, dtype=np.float32)
        for i in range(len(OFFSETS)):
            feature[i * 4 : i * 4 + 2] = self.future_positions[i][[0, 2]]
            feature[i * 4 + 2 : i * 4 + 4] = self.future_directions[i][[0, 2]]

        motion_data = self.motion_dataset[self.origin_data_idx]
        feature[12:] = motion_data.features[self.origin_frame, 12:]
        return feature

    def normalize(self, feature):
        return (feature - self.feature_mean) / self.feature_std

    def get_current_pose(self):
        motion_data = self.motion_dataset[self.origin_data_idx]
        joints = motion_data.joints
        edges = motion_data.edges
        positions = motion_data.positions[self.origin_frame]
        rotations = motion_data.rotations[self.origin_frame]

        root_position = positions[0]
        root_y_rotation = extract_y_rotation(rotations[0])
        root_R = R.from_euler("y", self.root_y_rotation - root_y_rotation)

        positions = self.root_position + root_R.apply(positions - root_position)
        rotations = (root_R * R.from_euler("xyz", rotations)).as_euler("xyz")

        return (
            joints,
            edges,
            positions,
            rotations,
            self.future_positions,
            self.future_directions,
        )
