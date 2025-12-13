import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from motion_matching.core.motion_data import MotionData
from motion_matching.core.feature import FEATURE_DIM, OFFSETS
from motion_matching.utils import (
    project_to_xz,
    extract_y_rotation,
    wrap_angle,
    spring_model,
)


class MotionMatchingController:
    """Class to control motion matching."""

    def __init__(self):
        self.SEARCH_INTERVAL = 5
        self.LOCK_FOOT = False

        self.motion_dataset = []
        self.feature_nns = []
        self.feature_mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.feature_std = np.ones(FEATURE_DIM, dtype=np.float32)
        self.preprocess()

        self.input_direction = np.array([0.0, 0.0, 0.0])
        self.future_positions = np.zeros((len(OFFSETS), 3), dtype=np.float32)
        self.future_directions = np.zeros((len(OFFSETS), 3), dtype=np.float32)

        self.target_data = 0
        self.target_frame = 0
        self.frame_after_search = 1

        y = self.motion_dataset[0].positions[0, 0][1]
        self.root_position = np.array([0.0, y, 0.0])
        self.root_y_rotation = np.pi / 2.0

        self.root_velocity = np.array([0.0, 0.0, 0.0])
        self.root_angular_velocity = 0.0

        self.offset_position = np.array([0.0, 0.0, 0.0])
        self.offset_velocity = np.array([0.0, 0.0, 0.0])
        self.offset_y_rotation = 0.0
        self.offset_angular_velocity = 0.0

        self.origin_positions, self.origin_rotations = self.get_positions_rotations()
        self.target_positions, self.target_rotations = self.get_positions_rotations()

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
            features = motion_data.features[: -2 * self.SEARCH_INTERVAL]
            features = self.normalize(features)
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(features)
            self.feature_nns.append(nn)

    def update(self):
        if self.frame_after_search == self.SEARCH_INTERVAL:
            self.frame_after_search = 1
            self.origin_positions, self.origin_rotations = (
                self.get_positions_rotations()
            )
            self.search()
            self.target_positions, self.target_rotations = (
                self.get_positions_rotations()
            )

        motion_data = self.motion_dataset[self.target_data]
        root_R = R.from_euler("y", self.root_y_rotation)
        translation = motion_data.translations[self.target_frame]
        translation[1] = 0.0  # Ignore y translation
        translation = root_R.apply(translation)
        dy_rotation = motion_data.dy_rotations[self.target_frame]

        # Update root position and y rotation
        self.root_position += translation
        self.root_velocity = translation
        self.root_y_rotation = wrap_angle(self.root_y_rotation + dy_rotation)
        self.root_angular_velocity = dy_rotation

        self.update_future()
        self.target_frame += 1
        self.frame_after_search += 1

    def update_future(self):
        MAX_SPEED = 20.0
        ESTIMATED_TIME = 60

        root_R = R.from_euler("y", self.root_y_rotation)
        root_xz_speed = np.linalg.norm(project_to_xz(self.root_velocity))
        local_root_velocity = np.array([-1.0, 0.0, 0.0]) * root_xz_speed
        local_input_direction = root_R.inv().apply(self.input_direction)

        target_position = local_input_direction * MAX_SPEED * ESTIMATED_TIME

        # Calculate expected future root positions and directions
        for i, offset in enumerate(OFFSETS):
            position0 = spring_model(local_root_velocity, target_position, offset)
            position1 = spring_model(local_root_velocity, target_position, offset + 1)
            direction = position1 - position0
            if np.linalg.norm(direction) > 1e-6:
                direction /= np.linalg.norm(direction)
            else:
                direction = np.array([-1.0, 0.0, 0.0])
            self.future_positions[i] = position0
            self.future_directions[i] = direction

    def search(self):
        time_start = time.perf_counter()

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

        self.target_data = best_data_idx
        self.target_frame = best_frame

        time_end = time.perf_counter()
        print(f"[SEARCH] data {best_data_idx:1d}", end=" ")
        print(f"frame {best_frame:4d} ", end=" ")
        print(f"distance {min_distance:.4f} ", end=" ")
        print(f"time {time_end - time_start:.4f}s", end=" ")
        print()

    def get_current_feature(self):
        feature = []
        for i in range(len(OFFSETS)):
            feature.extend(self.future_positions[i][[0, 2]])
            feature.extend(self.future_directions[i][[0, 2]])

        root_R = R.from_euler("y", self.root_y_rotation)
        feature.extend(root_R.inv().apply(self.root_velocity))

        motion_data = self.motion_dataset[self.target_data]
        feature.extend(motion_data.features[self.target_frame, -12:])
        return np.array(feature)

    def normalize(self, feature):
        return (feature - self.feature_mean) / self.feature_std

    def get_positions_rotations(self):
        motion_data = self.motion_dataset[self.target_data]
        positions = motion_data.positions[self.target_frame]
        rotations = motion_data.rotations[self.target_frame]

        root_position = positions[0]
        root_y_rotation = extract_y_rotation(rotations[0])
        root_R = R.from_euler("y", self.root_y_rotation - root_y_rotation)

        positions = self.root_position + root_R.apply(positions - root_position)
        rotations = (root_R * R.from_euler("xyz", rotations)).as_euler("xyz")
        return positions, rotations

    def get_current_pose(self):
        positions, rotations = self.get_positions_rotations()
        return (
            self.motion_dataset[self.target_data].joints,
            self.motion_dataset[self.target_data].edges,
            positions,
            rotations,
            self.future_positions,
            self.future_directions,
        )
