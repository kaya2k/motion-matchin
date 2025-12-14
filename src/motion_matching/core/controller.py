import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_matching.bvh import BVH
from motion_matching.core.pose import PoseSet
from motion_matching.core.feature import FeatureSet
from motion_matching.core.skeleton import Skeleton
from motion_matching.utils import wrap_angle, spring_model


class MotionMatchingController:
    """Class to perform motion matching"""

    def __init__(self):
        self.SEARCH_INTERVAL = 5

        # Datasets
        self.skeleton = Skeleton()
        self.pose_sets = []
        self.feature_sets = []
        self.load_datasets()
        self.normalize_and_fit_features()

        # Current state
        self.root_position = np.array([0.0, 0.0, 0.0])
        self.root_y_rotation = np.pi / 2.0
        self.root_xz_velocity = np.array([0.0, 0.0])
        self.data_index = 0
        self.frame = 0
        self.frame_after_search = 0

        # Future trajectory
        self.future_trajectories = np.zeros((len(FeatureSet.OFFSETS), 3))
        self.future_directions = np.zeros((len(FeatureSet.OFFSETS), 3))

    def load_datasets(self):
        BVH_DIR = "./data/bvh"
        for filename in sorted(os.listdir(BVH_DIR)):
            if not filename.endswith(".bvh"):
                continue
            print(f"[PREPROCESS] Loading bvh {len(self.pose_sets):02d} ({filename})")
            bvh = BVH(os.path.join(BVH_DIR, filename))
            if self.skeleton.n_joints == 0:
                self.skeleton.build_skeleton(bvh.root)
            pose_set = PoseSet(bvh)
            feature_set = FeatureSet(pose_set, self.skeleton, filename[:-4])
            self.pose_sets.append(pose_set)
            self.feature_sets.append(feature_set)

    def normalize_and_fit_features(self):
        all_features = []
        for feature_set in self.feature_sets:
            all_features.append(feature_set.features)
        all_features = np.concatenate(all_features, axis=0)
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0)
        for feature_set in self.feature_sets:
            feature_set.normalize_and_fit(self.feature_mean, self.feature_std)

    def update(self, input_direction):
        self.frame += 1
        self.frame_after_search += 1

        if self.frame_after_search == self.SEARCH_INTERVAL:
            self.update_future(input_direction)
            self.search()
            self.frame_after_search = 0

        pose_set = self.pose_sets[self.data_index]
        root_y_R = R.from_euler("y", self.root_y_rotation)
        self.root_position += root_y_R.apply(pose_set.xz_translations[self.frame])
        self.root_position[1] = pose_set.y_positions[self.frame]
        self.root_y_rotation += pose_set.dy_rotations[self.frame]
        self.root_y_rotation = wrap_angle(self.root_y_rotation)
        self.root_xz_velocity = pose_set.xz_translations[self.frame]

    def update_future(self, input_direction):
        root_y_R = R.from_euler("y", self.root_y_rotation)
        root_xz_speed = np.linalg.norm(self.root_xz_velocity)
        local_root_velocity = np.array(FeatureSet.FORWARD) * root_xz_speed
        local_input_direction = root_y_R.inv().apply(input_direction)
        target_position = local_input_direction * 20.0 * 60

        for index, offset in enumerate(FeatureSet.OFFSETS):
            position0 = spring_model(local_root_velocity, target_position, offset)
            position1 = spring_model(local_root_velocity, target_position, offset + 1)
            direction = position1 - position0
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = FeatureSet.FORWARD
            self.future_trajectories[index] = position0
            self.future_directions[index] = direction

    def search(self):
        time_start = time.perf_counter()
        feature = self.feature_sets[self.data_index].extract_current_feature(
            self.future_trajectories,
            self.future_directions,
            self.frame,
        )
        min_distance = np.inf
        for i, feature_set in enumerate(self.feature_sets):
            distance, frame = feature_set.search(feature)
            if distance < min_distance:
                min_distance = distance
                self.data_index = i
                self.frame = frame

        time_end = time.perf_counter()
        print(f"[SEARCH] Data {self.data_index:1d}", end=" ")
        print(f"Frame {self.frame:4d}", end=" ")
        print(f"Distance {min_distance:.4f}", end=" ")
        print(f"Time {time_end - time_start:.4f}s", end=" ")
        print()

    def get_global_positions_rotations(self):
        positions, rotations = self.skeleton.apply_pose(
            self.root_position,
            self.root_y_rotation,
            self.pose_sets[self.data_index],
            self.frame,
        )
        return positions, rotations
