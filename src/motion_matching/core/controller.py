import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_matching.bvh import BVH
from motion_matching.core.blender import InertialRotationBlender
from motion_matching.core.pose import PoseSet
from motion_matching.core.feature import FeatureSet
from motion_matching.core.skeleton import Skeleton
from motion_matching.utils import wrap_angle, spring_model


class MotionMatchingController:
    """Class to perform motion matching"""

    def __init__(self):
        self.SEARCH_INTERVAL = 5
        self.MAX_SPEED = 20.0
        self.FRAMES_TO_TARGET = 60
        self.ROOT_Y_BLENDING = 0.2

        # Datasets
        self.skeleton = Skeleton()
        self.pose_sets = []
        self.feature_sets = []
        self.load_datasets()
        self.normalize_and_fit_features()

        # Current state
        self.root_position = np.array([0.0, 0.0, 0.0])
        self.root_position[1] = self.pose_sets[0].y_positions[0]
        self.root_y_rotation = np.pi / 2.0
        self.root_xz_velocity = np.array([0.0, 0.0, 0.0])
        self.data_index = 0
        self.frame = 0
        self.frame_after_search = 0
        self.joint_rotations = np.zeros((self.skeleton.n_joints, 3))

        # Foot contact
        self.is_toe_contact = [False, False]
        self.toe_xz_position = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]

        # Blenders
        self.pose_blender = InertialRotationBlender(blending_time=self.SEARCH_INTERVAL)
        self.pose_blender.reset(
            self.pose_sets[0].rotations[0],
            self.pose_sets[0].drotations[0],
            self.pose_sets[0].rotations[0],
            self.pose_sets[0].drotations[0],
        )

        # Future trajectory
        self.future_trajectories = np.zeros((len(FeatureSet.OFFSETS), 3))
        self.future_directions = np.zeros((len(FeatureSet.OFFSETS), 3))

    def load_datasets(self):
        BVH_DIR = "./data/bvh"
        for filename in sorted(os.listdir(BVH_DIR)):
            if not filename.endswith(".bvh"):
                continue
            print(f"[LOAD_DATASETS] BVH {len(self.pose_sets):02d} ({filename})")
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
            prev_pose_set = self.pose_sets[self.data_index]
            prev_drotations = prev_pose_set.drotations[self.frame]

            self.search()
            self.frame_after_search = 0

            pose_set = self.pose_sets[self.data_index]
            translation = pose_set.xz_translations[self.frame]
            rotations = pose_set.rotations[self.frame]
            drotations = pose_set.drotations[self.frame]

            self.pose_blender.reset(
                self.joint_rotations, prev_drotations, rotations, drotations
            )

        self.pose_blender.update(dt=1)

        pose_set = self.pose_sets[self.data_index]
        root_y_R = R.from_euler("y", self.root_y_rotation)
        translation = pose_set.xz_translations[self.frame]
        self.root_position += root_y_R.apply(translation)
        self.root_position[1] *= 1.0 - self.ROOT_Y_BLENDING
        self.root_position[1] += self.ROOT_Y_BLENDING * pose_set.y_positions[self.frame]
        self.root_y_rotation += pose_set.dy_rotations[self.frame]
        self.root_y_rotation = wrap_angle(self.root_y_rotation)
        self.root_xz_velocity = root_y_R.apply(translation)

        self.update_future(input_direction)

    def update_future(self, input_direction):
        root_y_R = R.from_euler("y", self.root_y_rotation)
        root_xz_speed = np.linalg.norm(self.root_xz_velocity)
        local_root_velocity = np.array(FeatureSet.FORWARD) * root_xz_speed
        local_input_direction = root_y_R.inv().apply(input_direction)
        target_position = local_input_direction * self.MAX_SPEED * self.FRAMES_TO_TARGET

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
        rotations = self.pose_sets[self.data_index].rotations[self.frame]
        original_R = R.from_euler("xyz", rotations)
        offset = self.pose_blender.offset
        offset_R = R.from_rotvec(offset)
        blended_R = offset_R * original_R
        blended_rotations = blended_R.as_euler("xyz")
        self.joint_rotations = blended_rotations

        return self.skeleton.apply_pose(
            self.root_position,
            self.root_y_rotation,
            blended_rotations,
        )
