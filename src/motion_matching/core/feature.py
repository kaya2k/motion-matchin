import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from motion_matching.core.pose import PoseSet
from motion_matching.core.skeleton import Skeleton


class FeatureSet:
    """Class to hold feature data for motion matching."""

    OFFSETS = [10, 20, 30]
    FORWARD = np.array([-1.0, 0.0, 0.0])
    FEATURE_SAVE_DIR = "./data/feature"

    def __init__(self, pose_set: PoseSet, skeleton: Skeleton, name):
        self.features = self.extract_features(pose_set, skeleton, name)
        self.nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        self.mean = np.zeros_like(self.features[0])
        self.std = np.ones_like(self.features[0])

    def normalize_and_fit(self, mean, std):
        self.features = (self.features - mean) / std
        self.nn.fit(self.features[: -self.OFFSETS[-1]])
        self.mean = mean
        self.std = std

    def search(self, query_feature):
        distances, indices = self.nn.kneighbors([query_feature])
        return distances[0][0], indices[0][0]

    def extract_features(self, pose_set, skeleton, name):
        save_path = os.path.join(self.FEATURE_SAVE_DIR, f"{name}.npy")
        os.makedirs(self.FEATURE_SAVE_DIR, exist_ok=True)
        if os.path.exists(save_path):
            return np.load(save_path)

        features = []
        for frame in range(pose_set.n_frames):
            feature = []
            self.append_future_features(pose_set, frame, feature)
            self.append_foot_features(pose_set, skeleton, frame, feature)
            features.append(feature)

        features = np.array(features, dtype=np.float32)
        np.save(save_path, features)
        return np.array(features)

    def extract_current_feature(self, trajectories, directions, frame):
        feature = []
        for i in range(len(self.OFFSETS)):
            feature.extend(trajectories[i][[0, 2]])
            feature.extend(directions[i][[0, 2]])
        feature.extend(self.features[frame, -12:])
        feature = np.array(feature)
        feature[:-12] = (feature[:-12] - self.mean[:-12]) / self.std[:-12]
        return feature

    def append_future_features(self, pose_set, frame, feature):
        trajectory = np.array([0.0, 0.0, 0.0])
        y_rotation = 0.0
        for offset in range(1, self.OFFSETS[-1] + 1):
            next_frame = min(frame + offset, pose_set.n_frames - 1)
            xz_translation = pose_set.xz_translations[next_frame]
            translation = np.array([xz_translation[0], 0.0, xz_translation[1]])
            dy_rotation = pose_set.dy_rotations[next_frame]
            trajectory += R.from_euler("y", y_rotation).apply(translation)
            y_rotation += dy_rotation
            if offset in self.OFFSETS:
                direction = R.from_euler("y", y_rotation).apply(self.FORWARD)
                feature.extend(trajectory[[0, 2]])
                feature.extend(direction[[0, 2]])

    def append_foot_features(self, pose_set, skeleton, frame, feature):
        root_position = np.array([0.0, pose_set.y_positions[frame], 0.0])
        y_rotation = 0.0
        frame1 = min(frame + 1, pose_set.n_frames - 1)
        positions0, _ = skeleton.apply_pose(root_position, y_rotation, pose_set, frame)
        positions1, _ = skeleton.apply_pose(root_position, y_rotation, pose_set, frame1)
        for joint_idx in [skeleton.LFOOT_INDEX, skeleton.RFOOT_INDEX]:
            position = positions0[joint_idx]
            velocity = positions1[joint_idx] - positions0[joint_idx]
            feature.extend(position)
            feature.extend(velocity)
