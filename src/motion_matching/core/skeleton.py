import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_matching.bvh import BVHNode
from motion_matching.core.pose import PoseSet


class Joint:
    """Class representing a joint in a skeleton."""

    def __init__(self, name, offset, parent):
        self.name = name
        self.offset = np.zeros(3) if parent == -1 else np.array(offset)
        self.parent = parent


class Skeleton:
    """Class representing a skeleton structure."""

    def __init__(self):
        self.LFOOT_INDEX = -1
        self.RFOOT_INDEX = -1

        self.n_joints = 0
        self.joints = []
        self.joint_names = []
        self.edges = []

    def build_skeleton(self, node: BVHNode, parent=-1):
        self.joints.append(Joint(node.name, node.offset, parent))
        self.joint_names.append(node.name)
        node_index = self.n_joints
        self.n_joints += 1

        if parent != -1:
            self.edges.append((parent, node_index))
        if node.name == "LeftFoot":
            self.LFOOT_INDEX = node_index
        if node.name == "RightFoot":
            self.RFOOT_INDEX = node_index
        for child_node in node.children:
            self.build_skeleton(child_node, node_index)

    def apply_pose(self, root_position, root_y_rotation, joint_rotations):
        positions = []
        rotations = []
        for i, joint in enumerate(self.joints):
            if joint.parent == -1:  # Root joint
                parent_position = root_position
                parent_R = R.from_euler("y", root_y_rotation)
            else:
                parent_position = positions[joint.parent]
                parent_R = R.from_euler("xyz", rotations[joint.parent])

            local_R = R.from_euler("xyz", joint_rotations[i])
            position = parent_position + parent_R.apply(joint.offset)
            rotation = (parent_R * local_R).as_euler("xyz")
            positions.append(position)
            rotations.append(rotation)
        return positions, rotations
