import numpy as np
from scipy.spatial.transform import Rotation as R


class BVH:
    """Class to handle motion capture data from BVH files."""

    root: "BVHNode"
    n_frames: int
    frame_time: float

    def __init__(self, filename):
        self.parse(filename)

    def parse(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        reading_motion = False
        for index, line in enumerate(lines):
            line = line.strip()
            if line.startswith("HIERARCHY"):
                continue
            elif line.startswith("ROOT"):
                name = line.split()[1]
                self.root = BVHNode(name)
                self.root.parse(lines, index + 1)
            elif line.startswith("MOTION"):
                reading_motion = True
            elif reading_motion:
                if line.startswith("Frames:"):
                    self.n_frames = int(line.split()[1])
                elif line.startswith("Frame Time:"):
                    self.frame_time = float(line.split()[2])
                else:  # Channel values for each frame
                    channel_values = list(map(float, line.split()))
                    self.root.add_channel_values(channel_values)

    def get_motion_data(self):
        n_joints = self.root.count_nodes()
        edge_list = []
        self.root.find_edges(edge_list)
        positions = np.zeros((self.n_frames, n_joints, 3))
        rotations = np.zeros((self.n_frames, n_joints, 3))

        for frame_idx in range(self.n_frames):
            self.root.fill_motion_data(frame_idx, positions, rotations)

        return n_joints, edge_list, positions, rotations


class BVHNode:
    """Class to represent a node in the BVH hierarchy."""

    name: str
    offset: np.ndarray
    n_channels: int
    channels: list[str]
    channel_values: list[list[float]]
    children: list["BVHNode"]

    def __init__(self, name):
        self.name = name
        self.children = []
        self.channel_values = []

    def parse(self, lines, index):
        while index < len(lines):
            line = lines[index].strip()
            if line.startswith("{"):
                pass
            elif line.startswith("OFFSET"):
                self.offset = np.array(list(map(float, line.split()[1:])))
            elif line.startswith("CHANNELS"):
                self.n_channels = int(line.split()[1])
                self.channels = line.split()[2:]
            elif line.startswith("JOINT"):
                name = line.split()[1]
                child_node = BVHNode(name)
                index = child_node.parse(lines, index + 1)
                self.children.append(child_node)
            elif line.startswith("End Site"):
                endsite = BVHNode("End Site")
                index = endsite.parse(lines, index + 1)
            elif line.startswith("}"):
                return index
            index += 1

        raise ValueError("Malformed BVH file: missing closing '}'")

    def add_channel_values(self, values):
        self.channel_values.append(values[: self.n_channels])
        for child in self.children:
            child.add_channel_values(values[self.n_channels :])

    def count_nodes(self):
        count = 1
        for child in self.children:
            count += child.count_nodes()
        return count

    def find_edges(self, edges, node_index=0):
        child_index = node_index + 1
        for child in self.children:
            edges.append((node_index, child_index))
            child_index = child.find_edges(edges, child_index)
        return child_index

    def fill_motion_data(
        self, frame, positions, rotations, node_index=0, parent_index=-1
    ):
        if node_index == 0:  # ROOT node
            position = np.array(self.channel_values[frame][:3])
            rotation = np.array(self.channel_values[frame][3:6])
            # Convert rotation from degrees to radians
            rotation = R.from_euler("zyx", rotation, degrees=True).as_euler("xyz")
            positions[frame, node_index] = position
            rotations[frame, node_index] = rotation
        else:
            parent_position = positions[frame, parent_index]
            parent_rotation = rotations[frame, parent_index]
            local_rotation = np.array(self.channel_values[frame][:])
            local_R = R.from_euler("zyx", local_rotation, degrees=True)
            parent_R = R.from_euler("xyz", parent_rotation, degrees=True)
            posision_to_parent = local_R.as_matrix() @ self.offset
            # Calculate global position
            position = parent_R.as_matrix() @ (parent_position + posision_to_parent)
            # Calculate global rotation in radians
            rotation = (parent_R * local_R).as_euler("xyz")
            positions[frame, node_index] = position
            rotations[frame, node_index] = rotation
