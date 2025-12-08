import numpy as np
from scipy.spatial.transform import Rotation as R


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

    def parse(self, lines, line_index):
        while line_index < len(lines):
            line = lines[line_index].strip()
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
                line_index = child_node.parse(lines, line_index + 1)
                self.children.append(child_node)
            elif line.startswith("End Site"):
                endsite = BVHNode("End Site")
                line_index = endsite.parse(lines, line_index + 1)
            elif line.startswith("}"):
                return line_index
            line_index += 1

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

    def calculate(self, frame, positions, rotations, node_index=0, parent_index=-1):
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
            # Parent rotation is saved in radians
            parent_R = R.from_euler("xyz", parent_rotation, degrees=False)
            posision_to_parent = local_R.as_matrix() @ self.offset
            # Calculate global position
            position = parent_R.as_matrix() @ (parent_position + posision_to_parent)
            # Calculate global rotation in radians
            rotation = (parent_R * local_R).as_euler("xyz")
            positions[frame, node_index] = position
            rotations[frame, node_index] = rotation

        parent_index = node_index
        for child in self.children:
            node_index += 1
            node_index = child.calculate(
                frame, positions, rotations, node_index, parent_index
            )

        return node_index

    def print(self, indent=0):
        print(" " * indent + self.name)
        for child in self.children:
            child.print(indent + 1)
