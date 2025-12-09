import numpy as np
from .bvh_node import BVHNode


class BVH:
    """Class to handle motion capture data from BVH files."""

    root: BVHNode
    n_frames: int
    frame_time: float
    joints: list[str]
    edges: list[tuple[int, int]]

    def __init__(self, filename):
        self.parse(filename)
        self.joints = self.root.collect()
        edges = []
        self.root.find_edges(edges)
        self.edges = edges

    def parse(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        reading_motion = False
        for line_index, line in enumerate(lines):
            line = line.strip()
            if line.startswith("HIERARCHY"):
                continue
            elif line.startswith("ROOT"):
                name = line.split()[1]
                self.root = BVHNode(name)
                self.root.parse(lines, line_index + 1)
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

    def calculate_positions_rotations(self):
        n_joints = len(self.joints)
        positions = np.zeros((self.n_frames, n_joints, 3))
        rotations = np.zeros((self.n_frames, n_joints, 3))
        for frame in range(self.n_frames):
            self.root.calculate(frame, positions, rotations)
        return positions, rotations

    def print(self):
        self.root.print()
