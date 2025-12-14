class BVHNode:
    """Class to represent a node in the BVH hierarchy."""

    def __init__(self, name):
        self.name = name
        self.offset = []
        self.channels = []
        self.channel_values = []
        self.children = []

    def parse(self, lines, line_index):
        while line_index < len(lines):
            line = lines[line_index].strip()
            if line.startswith("{"):
                pass
            elif line.startswith("OFFSET"):
                self.offset = list(map(float, line.split()[1:]))
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

        raise ValueError("BVHNode.parse: missing closing '}'")

    def add_channel_values(self, values, index=0):
        channel_values = values[index : index + self.n_channels]
        self.channel_values.append(channel_values)
        index += self.n_channels
        for child in self.children:
            index = child.add_channel_values(values, index)
        return index

    def print(self, indent=0):
        print(" " * indent + self.name)
        for child in self.children:
            child.print(indent + 1)
