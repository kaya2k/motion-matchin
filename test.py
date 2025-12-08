import rerun as rr
import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)


def build_oriented_box(p1, p2, width=0.2, height=0.2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    center = (p1 + p2) / 2
    direction = normalize(p2 - p1)
    length = np.linalg.norm(p2 - p1)

    local_vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )

    scale = np.array([length, width, height])
    local_vertices = local_vertices * scale

    x_axis = np.array([1.0, 0.0, 0.0])
    v = normalize(direction)
    axis = np.cross(x_axis, v)
    angle = np.arccos(np.dot(x_axis, v))

    if np.linalg.norm(axis) < 1e-8:
        R = np.eye(3)
    else:
        axis = normalize(axis)
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    rotated_vertices = (R @ local_vertices.T).T

    # 4. 중심 위치로 이동
    world_vertices = rotated_vertices + center

    # 5. 삼각형 인덱스
    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ]
    )

    return world_vertices, indices


rr.init("oriented-box-demo", spawn=True)

p1 = [0, 0, 0]
p2 = [1, 0.5, 2]

vertices, idx = build_oriented_box(p1, p2, width=0.3, height=0.4)

rr.log("oriented_box", rr.Mesh3D(vertex_positions=vertices, triangle_indices=idx))
