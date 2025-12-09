import time
import numpy as np
import pygame
from core.motion_data import MotionData
from core.visualizer import MotionVisualizer


def main():
    bvh_filename = "./data/bvh/walk1_subject1.bvh"
    motiondata = MotionData()
    motiondata.load_from_bvh(bvh_filename)

    visualizer = MotionVisualizer()

    joy = pygame.joystick.Joystick(0)
    joy.init()

    for frame in range(motiondata.n_frames):
        # Handle joystick input
        pygame.event.pump()
        visualizer.input_direction = np.array([joy.get_axis(0), 0.0, joy.get_axis(1)])

        start_time = time.perf_counter()
        visualizer.update(
            motiondata.joints,
            motiondata.edge_list,
            motiondata.positions[frame],
            motiondata.rotations[frame],
        )
        elapsed_time = time.perf_counter() - start_time
        time.sleep(max(0, motiondata.frame_time - elapsed_time))


if __name__ == "__main__":
    pygame.init()
    pygame.joystick.init()
    main()
