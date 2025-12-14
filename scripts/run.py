import time
from motion_matching.core.controller import MotionMatchingController
from motion_matching.interface.visualizer import MotionVisualizer
from motion_matching.interface.user_input import UserInput


def main():
    controller = MotionMatchingController()
    visualizer = MotionVisualizer()
    user_input = UserInput()

    FPS = 30
    while True:
        try:
            start_time = time.perf_counter()
            input_direction = user_input.get()
            controller.update(input_direction)
            visualizer.update(
                controller.skeleton.joint_names,
                controller.skeleton.edges,
                *controller.get_global_positions_rotations(),
                controller.future_trajectories,
                controller.future_directions,
                controller.is_toe_contact,
                input_direction,
            )
            elapsed_time = time.perf_counter() - start_time
            time.sleep(max(0, 1 / FPS - elapsed_time))

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
