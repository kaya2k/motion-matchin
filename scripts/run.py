import time
from motion_matching.interface.graphic import MotionVisualizer
from motion_matching.interface.user_input import UserInput
from motion_matching.core.controller import MotionMatchingController


def main():
    controller = MotionMatchingController()
    visualizer = MotionVisualizer()
    user_input = UserInput()

    FPS = 30
    while True:
        try:
            start_time = time.perf_counter()
            input_direction = user_input.get()
            visualizer.input_direction = input_direction
            controller.input_direction = input_direction
            controller.update()
            visualizer.update(*controller.get_current_pose())
            elapsed_time = time.perf_counter() - start_time
            time.sleep(max(0, 1 / FPS - elapsed_time))

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
