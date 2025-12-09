import time
from core.motion_data import MotionData
from core.visualizer import MotionVisualizer


def main():
    bvh_filename = "./data/bvh/walk1_subject1.bvh"
    motiondata = MotionData()
    motiondata.load_from_bvh(bvh_filename)

    visualizer = MotionVisualizer()
    for frame in range(motiondata.n_frames):
        start_time = time.perf_counter()
        visualizer.update(
            motiondata.n_joints,
            motiondata.edge_list,
            motiondata.positions[frame],
            motiondata.rotations[frame],
        )
        elapsed_time = time.perf_counter() - start_time
        time.sleep(max(0, motiondata.frame_time - elapsed_time))


if __name__ == "__main__":
    main()
