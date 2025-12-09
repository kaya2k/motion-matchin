import hid
import numpy as np


class Joystick:
    def __init__(self):
        VID, PID = None, None
        for d in hid.enumerate():
            if "DualSense" in d["product_string"]:
                VID = d["vendor_id"]
                PID = d["product_id"]

        self.joystick = hid.device()
        self.joystick.open(VID, PID)
        print(f"Found DualSense controller(VID={VID},PID={PID})")

    def input(self):
        SCALE = 128
        joystick_input = self.joystick.read(max_length=64)
        dx = (joystick_input[1] - SCALE) / SCALE
        dy = (joystick_input[2] - SCALE) / SCALE
        return np.array([dx, 0.0, dy])
