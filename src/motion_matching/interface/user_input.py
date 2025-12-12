import hid
import numpy as np
from pynput import keyboard


class UserInput:
    """Class to handle joystick/keyboard input."""

    def __init__(self):
        self.joystick = None
        self.keys = set()
        self.init_joystick()
        self.init_keyboard()

    def init_joystick(self):
        VID, PID = None, None
        for d in hid.enumerate():
            if "DualSense" in d["product_string"]:
                VID = d["vendor_id"]
                PID = d["product_id"]

        if VID is None or PID is None:
            print("[USERINPUT] DualSense controller not found, using keyboard")
        else:
            self.joystick = hid.device()
            self.joystick.open(VID, PID)
            print(f"[USERINPUT] Found DualSense controller (VID={VID}, PID={PID})")

    def init_keyboard(self):
        def on_press(key):
            try:
                self.keys.add(key.char)
            except AttributeError:
                self.keys.add(key)

        def on_release(key):
            try:
                self.keys.discard(key.char)
            except AttributeError:
                self.keys.discard(key)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    def get(self):
        if self.joystick:
            SCALE = 128
            joystick_input = self.joystick.read(max_length=64)
            dx = joystick_input[1] - SCALE
            dy = joystick_input[2] - SCALE
            dx = dx if abs(dx) > 10 else 0
            dy = dy if abs(dy) > 10 else 0
            return np.array([dx / SCALE, 0.0, dy / SCALE])
        else:
            dx = 0.0
            dy = 0.0
            if "w" in self.keys:
                dy -= 1.0
            if "s" in self.keys:
                dy += 1.0
            if "a" in self.keys:
                dx -= 1.0
            if "d" in self.keys:
                dx += 1.0
            direction = np.array([dx, 0.0, dy])
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1.0 else direction
            return direction
