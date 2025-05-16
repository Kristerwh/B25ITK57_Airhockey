import tkinter as tk
import threading
import keyboard
import subprocess
import os
import sys
import time

overlay_thread = None

def block_input_overlay():
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.attributes('-topmost', True)
    root.attributes('-alpha', 0.01)
    root.config(cursor='none')
    root.configure(bg='black')
    root.lift()
    root.focus_force()
    root.after(500, lambda: (root.lift(), root.focus_force()))
    root.bind("<Button>", lambda e: "break")
    root.bind("<Motion>", lambda e: "break")
    root.bind("<ButtonRelease>", lambda e: "break")
    root.bind("<Key>", lambda e: "break")
    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()

def launch_overlay():
    global overlay_thread
    if overlay_thread is None or not overlay_thread.is_alive():
        overlay_thread = threading.Thread(target=block_input_overlay, daemon=True)
        overlay_thread.start()

def monitor_keys():
    while True:
        if keyboard.is_pressed('o'):
            launch_overlay()
            time.sleep(0.5)

if __name__ == "__main__":
    launch_overlay()

    threading.Thread(target=monitor_keys, daemon=True).start()

    script_path = os.path.join(os.path.dirname(__file__), "main_copy_for_ui_testing_touchscreen.py")
    subprocess.run([sys.executable, script_path])
