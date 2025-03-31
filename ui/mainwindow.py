import tkinter as tk
import subprocess

def launch_ai_vs_ai():
    try:
        subprocess.Popen(["python", "../environment/main.py"])
    except Exception as e:
        print(f"Failed to open simulation: {e}")

def temp_def_name():
    print("temporary placeholder")

root = tk.Tk()
root.title("B25ITK57 Air Hockey Simulation")
root.geometry("1400x800")
root.configure(bg="#1e1e1e")

def on_enter(e):
    e.widget['background'] = "#2a2a2a"

def on_leave(e):
    e.widget['background'] = "#1e1e1e"

def create_button(text, command):
    frame = tk.Frame(root, bg="#ff4c4c", bd=0)
    frame.pack(pady=10)

    btn = tk.Button(
        frame,
        text=text,
        command=command,
        bg="#1e1e1e",
        fg="#f0f0f0",
        activebackground="#2a2a2a",
        activeforeground="#ffffff",
        font=("Consolas", 14),
        bd=0,
        width=30,
        height=2
    )
    btn.pack(padx=2, pady=2)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    return btn

tk.Label(
    root,
    text="B25ITK57 Air Hockey AI",
    font=("Consolas", 22, "bold"),
    fg="#ffffff",
    bg="#1e1e1e"
).pack(pady=30)

create_button("Manual AI vs Manual AI", launch_ai_vs_ai)
create_button("Not done yet", temp_def_name)
create_button("Not done yet", temp_def_name)
create_button("Not done yet", temp_def_name)

root.mainloop()
