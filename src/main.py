"""Launch the exercise reference-extraction and real-time comparison tools."""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox

# Map each exercise to its extraction script, comparison script, reference,
# and optional icon.
EXERCISES = {
    "Squat": {
        "extractor_script": os.path.join("src", "reference_extraction", "reference_extraction_squat.py"),
        "comparison_script": os.path.join("src", "realtime_comparison", "realtime_comparison_squat.py"),
        "reference_file": "squat_reference.npy",
        "icon_path": os.path.join("assets", "icons", "squat.png"),
        "emoji": "🏋",
    },
    "Lunge Right": {
        "extractor_script": os.path.join("src", "reference_extraction", "reference_extraction_lunge.py"),
        "comparison_script": os.path.join("src", "realtime_comparison", "realtime_comparison_lunge_movements.py"),
        "reference_file": "lunge_right_reference.npy",
        "icon_path": os.path.join("assets", "icons", "lunge_right.png"),
        "emoji": "🦵➡",
    },
    "Lunge Left": {
        "extractor_script": os.path.join("src", "reference_extraction", "reference_extraction_lunge.py"),
        "comparison_script": os.path.join("src", "realtime_comparison", "realtime_comparison_lunge_movements.py"),
        "reference_file": "lunge_left_reference.npy",
        "icon_path": os.path.join("assets", "icons", "lunge_left.png"),
        "emoji": "⬅🦵",
    },
    "Limb Extension Right": {
        "extractor_script": os.path.join("src", "reference_extraction", "reference_extraction_limb_extensions.py"),
        "comparison_script": os.path.join("src", "realtime_comparison", "realtime_comparison_limb_extensions.py"),
        "reference_file": "limb_extension_right_reference.npy",
        "icon_path": os.path.join("assets", "icons", "limb_extension_right.png"),
        "emoji": "💪➡",
    },
    "Limb Extension Left": {
        "extractor_script": os.path.join("src", "reference_extraction", "reference_extraction_limb_extensions.py"),
        "comparison_script": os.path.join("src", "realtime_comparison", "realtime_comparison_limb_extensions.py"),
        "reference_file": "limb_extension_left_reference.npy",
        "icon_path": os.path.join("assets", "icons", "limb_extension_left.png"),
        "emoji": "⬅💪",
    },
    "Jumping Jacks": {
        "extractor_script": os.path.join("src", "reference_extraction", "reference_extraction_jumping_jacks.py"),
        "comparison_script": os.path.join("src", "realtime_comparison", "realtime_comparison_jumping_jacks.py"),
        "reference_file": "jumping_jacks_reference.npy",
        "icon_path": os.path.join("assets", "icons", "jumping_jacks.png"),
        "emoji": "🤸",
    },
}

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class PhysicalTherapyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("In-Home Physical Therapy - Movement Selection")
        self.root.configure(bg="#2c3e50")
        self.root.geometry("560x420")

        # Keep references to PhotoImage objects alive (Tkinter drops them
        # otherwise, since it only holds a weak reference internally).
        self._icon_images = {}

        title_label = tk.Label(
            root,
            text="In-Home Physical Therapy System",
            font=("Helvetica", 16, "bold"),
            fg="#ecf0f1",
            bg="#2c3e50"
        )
        title_label.pack(pady=15)

        subtitle_label = tk.Label(
            root,
            text="Tap an exercise to start it",
            font=("Helvetica", 11),
            fg="#bdc3c7",
            bg="#2c3e50"
        )
        subtitle_label.pack(pady=(0, 15))

        # Each button runs extraction and then starts real-time comparison.
        grid_frame = tk.Frame(root, bg="#2c3e50")
        grid_frame.pack(pady=5)

        COLUMNS = 3
        for idx, ex_name in enumerate(EXERCISES.keys()):
            row, col = divmod(idx, COLUMNS)
            self._build_exercise_button(grid_frame, ex_name, row, col)

        self.status_label = tk.Label(
            root,
            text="Ready.",
            font=("Helvetica", 10, "italic"),
            fg="#3498db",
            bg="#2c3e50"
        )
        self.status_label.pack(pady=15)

        exit_btn = tk.Button(
            root,
            text="Exit",
            font=("Helvetica", 10),
            bg="#e74c3c",
            fg="white",
            command=root.quit
        )
        exit_btn.pack(pady=5)

    def _load_icon(self, icon_path):
        """Load an icon if available; otherwise let the caller use an emoji."""
        full_path = os.path.join(PROJECT_ROOT, icon_path)
        if not os.path.exists(full_path):
            return None
        try:
            return tk.PhotoImage(file=full_path)
        except Exception as e:
            print(f"[WARNING] Could not load icon '{full_path}': {e}")
            return None

    def _build_exercise_button(self, parent, ex_name, row, col):
        ex_info = EXERCISES[ex_name]
        icon_img = self._load_icon(ex_info["icon_path"])

        cell = tk.Frame(parent, bg="#2c3e50")
        cell.grid(row=row, column=col, padx=12, pady=12)

        if icon_img is not None:
            self._icon_images[ex_name] = icon_img  # keep alive
            btn = tk.Button(
                cell,
                image=icon_img,
                bg="#34495e",
                activebackground="#3d566e",
                relief=tk.RAISED,
                width=90,
                height=90,
                command=lambda name=ex_name: self.start_exercise(name)
            )
        else:
            # Use the configured emoji when no image file is available.
            btn = tk.Button(
                cell,
                text=ex_info["emoji"],
                font=("Segoe UI Emoji", 28),
                bg="#34495e",
                fg="white",
                activebackground="#3d566e",
                relief=tk.RAISED,
                width=4,
                height=2,
                command=lambda name=ex_name: self.start_exercise(name)
            )
        btn.pack()

        label = tk.Label(
            cell,
            text=ex_name,
            font=("Helvetica", 9),
            fg="#ecf0f1",
            bg="#2c3e50",
            wraplength=100,
            justify="center"
        )
        label.pack(pady=(4, 0))

    def _bring_window_to_front(self):
        """Restore the launcher and bring it to the foreground."""
        self.root.deiconify()
        self.root.state('normal')
        self.root.update()

        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.lift()
        self.root.focus_force()

        if sys.platform.startswith('win'):
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception as e:
                print(f"[WARNING] SetForegroundWindow failed: {e}")

        # Release topmost after the window has been restored.
        self.root.after(300, lambda: self.root.attributes('-topmost', False))

    def start_exercise(self, ex_name):
        ex_info = EXERCISES.get(ex_name)

        if not ex_info:
            messagebox.showerror("Error", "Exercise not found!")
            return

        extractor = ex_info["extractor_script"]
        comparison = ex_info["comparison_script"]

        try:
            # Build the selected exercise reference curve.
            self.status_label.config(text=f"Updating reference for: {ex_name}...", fg="#f1c40f")
            self.root.update_idletasks()

            if os.path.exists(extractor):
                print(f"[INFO] Running reference extraction script: {extractor}")
                subprocess.run([sys.executable, extractor], check=True)
            else:
                print(f"[WARNING] Extraction script '{extractor}' not found. Using existing reference if available.")

            # Start the real-time comparison tool.
            self.status_label.config(text="Running real-time evaluation...", fg="#2ecc71")
            self.root.update_idletasks()

            self.root.iconify()

            if os.path.exists(comparison):
                print(f"[INFO] Running comparison script: {comparison}")
                subprocess.run([sys.executable, comparison], check=True)
            else:
                # Bring the window back BEFORE showing the error dialog,
                # otherwise the dialog can also end up hidden behind other
                # windows while the launcher is still minimized.
                self._bring_window_to_front()
                messagebox.showerror("Error", f"Real-time comparison script not found: {comparison}")
                self.status_label.config(text="Ready.", fg="#3498db")
                return

            self._bring_window_to_front()
            self.status_label.config(text="Evaluation complete. Choose another exercise.", fg="#3498db")

        except subprocess.CalledProcessError as e:
            self._bring_window_to_front()
            self.status_label.config(text="Error during execution.", fg="#e74c3c")
            messagebox.showerror("Execution Error", f"An error occurred while running the script:\n{e}")
        except Exception as e:
            self._bring_window_to_front()
            self.status_label.config(text="Unexpected error.", fg="#e74c3c")
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PhysicalTherapyApp(root)
    root.mainloop()