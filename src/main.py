import sys
import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# Exercise Mapping Configuration:
# Map each exercise name to its corresponding reference extraction script, 
# real-time comparison script, and .npy reference file.
EXERCISES = {
    "Squat": {
        "extractor_script": os.path.join("src", "reference_extraction_squat.py"),
        "comparison_script": os.path.join("src", "realtime_comparison_squat.py"),
        "reference_file": "squat_reference.npy"
    },
    "Lunge Right": {
        "extractor_script": os.path.join("src", "reference_extraction_lunge.py"),
        "comparison_script": os.path.join("src", "realtime_comparison_lunge_movements.py"),
        "reference_file": "lunge_right_reference.npy"
    },
    "Lunge Left": {
        "extractor_script": os.path.join("src", "reference_extraction_lunge.py"),
        "comparison_script": os.path.join("src", "realtime_comparison_lunge_movements.py"),
        "reference_file": "lunge_left_reference.npy"
    },
    "Limb Extension Right": {
        "extractor_script": os.path.join("src", "reference_extraction_limb_extensions.py"),
        "comparison_script": os.path.join("src", "realtime_comparison_limb_extensions.py"),
        "reference_file": "limb_extension_right_reference.npy"
    },
    "Limb Extension Left": {
        "extractor_script": os.path.join("src", "reference_extraction_limb_extensions.py"),
        "comparison_script": os.path.join("src", "realtime_comparison_limb_extensions.py"),
        "reference_file": "limb_extension_left_reference.npy"
    },
    "Jumping Jacks": {
        "extractor_script": os.path.join("src", "reference_extraction_jumping_jacks.py"),
        "comparison_script": os.path.join("src", "realtime_comparison_jumping_jacks.py"),
        "reference_file": "jumping_jacks_reference.npy"
    }
}

class PhysicalTherapyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("In-Home Physical Therapy - Movement Selection")
        self.root.geometry("500x380")
        self.root.configure(bg="#2c3e50")

        # Main Title
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
            text="Select the exercise to perform:", 
            font=("Helvetica", 11), 
            fg="#bdc3c7", 
            bg="#2c3e50"
        )
        subtitle_label.pack(pady=5)

        # Dropdown Selection Menu
        self.selected_exercise = tk.StringVar(value=list(EXERCISES.keys())[0])
        
        dropdown_frame = tk.Frame(root, bg="#2c3e50")
        dropdown_frame.pack(pady=15)

        dropdown = ttk.Combobox(
            dropdown_frame, 
            textvariable=self.selected_exercise, 
            values=list(EXERCISES.keys()),
            state="readonly",
            width=32,
            font=("Helvetica", 10)
        )
        dropdown.pack()

        # Status Label to provide feedback (e.g., "Updating reference...")
        self.status_label = tk.Label(
            root, 
            text="Ready.", 
            font=("Helvetica", 10, "italic"), 
            fg="#3498db", 
            bg="#2c3e50"
        )
        self.status_label.pack(pady=10)

        # Action Buttons
        start_btn = tk.Button(
            root, 
            text="Regenerate Reference & Start", 
            font=("Helvetica", 11, "bold"),
            bg="#27ae60", 
            fg="white", 
            activebackground="#2ecc71",
            activeforeground="white",
            relief=tk.RAISED,
            padx=15,
            pady=8,
            command=self.start_exercise
        )
        start_btn.pack(pady=15)

        exit_btn = tk.Button(
            root, 
            text="Exit", 
            font=("Helvetica", 10),
            bg="#e74c3c", 
            fg="white", 
            command=root.quit
        )
        exit_btn.pack(pady=5)

    def start_exercise(self):
        ex_name = self.selected_exercise.get()
        ex_info = EXERCISES.get(ex_name)

        if not ex_info:
            messagebox.showerror("Error", "Exercise not found!")
            return

        extractor = ex_info["extractor_script"]
        comparison = ex_info["comparison_script"]

        try:
            # 1. Reference Extraction Stage
            self.status_label.config(text=f"Updating reference for: {ex_name}...", fg="#f1c40f")
            self.root.update_idletasks()

            if os.path.exists(extractor):
                # Execute the reference extraction script to update/generate the .npy file
                print(f"[INFO] Running reference extraction script: {extractor}")
                subprocess.run([sys.executable, extractor], check=True)
            else:
                print(f"[WARNING] Extraction script '{extractor}' not found. Using existing reference if available.")

            # 2. Real-time Comparison Stage
            self.status_label.config(text="Running real-time evaluation...", fg="#2ecc71")
            self.root.update_idletasks()
            
            # Minimize launcher GUI during execution
            self.root.iconify()
            
            if os.path.exists(comparison):
                print(f"[INFO] Running comparison script: {comparison}")
                subprocess.run([sys.executable, comparison], check=True)
            else:
                messagebox.showerror("Error", f"Real-time comparison script not found: {comparison}")

            # Restore launcher GUI after execution finishes
            self.root.deiconify()
            self.status_label.config(text="Evaluation complete. Choose another exercise.", fg="#3498db")

        except subprocess.CalledProcessError as e:
            self.root.deiconify()
            self.status_label.config(text="Error during execution.", fg="#e74c3c")
            messagebox.showerror("Execution Error", f"An error occurred while running the script:\n{e}")
        except Exception as e:
            self.root.deiconify()
            self.status_label.config(text="Unexpected error.", fg="#e74c3c")
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PhysicalTherapyApp(root)
    root.mainloop()