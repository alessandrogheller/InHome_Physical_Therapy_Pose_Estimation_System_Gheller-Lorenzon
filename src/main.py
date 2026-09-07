import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# Dizionario che associa il nome dell'esercizio allo script o al file di riferimento .npy
EXERCISES = {
    "Squat": {
        "script": "src/realtime_comparison_squat.py", # oppure lo script specifico che usi per lo squat
        "reference": "squat_reference.npy"
    },
    "Lunge Right": {
        "script": "src/realtime_comparison_lunge_movements.py",
        "reference": "lunge_right_reference.npy"
    },
    "Lunge Left": {
        "script": "src/realtime_comparison_lunge_movements.py",
        "reference": "lunge_left_reference.npy"
    },
    "Limb Extension Right": {
        "script": "src/realtime_comparison_limb_extensions.py",
        "reference": "limb_extension_right_reference.npy"
    },
    "Limb Extension Left": {
        "script": "src/realtime_comparison_limb_extensions.py",
        "reference": "limb_extension_left_reference.npy"
    },
    "jumping jacks": {
        "script": "src/realtime_comparison_jumping_jacks.py",
        "reference": "jumping_jacks_reference.npy"
    }
}

class PhysicalTherapyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("In-Home Physical Therapy - Selection")
        self.root.geometry("450x350")
        self.root.configure(bg="#2c3e50")

        # Titolo
        title_label = tk.Label(
            root, 
            text="Seleziona l'Esercizio", 
            font=("Helvetica", 16, "bold"), 
            fg="#ecf0f1", 
            bg="#2c3e50"
        )
        title_label.pack(pady=20)

        # Dropdown / Menu di selezione
        self.selected_exercise = tk.StringVar(value=list(EXERCISES.keys())[0])
        
        dropdown_frame = tk.Frame(root, bg="#2c3e50")
        dropdown_frame.pack(pady=10)

        label_select = tk.Label(
            dropdown_frame, 
            text="Movimento:", 
            font=("Helvetica", 11), 
            fg="#ecf0f1", 
            bg="#2c3e50"
        )
        label_select.pack(side=tk.LEFT, padx=10)

        dropdown = ttk.Combobox(
            dropdown_frame, 
            textvariable=self.selected_exercise, 
            values=list(EXERCISES.keys()),
            state="readonly",
            width=25,
            font=("Helvetica", 10)
        )
        dropdown.pack(side=tk.LEFT)

        # Bottone di avvio
        start_btn = tk.Button(
            root, 
            text="Avvia Valutazione", 
            font=("Helvetica", 12, "bold"),
            bg="#27ae60", 
            fg="white", 
            activebackground="#2ecc71",
            activeforeground="white",
            relief=tk.RAISED,
            padx=15,
            pady=8,
            command=self.start_exercise
        )
        start_btn.pack(pady=30)

        # Bottone Esci
        exit_btn = tk.Button(
            root, 
            text="Esci", 
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
            messagebox.showerror("Errore", "Esercizio non trovato!")
            return

        script_path = ex_info["script"]
        ref_file = ex_info["reference"]

        # Esegue lo script python corrispondente passandogli l'argomento del file di riferimento
        try:
            # Riduciamo la finestra principale durante l'esecuzione
            self.root.iconify()
            
            # Esegue lo script Python come processo separato
            # Esempio comando: python src/realtime_comparison.py --ref lunge_right_reference.npy
            cmd = [sys.executable, script_path, "--ref", ref_file]
            subprocess.run(cmd, check=True)
            
            # Ripristina la finestra al termine dell'esercizio
            self.root.deiconify()
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("Errore nell'esecuzione", f"Impossibile avviare lo script:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PhysicalTherapyApp(root)
    root.mainloop()