import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from auto_lattice_planner import lattice_planner_helper


def main():

    output = Path(".")
    with open(output / "paths.json", "r") as f:
        data = lattice_planner_helper.LatticeData.from_json(json.load(f))
    rtstruct_new = lattice_planner_helper.convert_to_rtstruct(data)

    def get_rtstruct_labels():
        result = {}

        def submit():
            result["rtstruct_series_desc"] = rtstruct_series_desc.get()
            result["ds_series_desc"] = ds_series_desc.get()
            result["structure_set_label"] = structure_set_label.get()
            result["structure_set_name"] = structure_set_name.get()
            result["series_number"] = int(series_number.get())
            root.destroy()

        root = tk.Tk()
        root.title("RTSTRUCT Label Editor")
        root.resizable(False, False)

        padding = {"padx": 10, "pady": 5}

        # Variables with defaults
        rtstruct_series_desc = tk.StringVar(value="Lattice Generator")
        ds_series_desc = tk.StringVar(value="LucaScript")
        structure_set_label = tk.StringVar(value="Luca-Lat")
        structure_set_name = tk.StringVar(value="Luca-Lat")
        series_number = tk.StringVar(value="990")

        # Layout
        ttk.Label(root, text="RTSTRUCT Series Description").grid(
            row=0, column=0, sticky="w", **padding
        )
        ttk.Entry(root, textvariable=rtstruct_series_desc, width=30).grid(
            row=0, column=1, **padding
        )

        ttk.Label(root, text="DS Series Description").grid(
            row=1, column=0, sticky="w", **padding
        )
        ttk.Entry(root, textvariable=ds_series_desc, width=30).grid(
            row=1, column=1, **padding
        )

        ttk.Label(root, text="Structure Set Label").grid(
            row=2, column=0, sticky="w", **padding
        )
        ttk.Entry(root, textvariable=structure_set_label, width=30).grid(
            row=2, column=1, **padding
        )

        ttk.Label(root, text="Structure Set Name").grid(
            row=3, column=0, sticky="w", **padding
        )
        ttk.Entry(root, textvariable=structure_set_name, width=30).grid(
            row=3, column=1, **padding
        )

        ttk.Label(root, text="Series Number").grid(
            row=4, column=0, sticky="w", **padding
        )
        ttk.Entry(root, textvariable=series_number, width=30).grid(
            row=4, column=1, **padding
        )

        ttk.Button(root, text="Apply", command=submit).grid(
            row=5, column=0, columnspan=2, pady=15
        )

        root.mainloop()
        return result

    print("DEBUG: reached file save section")

    # 1 Ask for file path
    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()

        file_path = filedialog.asksaveasfilename(
            defaultextension=".dcm",
            filetypes=[("DICOM files", "*.dcm")],
            title="Save RTSTRUCT As",
        )
        root.destroy()
    except Exception as e:
        print("! File dialog failed, using default path instead:", e)
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path = os.path.join(desktop, "default_rtstruct.dcm")

    if not file_path:
        print("Save cancelled or no file path returned.")
        exit()

    labels = get_rtstruct_labels()
    rtstruct_new = lattice_planner_helper.save_rtstruct(rtstruct_new, file_path, labels)

    print("DEBUG: file path selected ->", file_path)


if __name__ == "__main__":
    main()
