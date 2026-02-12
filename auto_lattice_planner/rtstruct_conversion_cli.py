import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import click

from auto_lattice_planner import lattice_planner_helper

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--rtstruct_path",
    "-s",
    help="RTStruct serie pathname",
    required=True,
    type=click.Path(dir_okay=False),
)
@click.option(
    "--output_folder",
    "-o",
    help="Output folder path",
    default=".",
    type=click.Path(dir_okay=True),
)
def main(rtstruct_path, output_folder):

    output = Path(output_folder)
    with open(output / "paths.json", "r") as f:
        data = lattice_planner_helper.LatticeData.from_json(json.load(f))
    rtstruct_new = lattice_planner_helper.convert_to_rtstruct(data)

    labels = {
        "rtstruct_series_desc": "Lattice Generator",
        "ds_series_desc": "LucaScript",
        "structure_set_label": "Luca-Lat",
        "structure_set_name": "Luca-Lat",
        "series_number": 990,
    }

    rtstruct_new = lattice_planner_helper.save_rtstruct(
        rtstruct_new, rtstruct_path, labels
    )


if __name__ == "__main__":
    main()
