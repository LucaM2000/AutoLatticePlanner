import json
import os
import time

import click

from auto_lattice_planner import lattice_planner_helper

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--dicom_series_path",
    "-d",
    help="CT Dicom series folder path",
    required=True,
    type=click.Path(dir_okay=True),
)
@click.option(
    "--rtstruct_path",
    "-s",
    help="RTStruct serie pathname",
    required=True,
    type=click.Path(dir_okay=False),
)
@click.option(
    "--primary_roi_name",
    "-p",
    help="Primary ROI name",
    required=True,
    default="gtv",
)
@click.option(
    "--lattice_geometry",
    "-l",
    help="lattice geometry",
    required=True,
    type=click.Choice(["Normal (Duriseti)", "product"]),
)
@click.option(
    "--tolerance",
    "-r",
    help="tolerance",
    required=True,
    default=1.0,
)
@click.option(
    "--secondary_roi_name",
    "-se",
    help="Secondary ROI name",
    default=None,
    type=str,
)
@click.option(
    "--oar_roi_names",
    "-oar",
    help="OAR ROI names",
    default=None,
    type=str,
)
@click.option(
    "--output_folder",
    "-o",
    help="Output folder path",
    default=".",
    type=click.Path(dir_okay=True),
)
def main(
    dicom_series_path,
    rtstruct_path,
    primary_roi_name,
    lattice_geometry,
    tolerance,
    secondary_roi_name,
    oar_roi_names,
    output_folder,
):

    dicom_series_path = os.path.normpath(dicom_series_path)

    primary_roi = primary_roi_name
    lattice_geometry = lattice_geometry
    print(f"Primary ROI selected: {primary_roi}")

    secondary_roi = secondary_roi_name

    additional_target = False
    if secondary_roi is not None:
        print(f"Additional ROI selected: {secondary_roi}")
        additional_target = True
    else:
        print("No additional ROI selected.")

    selected_oars = oar_roi_names
    print("OARs to avoid:", selected_oars)

    # Create the lattice data object
    data = lattice_planner_helper.LatticeData(
        dicom_series_path=dicom_series_path,
        rtstruct_path=rtstruct_path,
        additional_target=additional_target,
        lattice_geometry=lattice_geometry,
        tolerance=tolerance,
        primary_roi=[primary_roi],
        secondary_roi=secondary_roi,
        selected_oars=[selected_oars] if selected_oars is not None else [],
        output_folder=output_folder,
    )

    with open("paths.json", "w") as f:
        json.dump(data.to_json(), f, indent=4)

    start = time.time()
    with open("timer.txt", "w") as f:
        f.write(str(start))

    lattice_planner_helper.compute_volumes(data)


if __name__ == "__main__":

    main()
