from .lattice_grid_target_cli import compute_lattice_grid_target
from .rtstruct_conversion_gui import main as rtstruct_conversion_gui_main
from .select_image_gui import main as select_image_main


def main():
    select_image_main()
    compute_lattice_grid_target(".")
    rtstruct_conversion_gui_main()


if __name__ == "__main__":
    main()
