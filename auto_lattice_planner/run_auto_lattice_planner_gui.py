from .lattice_grid_target_cli import compute_lattice_grid_target
from .rtstruct_conversion_gui import main as rtstruct_conversion_gui_main
from .select_image_gui import main as select_image_main
import time


def main():
    start_time = time.time()
    select_image_main()
    compute_lattice_grid_target(".")
    end_time = time.time()
    print("Time elapsed: ", end_time - start_time)
    rtstruct_conversion_gui_main()


if __name__ == "__main__":
    main()
