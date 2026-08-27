from .lattice_grid_target_cli import compute_lattice_grid_target
from .rtstruct_conversion_gui import main as rtstruct_conversion_gui_main
from .select_image_gui import main as select_image_main
import time


def main():

    select_image_main()

    compute_lattice_grid_target(".")

    with open("timer.txt", "r") as f:
        start = float(f.read())

    spheres_placed_time = time.time()

    print(f"Spheres placed in: {spheres_placed_time - start:.2f} seconds")

    rtstruct_conversion_gui_main()


if __name__ == "__main__":
    main()
