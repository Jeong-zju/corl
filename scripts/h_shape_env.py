import numpy as np
import matplotlib.pyplot as plt


def create_h_shape_grid(
    grid_size=(100, 100),
    total_width=6.0,
    total_height=8.0,
    bar_thickness=1.5,
    mid_bar_height=2.0,
    wall_thickness=0.3,
):
    """Create an H-shaped 2D occupancy grid environment.

    The H interior is free space (0 / white), while the H boundary
    and exterior are obstacles (1 / black).  This produces two vertical
    corridors connected by a narrow horizontal corridor.

    Parameters
    ----------
    grid_size : tuple of int
        (rows, cols) of the output occupancy grid.
    total_width : float
        Total width of the H shape in world units.
    total_height : float
        Total height of the H shape in world units.
    bar_thickness : float
        Width of each vertical bar of the H.
    mid_bar_height : float
        Height of the horizontal middle bar.
    wall_thickness : float
        Thickness of the H boundary walls (in world units).

    Returns
    -------
    grid : np.ndarray  (rows, cols), dtype float
        Occupancy grid: 0 = free, 1 = obstacle.
    extent : tuple
        (xmin, xmax, ymin, ymax) for plotting with imshow.
    """
    rows, cols = grid_size
    hw = total_width / 2
    hh = total_height / 2
    bt = bar_thickness
    mh = mid_bar_height / 2
    wt = wall_thickness

    # World coordinate ranges (centered at origin)
    xmin, xmax = -hw - 1.0, hw + 1.0
    ymin, ymax = -hh - 1.0, hh + 1.0

    xs = np.linspace(xmin, xmax, cols)
    ys = np.linspace(ymin, ymax, rows)
    X, Y = np.meshgrid(xs, ys)

    # --- Define three interior corridor regions (without walls) ----------
    # Left vertical corridor interior
    left_inner = (
        (X > -hw + wt) & (X < -hw + bt - wt) &
        (Y > -hh + wt) & (Y < hh - wt)
    )
    # Right vertical corridor interior
    right_inner = (
        (X > hw - bt + wt) & (X < hw - wt) &
        (Y > -hh + wt) & (Y < hh - wt)
    )
    # Middle horizontal corridor interior
    mid_inner = (
        (X > -hw + bt - wt) & (X < hw - bt + wt) &
        (Y > -mh + wt) & (Y < mh - wt)
    )

    # Free space = union of three corridor interiors
    free = left_inner | right_inner | mid_inner

    # Build grid: default obstacle, mark free cells
    grid = np.ones((rows, cols), dtype=np.float64)
    grid[free] = 0.0

    extent = (xmin, xmax, ymin, ymax)
    return grid, extent


def plot_h_env(grid, extent, ax=None, show=True, save_path=None):
    """Plot the H-shaped occupancy grid environment."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    ax.imshow(
        grid,
        cmap="gray_r",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )

    ax.set_aspect("equal")
    ax.set_title("H-Shaped Corridor Environment", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    grid, extent = create_h_shape_grid(
        grid_size=(200, 200),
        total_width=6.0,
        total_height=8.0,
        bar_thickness=1.5,
        mid_bar_height=1.0,
        wall_thickness=0.3,
    )
    print(f"Grid shape: {grid.shape}")
    print(f"Free cells: {int((grid == 0).sum())}")
    print(f"Obstacle cells: {int((grid == 1).sum())}")

    save_path = "/home/jeong/zeno/corl/main/scripts/h_shape_env.png"
    plot_h_env(grid, extent, show=False, save_path=save_path)
