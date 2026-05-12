import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EMPTY = 0   # Cell empty (path)
SHELF = 1   # Shelf (obstacle)
 
def make_grid(rows, cols):
    """
    Return a rows×cols grid, initially all EMPTY.
    """
    return [[EMPTY] * cols for _ in range(rows)]

def add_shelves(grid):
    """
    Place shelves in the grid.
    Keep outer border, 4 corners always EMPTY.
    """
    grid = np.array(grid)  # Convert to NumPy array for easier slicing
    rows, cols = grid.shape

    grid[1:-1, 1:-1:2] = SHELF # Place shelves in odd columns (1, 3, 5, ...)
    grid[rows//2, :] = EMPTY # Clear the middle row for a way
    return grid

def draw_grid(grid):
    """
    Display the grid: EMPTY cells = white, SHELF cells = gray.
    """
    rows = len(grid)
    cols = len(grid[0])
 
    img = np.ones((rows, cols, 3)) # Create an RGB image starting all EMPTY
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == SHELF:
                img[r, c] = [0.4, 0.4, 0.4]   # Xám đậm = kệ
    # Tô màu xanh nhạt cho 4 góc (vị trí home của robot – bước sau)
    corners = [(0,0), (0,cols-1), (rows-1,0), (rows-1,cols-1)]
    for r, c in corners:
        img[r, c] = [0.7, 0.9, 0.7]   # Light green for corners (robot home)
 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, interpolation="nearest", aspect="equal")
 
    # Set grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
 
    # Add legend
    legend = [
        mpatches.Patch(color=[1,1,1],       label="Way (EMPTY)"),
        mpatches.Patch(color=[0.4,0.4,0.4], label="Shelf (SHELF)"),
        mpatches.Patch(color=[0.7,0.9,0.7], label="Corner – robot home"),
    ]
    ax.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ROWS = 12
    COLS = 16
 
    grid = make_grid(ROWS, COLS)
    grid = add_shelves(grid)
    for i in range(ROWS):
        print(grid[i])
    draw_grid(grid)