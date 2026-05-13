import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Constants for cell types
EMPTY = 0   # Cell empty (path)
SHELF = 1   # Shelf (obstacle)

# Constants for robot states
STATE_IDLE      = "idle"
STATE_MOVING    = "moving_to_good"
STATE_RETURNING = "returning"

# Colors for robots (up to 8)
ROBOT_COLOURS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
                 "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

# Possible movement directions (up, down, left, right)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class Robot:
    """
    Automatic warehouse robot.
 
    robot_id  : unique identifier (1, 2, 3, ...)
    row, col  : current position on the grid
    home_row, home_col : home corner (unchanged)
    state     : current state ("idle" / "moving_to_good" / "returning")
    carrying  : True if carrying goods
    goods_delivered : number of goods delivered successfully
    """
 
    def __init__(self, robot_id, row, col):
        self.robot_id        = robot_id
        self.row             = row
        self.col             = col
        self.home_row        = row
        self.home_col        = col
        self.state           = STATE_IDLE
        self.carrying        = False
        self.goods_delivered = 0
 
    def __repr__(self):
        return (f"Robot({self.robot_id}) "
                f"pos=({self.row},{self.col}) "
                f"home=({self.home_row},{self.home_col}) "
                f"state={self.state}")

class Good:
    """
    A good in the warehouse
 
    good_id   : unique identifier (1, 2, 3, ...)
    row, col  : position on the grid
    available : True  → not taken yet
                False → robot has reserved it
    """
 
    counter = 0   # Global counter for all Good instances
 
    def __init__(self, row, col):
        Good.counter += 1
        self.good_id   = Good.counter
        self.row       = row
        self.col       = col
        self.available = True
 
    def __repr__(self):
        status = "available" if self.available else "reserved"
        return f"Good({self.good_id}) @ ({self.row},{self.col}) [{status}]"

# ── Hàm tìm hàng gần nhất (trả lời Prompt 1, 2, 3) ──────────────────────────
def find_nearest_good(goods, robot_row, robot_col):
    """
    Find the nearest available good to the robot's current position.
 
    - Only consider goods that are still available (available=True)
    - Return the Good object (not an index) → robot will set available=False
    - If no available goods remain → return None
    """
    best      = None
    best_dist = float("inf")
 
    for good in goods:
        if not good.available:
            continue   # Bỏ qua kiện đã bị đặt chỗ
 
        dist = abs(good.row - robot_row) + abs(good.col - robot_col)
        if dist < best_dist:
            best_dist = dist
            best      = good
 
    return best   # None nếu không còn hàng available

def make_robots(grid, num_robots):
    """
    Create num_robots Robot instances, placing them in the 4 corners of the grid.
    If num_robots > 4, the additional robots share corners (rotating around).
 
    Returns: list of Robot instances
    """
    rows = len(grid)
    cols = len(grid[0])
 
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
 
    return [Robot(i + 1, *corners[i % 4]) for i in range(num_robots)]

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

def reachable_shelf_cells(grid):
    """
    Return a list of SHELF cells that are reachable by a robot,
    i.e., have at least one adjacent EMPTY cell (where the robot can stand to pick up the good).
    """
    rows = len(grid)
    cols = len(grid[0])
    result = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != SHELF:
                continue   # Chỉ xét ô kệ
            # Kiểm tra 4 ô xung quanh có ô lối đi không
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and grid[nr][nc] == EMPTY):
                    result.append((r, c))
                    break   # Có 1 ô lối đi cạnh là đủ

    return result
 
# ── MỚI: Tạo danh sách hàng hóa ─────────────────────────────────────────────
def make_goods(grid, num_goods):
    """
    Đặt num_goods kiện hàng lên các ô KỆ có thể tiếp cận được.
    Nhiều hàng có thể cùng ô (đề bài cho phép).
    """
    candidates = reachable_shelf_cells(grid)
 
    if not candidates:
        print("Cảnh báo: Không có ô kệ hợp lệ để đặt hàng!")
        return []
    return [Good(*random.choice(candidates)) for _ in range(num_goods)]

def draw_grid(grid, robots, goods):
    """
    Display the grid: EMPTY cells = white, SHELF cells = gray.
    """
    rows = len(grid)
    cols = len(grid[0])

    img = np.ones((rows, cols, 3)) # Create an RGB image starting all EMPTY

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == SHELF:
                img[r, c] = [0.4, 0.4, 0.4]   # Grey = shelf

    corners = [(0,0), (0,cols-1), (rows-1,0), (rows-1,cols-1)] # 4 corners for robot home
    for r, c in corners:
        img[r, c] = [0.7, 0.9, 0.7]   # Light green for corners (robot home)

    # Ô kệ có hàng → vàng; ô kệ có ≥2 hàng → cam đậm hơn
    from collections import Counter
    good_count = Counter((g.row, g.col) for g in goods)
    for (r, c), count in good_count.items():
        if count == 1:
            img[r, c] = [1.0, 0.85, 0.0]    # Vàng = 1 kiện
        else:
            img[r, c] = [1.0, 0.55, 0.0]    # Cam = nhiều kiện
 
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(img, interpolation="nearest", aspect="equal")
    
    # Số lượng kiện trên ô có nhiều hơn 1
    for (r, c), count in good_count.items():
        if count > 1:
            ax.text(c, r, str(count), ha="center", va="center",
                    fontsize=7, color="black", fontweight="bold", zorder=6)
    
    # Draw robots in round circles with their ID
    for robot in robots:
        colour = ROBOT_COLOURS[(robot.robot_id - 1) % len(ROBOT_COLOURS)]
        ax.plot(robot.col, robot.row, "o", color=colour, markersize=14, zorder=5)
        ax.text(robot.col, robot.row, str(robot.robot_id), color="white",
                fontsize=8, ha="center", va="center", zorder=6)
    
    # Set grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(1, cols + 1))
    ax.set_yticklabels(np.arange(1, rows + 1))
 
    # Add legend
    legend = [
        mpatches.Patch(color=[1,1,1], label="Way (EMPTY)"),
        mpatches.Patch(color=[0.4,0.4,0.4], label="Shelf (SHELF)"),
        mpatches.Patch(color=[0.7,0.9,0.7], label="Corner - robot home"),
        mpatches.Patch(color=[1.0,0.85,0.0],  label="Kệ có 1 kiện"),
        mpatches.Patch(color=[1.0,0.55,0.0],  label="Kệ có nhiều kiện"),
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#e74c3c", markersize=10, label="Robot")
    ]
    ax.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

def demo_availability(goods, robots):
    """
    Mô phỏng trả lời Prompt 3:
    'How will the system update when one robot takes an item
     before another robot arrives?'
    """
    print("=" * 50)
    print("DEMO: Cơ chế đặt chỗ (available)")
    print("=" * 50)

    # Robot 1 tìm hàng gần nhất
    r1 = robots[0]
    target1 = find_nearest_good(goods, r1.row, r1.col)
    if target1:
        print(f"\nRobot 1 @ ({r1.row},{r1.col}) chọn: {target1}")
        target1.available = False   # Đặt chỗ ngay lập tức
        print(f"  → Sau khi đặt chỗ: available = {target1.available}")

    # Robot 2 tìm hàng — cùng ô kệ đó vẫn có thể có kiện khác
    r2 = robots[1]
    target2 = find_nearest_good(goods, r2.row, r2.col)
    if target2:
        print(f"\nRobot 2 @ ({r2.row},{r2.col}) chọn: {target2}")
        if target2 is target1:
            print("  ❌ Lỗi: cùng kiện!")
        else:
            print("  ✓ Kiện khác, không xung đột")
        target2.available = False
 
    # Kiểm tra trạng thái kệ (3 kiện cùng ô)
    print("\n--- Trạng thái tất cả goods ---")
    for g in goods[:8]:
        print(f"  {g}")
    print("=" * 50)
    
if __name__ == "__main__":
    random.seed(42)
    
    ROWS = 12
    COLS = 14
    NUM_ROBOTS = 4
    NUM_GOODS  = 15
 
    grid = make_grid(ROWS, COLS)
    grid = add_shelves(grid)
    robots = make_robots(grid, NUM_ROBOTS)
    goods  = make_goods(grid, NUM_GOODS)
    # Thống kê ô bị trùng
    from collections import Counter
    counts = Counter((g.row, g.col) for g in goods)
    multi  = {pos: n for pos, n in counts.items() if n > 1}
 
    print(f"Tổng kiện hàng  : {len(goods)}")
    print(f"Ô kệ có hàng    : {len(counts)}")
    print(f"Ô có nhiều kiện : {len(multi)}")
    for pos, n in multi.items():
        print(f"  Kệ {pos} → {n} kiện")
 
    print("\n=== Bản đồ (G=1 kiện, 2G=2 kiện...) ===")
 
    demo_availability(goods, robots)
 
    draw_grid(grid, robots, goods)