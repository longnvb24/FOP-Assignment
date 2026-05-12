"""
BƯỚC 2 – Thêm kệ hàng (Shelves)
=================================
Mục tiêu:
  - Đặt kệ hàng vào lưới theo pattern siêu thị
  - Cập nhật hàm draw_grid để hiển thị kệ (màu xám)
  - Giữ 4 góc + viền luôn trống để robot có thể di chuyển

Pattern kệ (lặp lại):
  Cột: 2 ô kệ, 1 ô lối đi  → col_phase = (c-1) % 3  → 0,1 = kệ; 2 = lối
  Hàng: 3 ô kệ, 2 ô lối đi  → row_phase = (r-1) % 5  → 0,1,2 = kệ; 3,4 = lối

Chạy:  python3 step2_shelves.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Hằng số ─────────────────────────────────────────────────────────────────
EMPTY = 0
SHELF = 1

# ── Tạo lưới (giữ nguyên từ bước 1) ────────────────────────────────────────
def make_grid(rows, cols):
    return [[EMPTY] * cols for _ in range(rows)]

# ── MỚI: Thêm kệ hàng vào lưới ──────────────────────────────────────────────
def add_shelves(grid):
    """
    Đặt kệ hàng theo pattern siêu thị vào lưới (sửa trực tiếp grid).
    Giữ nguyên: viền ngoài, 4 góc → luôn EMPTY.
    """
    rows = len(grid)
    cols = len(grid[0])

    # Tập hợp 4 góc để bảo vệ (robots sẽ spawn ở đây)
    corners = {(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)}

    for r in range(rows):
        for c in range(cols):
            # Bỏ qua viền ngoài và 4 góc
            if (r, c) in corners:
                continue
            if r == 0 or r == rows - 1:
                continue
            if c == 0 or c == cols - 1:
                continue

            # Pattern: lối, kệ, lối, kệ, lối  (xen kẽ theo cột)
            # c=1 → lối, c=2 → kệ, c=3 → lối, c=4 → kệ, ...
            col_is_shelf = (c % 2 == 0)   # cột chẵn = kệ, cột lẻ = lối

            if col_is_shelf:
                grid[r][c] = SHELF

# ── Cập nhật draw_grid: thêm màu sắc ────────────────────────────────────────
def draw_grid(grid, title="Bước 2 – Kệ dọc: lối, kệ, lối, kệ, lối"):
    rows = len(grid)
    cols = len(grid[0])

    # Xây dựng ảnh RGB: mỗi ô là 1 màu
    img = np.ones((rows, cols, 3))   # Mặc định: trắng (EMPTY)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == SHELF:
                img[r, c] = [0.4, 0.4, 0.4]   # Xám đậm = kệ

    # Tô màu xanh nhạt cho 4 góc (vị trí home của robot – bước sau)
    corners = [(0,0), (0,cols-1), (rows-1,0), (rows-1,cols-1)]
    for r, c in corners:
        img[r, c] = [0.7, 0.9, 0.7]   # Xanh nhạt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, interpolation="nearest", aspect="equal")

    # Kẻ lưới
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    # Chú thích
    legend = [
        mpatches.Patch(color=[1,1,1],       label="Lối đi (EMPTY)"),
        mpatches.Patch(color=[0.4,0.4,0.4], label="Kệ hàng (SHELF)"),
        mpatches.Patch(color=[0.7,0.9,0.7], label="Góc – home robot"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ── In ASCII để kiểm tra nhanh ───────────────────────────────────────────────
def print_grid(grid):
    corners = {(0,0),(0,len(grid[0])-1),(len(grid)-1,0),(len(grid)-1,len(grid[0])-1)}
    for r, row in enumerate(grid):
        line = ""
        for c, val in enumerate(row):
            if (r, c) in corners:
                line += " H"
            elif val == SHELF:
                line += " #"
            else:
                line += " ."
        print(line)
    print()

# ── Chạy thử ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ROWS, COLS = 12, 16

    grid = make_grid(ROWS, COLS)
    add_shelves(grid)

    # Đếm số ô kệ
    shelf_count = sum(grid[r][c] == SHELF for r in range(ROWS) for c in range(COLS))
    print(f"Lưới {ROWS}×{COLS}  →  {shelf_count} ô kệ, "
          f"{ROWS*COLS - shelf_count} ô lối đi")
    print()
    print_grid(grid)

    draw_grid(grid)