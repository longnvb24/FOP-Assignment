"""
warehouse.py  -  Robotic Warehouse Simulation
==============================================
Simulates autonomous robots collecting goods from shelf locations
and returning them to their home corners in a grid-based environment.

Usage
-----
Interactive : python3 warehouse.py -i
Batch       : python3 warehouse.py -f map1.csv -p params1.csv
Save map    : python3 warehouse.py -i --save-map my_map.csv
"""

import argparse
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Grid cell values
EMPTY = 0   # Walkable aisle cell
SHELF = 1   # Impassable shelf cell

# Robot state labels
STATE_IDLE      = "idle"             # No current task
STATE_MOVING    = "moving_to_good"   # Travelling toward a target good
STATE_RETURNING = "returning"        # Carrying a good back to home corner

# Colours for up to 8 robots in the visualisation
ROBOT_COLOURS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
                 "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

# Four possible movement directions (up, down, left, right)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Good:
    '''
    Good: represents a single collectable item stored on a shelf cell.
    Attributes
        good_id   - unique identifier (int)
        row       - row index of the shelf cell (int)
        col       - column index of the shelf cell (int)
        available - True if unclaimed; False once reserved by a robot (bool)
    '''

    counter = 0

    def __init__(self, row, col):
        '''
        __init__ - creates a new Good at the given shelf cell.
            row - row index of the shelf location (int)
            col - column index of the shelf location (int)
        '''
        Good.counter  += 1
        self.good_id   = Good.counter # auto-incrementing unique ID
        self.row       = row
        self.col       = col
        self.available = True

    def __repr__(self):
        status = "available" if self.available else "reserved"
        return f"Good({self.good_id}) @ ({self.row},{self.col}) [{status}]"


class Robot:
    '''
    Robot: an autonomous warehouse robot that collects goods and returns home.

    Attributes
        robot_id           - unique identifier (int)
        row, col           - current grid position (int)
        home_row, home_col - row and column of the home corner; never changes (int)
        grid               - reference to the warehouse grid (list of lists)
        state              - current state string (str)
        target_good        - Good object this robot is heading for, or None
        path               - remaining waypoints [(row,col), ...] (list)
        carrying           - True while the robot holds a good (bool)
        goods_delivered    - count of goods successfully returned home (int)
        steps_taken        - total movement steps taken (int)
        idle_steps         - timesteps spent in idle state (int)
    '''

    def __init__(self, robot_id, row, col, grid):
        '''
        __init__ - initialises a robot at a corner spawn position.

        robot_id - unique robot number (int)
        row      - starting row / home row (int)
        col      - starting column / home column (int)
        grid     - warehouse grid used for pathfinding (list of lists)
        '''
        self.robot_id        = robot_id
        self.row             = row
        self.col             = col
        self.home_row        = row
        self.home_col        = col
        self.grid            = grid
        self.state           = STATE_IDLE
        self.target_good     = None
        self.path            = []
        self.carrying        = False
        self.goods_delivered = 0
        self.steps_taken     = 0
        self.idle_steps      = 0

    def step_change(self, goods):
        '''
        step_change - advances the robot by one timestep.
        Called once per timestep by the simulation loop.

        goods - list of all Good objects currently in the warehouse (list)
        '''
        if self.state == STATE_IDLE: # if idle, try to find a new target
            self.assign_target(goods)

        elif self.state == STATE_MOVING: # if moving, take a step along the path
            if self.path: # if path is not empty, pop the next waypoint and move there
                self.row, self.col = self.path.pop(0)
                self.steps_taken  += 1
            self.check_pickup(goods) # check if robot reached the target good

        elif self.state == STATE_RETURNING: # if returning, take a step toward home
            if self.path: # if path is not empty, pop the next waypoint and move there
                self.row, self.col = self.path.pop(0)
                self.steps_taken  += 1
            if (self.row, self.col) == (self.home_row, self.home_col): # if reached home, drop the good and reset state
                self.carrying         = False
                self.goods_delivered += 1
                self.state            = STATE_IDLE

    def assign_target(self, goods):
        '''
        assign_target - determines the nearest available good and path

        goods - list of all Good objects (list)
        '''
        target = find_nearest_good(goods, self.row, self.col) # find the closest available good
        if target is None:
            self.idle_steps += 1
            return

        pickup = get_pickup_cells(self.grid, target) # get the valid pickup cells for this target
        path   = bfs(self.grid, self.row, self.col, pickup) # find the shortest path to this pickup cell

        if not path and (self.row, self.col) not in set(pickup): # if no path and pickup cell exists, remain idle
            self.idle_steps += 1
            return

        target.available = False   # reserve immediately; prevents other robots claiming it
        self.target_good = target
        self.path        = path
        self.state       = STATE_MOVING # transition to moving state

    def check_pickup(self, goods):
        '''
        check_pickup - checks whether the robot has reached a pickup cell.

        goods - list of all Good objects (list)
        '''
        if not self.target_good: # if no target, reset to idle
            self.state = STATE_IDLE
            return

        if self.target_good not in goods: # retarget if the good was removed before this robot arrived
            self.target_good = None
            self.path        = []
            self.state       = STATE_IDLE
            return

        pickup = set(get_pickup_cells(self.grid, self.target_good)) # get the valid pickup cells for this target
        if (self.row, self.col) in pickup and not self.path: # if currently on a pickup cell and path is empty, pick up the good and plan return path
            goods.remove(self.target_good) # remove the good from the warehouse
            self.carrying    = True # mark the robot as now carrying a good
            self.target_good = None # clear the target good reference
            self.state       = STATE_RETURNING # transition to returning state
            self.path        = bfs(self.grid, self.row, self.col, 
                                   [(self.home_row, self.home_col)]) # plan path back to home

def get_pickup_cells(grid, good):
    '''
    get_pickup_cells - returns all EMPTY cells near shelf cell, where the robot can pick up the good.

    grid - warehouse grid (list of lists)
    good - the Good object
    '''
    rows = len(grid)
    cols = len(grid[0])
    return [
        (good.row + dr, good.col + dc)
        for dr, dc in DIRECTIONS
        if (0 <= good.row + dr < rows and 0 <= good.col + dc < cols
            and grid[good.row + dr][good.col + dc] == EMPTY)
    ]

def bfs(grid, sr, sc, goals):
    '''
    bfs - find the shortest path from (sr, sc) to any of the goal cells using Breadth-First Search.
    Returns a list of steps [(row, col), ...], or [] if no path exists.
    '''
    rows  = len(grid)
    cols  = len(grid[0])
    goals = set(goals)

    if (sr, sc) in goals: # if already on a goal cell, return empty path
        return []

    queue = [(sr, sc, [])] # queue of (current_row, current_col, path)

    visited = set()
    visited.add((sr, sc)) # mark the starting cell as visited

    while len(queue) > 0:
        current_row, current_col, path = queue.pop(0)

        for dr, dc in DIRECTIONS: # check all four directions
            next_row = current_row + dr
            next_col = current_col + dc

            if (0 <= next_row < rows) and (0 <= next_col < cols) \
                and (next_row, next_col) not in visited \
                and grid[next_row][next_col] != SHELF:
                new_path = path + [(next_row, next_col)] # concatenate new cells to the path

                if (next_row, next_col) in goals: # if reached a goal cell, return the path to it
                    return new_path

                visited.add((next_row, next_col)) # mark cell as visited
                queue.append((next_row, next_col, new_path)) # add cell to the queue

    return []


def find_nearest_good(goods, robot_row, robot_col):
    '''
    find_nearest_good - returns the nearest available Good by Manhattan distance.

    goods      - list of all Good objects (list)
    robot_row  - current row of the robot (int)
    robot_col  - current column of the robot (int)
    '''
    nearest      = None
    nearest_dist = float("inf")

    for good in goods:
        if good.available: # just find the available goods
            dist = abs(good.row - robot_row) + abs(good.col - robot_col)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest      = good
    return nearest


def make_grid(rows, cols):
    '''
    make_grid - creates an empty warehouse

    rows - number of rows (int)
    cols - number of columns (int)
    '''
    return [[EMPTY] * cols for _ in range(rows)]


def add_shelves(grid):
    '''
    add_shelves - populates the grid with a structured shelf layout as supermarkets have.

    grid - warehouse grid to modify in-place (list of lists)
    '''
    rows    = len(grid)
    cols    = len(grid[0])

    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if c % 2 == 1:
                grid[r][c] = SHELF

def load_map_csv(filepath):
    '''
    load_map_csv - reads warehouse terrain from a CSV file.

    filepath - path to the CSV file (str)
    '''
    try:
        with open(filepath, 'r') as f:
            grid = []
            for line in f:
                line = line.strip()
                line = line.split(",")
                grid.append([int(v) for v in line])
        return grid
    except ValueError as err:
        print(f"Error: invalid value in map file '{filepath}': {err}")

def save_map_csv(grid, filepath):
    '''
    save_map_csv - writes the current terrain grid to a CSV file.

    grid     - warehouse grid (list of lists)
    filepath - destination file path (str)
    '''
    with open(filepath, "w") as f:
        for row in grid:
            line = ",".join(str(cell) for cell in row)
            f.write(line + "\n")
    print(f"Map saved to: {filepath}")


def reachable_shelf_cells(grid):
    '''
    reachable_shelf_cells - finds all shelf cells that a robot can reach.

    grid - warehouse grid (list of lists)
    '''
    rows   = len(grid)
    cols   = len(grid[0])
    result = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == SHELF: # only check shelf cells
                found_empty = False
                for dr, dc in DIRECTIONS:
                    if not found_empty:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == EMPTY:
                            result.append((r, c))
                            found_empty = True # found at least 1 reachable cell
    return result


def make_robots(grid, num_robots):
    '''
    make_robots - spawns robots at the four grid corners

    grid       - warehouse grid (list of lists)
    num_robots - number of robots to create (int)
    '''
    rows    = len(grid)
    cols    = len(grid[0])
    corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
    robots = []
    for i in range(num_robots):
        r, c = corners[i % 4]
        robots.append(Robot(i + 1, r, c, grid))
    return robots


def make_goods(grid, num_goods):
    '''
    make_goods - places goods on reachable shelf cells at random.

    grid      - warehouse grid (list of lists)
    num_goods - number of Good objects to create (int)
    '''
    candidates = reachable_shelf_cells(grid)
    if not candidates:
        print("Warning: no reachable shelf cells found for good placement.")
        return []
    
    goods_list = []
    for i in range(num_goods):
        chosen_cell = random.choice(candidates)
        r = chosen_cell[0]
        c = chosen_cell[1]
        goods_list.append(Good(r, c))
    return goods_list

def run_simulation(grid, robots, goods,
                   max_steps=120, step_delay=0.25, spawn_prob=0.0):
    '''
    run_simulation - main simulation loop with live visualisation.

    grid       - warehouse grid (list of lists)
    robots     - list of Robot objects (list)
    goods      - list of Good objects; modified in-place (list)
    max_steps  - maximum number of timesteps to run (int)
    step_delay - pause duration between frames in seconds (float)
    spawn_prob - probability of a new good appearing each step (float)
    '''
    rows = len(grid)
    cols = len(grid[0])

    # Per-timestep statistics
    history_delivered  = []   # cumulative goods delivered
    history_remaining  = []   # goods still in the warehouse
    history_idle       = []   # number of idle robots
    history_throughput = []   # goods delivered per 5-step window

    # Heatmap: counts how many times each cell is visited by any robot
    heatmap    = np.zeros((rows, cols), dtype=float)
    candidates = reachable_shelf_cells(grid)
    prev_total = 0

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Robotic Warehouse Simulation", fontsize=14, fontweight="bold")
    plt.tight_layout(pad=2.5)

    for step in range(1, max_steps + 1):

        # 1. Update all robots
        for robot in robots:
            robot.step_change(goods)
            heatmap[robot.row, robot.col] += 1

        # 2. Randomly spawn a new good (if enabled)
        if spawn_prob > 0 and random.random() < spawn_prob and candidates:
            goods.append(Good(*random.choice(candidates)))

        # 3. Record statistics
        total_delivered = sum(r.goods_delivered for r in robots)
        num_idle        = sum(1 for r in robots if r.state == STATE_IDLE)
        history_delivered.append(total_delivered)
        history_remaining.append(len(goods))
        history_idle.append(num_idle)

        if step % 5 == 0:
            history_throughput.append(total_delivered - prev_total)
            prev_total = total_delivered

        # 4. Draw frame
        draw_frame(fig, axes, grid, robots, goods,
                    history_delivered, history_remaining,
                    history_idle, history_throughput, heatmap, step)
        plt.pause(step_delay)

        # 5. Early exit: all goods collected and no more will spawn
        if not goods and spawn_prob == 0 and num_idle == len(robots):
            print(f"\nAll goods collected after {step} steps.")
            break

    plt.ioff()
    draw_frame(fig, axes, grid, robots, goods,
                history_delivered, history_remaining,
                history_idle, history_throughput, heatmap, step)
    fig.suptitle(
        f"Complete  -  {sum(r.goods_delivered for r in robots)} goods "
        f"delivered in {step} steps",
        fontsize=13, fontweight="bold"
    )
    print_summary(robots, step)
    plt.show()


def draw_frame(fig, axes, grid, robots, goods,
                hist_del, hist_rem, hist_idle,
                hist_throughput, heatmap, step):
    '''
    draw_frame - redraws all four subplots for the current timestep.

    fig              - matplotlib Figure object
    axes             - 2x2 array of Axes objects
    grid             - warehouse grid (list of lists)
    robots           - list of Robot objects (list)
    goods            - list of Good objects (list)
    hist_del         - cumulative deliveries per step (list)
    hist_rem         - goods remaining per step (list)
    hist_idle        - idle robot count per step (list)
    hist_throughput  - deliveries per 5-step window (list)
    heatmap          - visit-count array, shape (rows, cols) (np.ndarray)
    step             - current timestep number (int)
    '''
    ax_map  = axes[0, 0]
    ax_del  = axes[0, 1]
    ax_tp   = axes[1, 0]
    ax_stat = axes[1, 1]
    rows = len(grid)
    cols = len(grid[0])

    # ── Top-left: warehouse map ───────────────────────────────────────────────
    ax_map.cla()
    img = np.ones((rows, cols, 3))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == SHELF:
                img[r, c] = [0.40, 0.40, 0.40]   # dark grey = shelf

    for r, c in [(0,0),(0,cols-1),(rows-1,0),(rows-1,cols-1)]:
        img[r, c] = [0.72, 0.93, 0.72]            # light green = home corner

    for good in goods:
        img[good.row, good.col] = [1.0, 0.84, 0.0]   # gold = 1 good

    for pos, n in Counter((g.row, g.col) for g in goods).items():
        if n > 1:
            img[pos[0], pos[1]] = [1.0, 0.55, 0.0]   # orange = multiple goods

    ax_map.imshow(img, interpolation="nearest", aspect="equal")

    # Draw planned paths (faint lines)
    for robot in robots:
        if robot.path:
            colour = ROBOT_COLOURS[(robot.robot_id - 1) % len(ROBOT_COLOURS)]
            pr = [robot.row] + [p[0] for p in robot.path]
            pc = [robot.col] + [p[1] for p in robot.path]
            ax_map.plot(pc, pr, "-", color=colour, alpha=0.25, linewidth=1.5)

    # Draw robots
    for robot in robots:
        colour = ROBOT_COLOURS[(robot.robot_id - 1) % len(ROBOT_COLOURS)]
        marker = "^" if robot.carrying else "o"   # triangle = carrying good
        ax_map.plot(robot.col, robot.row, marker, color=colour,
                    markersize=13, zorder=5)
        ax_map.text(robot.col, robot.row, str(robot.robot_id),
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=6)

    ax_map.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax_map.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax_map.grid(which="minor", color="lightgray", linewidth=0.5)
    ax_map.tick_params(which="minor", length=0)
    ax_map.set_title(
        f"Step {step}  |  Remaining: {len(goods)}  |  Delivered: {hist_del[-1]}",
        fontsize=10
    )
    legend = [
        mpatches.Patch(color=[0.40,0.40,0.40], label="Shelf"),
        mpatches.Patch(color=[1.0,0.84,0.0],   label="Good (1 item)"),
        mpatches.Patch(color=[1.0,0.55,0.0],   label="Good (multiple)"),
        mpatches.Patch(color=[0.72,0.93,0.72],  label="Home corner"),
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#e74c3c", markersize=9, label="Robot"),
        plt.Line2D([0],[0], marker="^", color="w",
                   markerfacecolor="#e74c3c", markersize=9, label="Carrying"),
    ]
    ax_map.legend(handles=legend, loc="upper right", fontsize=7, framealpha=0.9)

    x = list(range(1, len(hist_del) + 1))

    # ── Top-right: cumulative deliveries ─────────────────────────────────────
    ax_del.cla()
    ax_del.plot(x, hist_del, color="#2ecc71", linewidth=2, label="Delivered")
    ax_del.fill_between(x, hist_del, alpha=0.2, color="#2ecc71")
    if hist_del:
        ideal = [hist_del[-1] / len(hist_del) * i for i in range(1, len(x)+1)]
        ax_del.plot(x, ideal, color="gray", linewidth=1,
                    linestyle=":", label="Ideal rate")
        ax_del.legend(fontsize=8)
    ax_del.set_title("Cumulative Deliveries", fontsize=10)
    ax_del.set_xlabel("Timestep")
    ax_del.set_ylabel("Goods delivered")
    ax_del.set_xlim(left=1)
    ax_del.set_ylim(bottom=0)
    ax_del.grid(True, alpha=0.3)

    # ── Bottom-left: throughput per 5-step window ─────────────────────────────
    ax_tp.cla()
    if hist_throughput:
        tp_x = [i * 5 for i in range(1, len(hist_throughput) + 1)]
        ax_tp.bar(tp_x, hist_throughput, width=4, color="#9b59b6", alpha=0.75)
        avg = sum(hist_throughput) / len(hist_throughput)
        ax_tp.axhline(avg, color="#e74c3c", linewidth=1.5,
                      linestyle="--", label=f"Avg = {avg:.1f}")
        ax_tp.legend(fontsize=8)
    ax_tp.set_title("Throughput (goods per 5 steps)", fontsize=10)
    ax_tp.set_xlabel("Timestep")
    ax_tp.set_ylabel("Goods delivered")
    ax_tp.set_xlim(left=0)
    ax_tp.set_ylim(bottom=0)
    ax_tp.grid(True, alpha=0.3)

    # ── Bottom-right: goods remaining + idle robots ───────────────────────────
    ax_stat.cla()
    ax_stat.plot(x, hist_rem,  color="#e74c3c", linewidth=2,
                 label="Goods remaining")
    ax_stat.plot(x, hist_idle, color="#3498db", linewidth=2,
                 linestyle="--", label="Idle robots")
    ax_stat.set_title("Goods Remaining & Idle Robots", fontsize=10)
    ax_stat.set_xlabel("Timestep")
    ax_stat.set_ylabel("Count")
    ax_stat.set_xlim(left=1)
    ax_stat.set_ylim(bottom=0)
    ax_stat.legend(fontsize=9)
    ax_stat.grid(True, alpha=0.3)

    fig.canvas.draw()
    fig.canvas.flush_events()


# ══════════════════════════════════════════════════════════════════════════════
# Interactive mode  (-i)
# ══════════════════════════════════════════════════════════════════════════════
def prompt_int(msg, lo, hi, default):
    '''
    prompt_int - prompts the user for an integer within [lo, hi].
    Press Enter to accept the default value.

    msg     - prompt message (str)
    lo      - minimum accepted value (int)
    hi      - maximum accepted value (int)
    default - value returned when the user presses Enter (int)
    '''
    while True:
        raw = input(f"  {msg} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"    Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("    Please enter a whole number.")


def prompt_float(msg, lo, hi, default):
    '''
    prompt_float - prompts the user for a float within [lo, hi].
    Press Enter to accept the default value.

    msg     - prompt message (str)
    lo      - minimum accepted value (float)
    hi      - maximum accepted value (float)
    default - value returned when the user presses Enter (float)
    '''
    while True:
        raw = input(f"  {msg} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
            print(f"    Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("    Please enter a decimal number.")


def interactive_setup():
    '''
    interactive_setup - prompts the user for all simulation parameters.
    Returns (grid, robots, goods, params).
    Run with: python3 warehouse.py -i
    '''
    print("\n" + "=" * 52)
    print("  WAREHOUSE SIMULATION  –  Interactive Setup")
    print("=" * 52)

    rows       = prompt_int("Grid rows (5-50)",              5,   50,  12)
    cols       = prompt_int("Grid columns (5-50)",           5,   50,  14)
    num_robots = prompt_int("Number of robots (1-8)",        1,    8,   4)
    num_goods  = prompt_int("Initial number of goods",       1,  200,  12)
    max_steps  = prompt_int("Simulation length (steps)",     1, 2000, 120)
    spawn_prob = prompt_float("Good spawn probability per step (0-1)",
                              0.0, 1.0, 0.0)
    step_delay = prompt_float("Animation delay per frame (seconds)",
                              0.0, 5.0, 0.25)

    grid   = make_grid(rows, cols)
    add_shelves(grid)
    robots = make_robots(grid, num_robots)
    goods  = make_goods(grid, num_goods)

    params = {
        "max_steps":  max_steps,
        "spawn_prob": spawn_prob,
        "step_delay": step_delay,
    }
    return grid, robots, goods, params


# ══════════════════════════════════════════════════════════════════════════════
# Batch mode  (-f map.csv -p params.csv)
# ══════════════════════════════════════════════════════════════════════════════
def load_params_csv(filepath):
    '''
    load_params_csv - reads simulation parameters from a key-value CSV file.
    Expected format (one parameter per row):
        num_robots,4
        num_goods,12
        max_steps,120
        spawn_probability,0.05
        step_delay,0.25
    Values are automatically cast to int or float where possible.

    filepath - path to the parameters CSV file (str)

    Returns a dict mapping parameter names to values.
    '''
    params = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                key = row[0].strip()
                val = row[1].strip()
                try:
                    params[key] = int(val)
                except ValueError:
                    try:
                        params[key] = float(val)
                    except ValueError:
                        params[key] = val
    return params


def batch_setup(map_file, params_file):
    '''
    batch_setup - loads terrain and parameters from CSV files.
    Falls back to a default 12x14 grid and default parameters if either
    file is missing or contains invalid data.
    Run with: python3 warehouse.py -f map1.csv -p params1.csv

    map_file    - path to the warehouse terrain CSV (str)
    params_file - path to the simulation parameters CSV (str)

    Returns (grid, robots, goods, params).
    '''
    try:
        grid = load_map_csv(map_file)
        if not grid or not grid[0]:
            raise ValueError("Map file is empty.")
    except FileNotFoundError:
        print(f"Warning: '{map_file}' not found – using default 12x14 grid.")
        grid = make_grid(12, 14)
        add_shelves(grid)
    except Exception as e:
        print(f"Warning: could not read map ({e}) – using default grid.")
        grid = make_grid(12, 14)
        add_shelves(grid)

    try:
        params = load_params_csv(params_file)
    except FileNotFoundError:
        print(f"Warning: '{params_file}' not found – using default parameters.")
        params = {}
    except Exception as e:
        print(f"Warning: could not read parameters ({e}) – using defaults.")
        params = {}

    num_robots = int(params.get("num_robots",   4))
    num_goods  = int(params.get("num_goods",   12))

    robots = make_robots(grid, num_robots)
    goods  = make_goods(grid, num_goods)

    sim_params = {
        "max_steps":  int(params.get("max_steps",          120)),
        "spawn_prob": float(params.get("spawn_probability", 0.0)),
        "step_delay": float(params.get("step_delay",        0.25)),
    }
    return grid, robots, goods, sim_params


# ── Summary statistics ────────────────────────────────────────────────────────
def print_summary(robots, total_steps):
    '''
    print_summary - prints a formatted results table to the console.

    robots      - list of Robot objects (list)
    total_steps - number of timesteps the simulation ran (int)
    '''
    total = sum(r.goods_delivered for r in robots)
    total_steps_all = sum(r.steps_taken for r in robots)

    print("\n" + "=" * 54)
    print("  SIMULATION RESULTS")
    print("=" * 54)
    print(f"  Total steps          : {total_steps}")
    print(f"  Total goods delivered: {total}")
    print(f"  Avg throughput       : {total / total_steps * 10:.2f} goods / 10 steps")
    print(f"  Total robot steps    : {total_steps_all}")
    print(f"  Avg steps per item   : {total_steps_all / max(total, 1):.1f}")
    print()
    print(f"  {'Robot':<8} {'Delivered':>9} {'Steps':>6} "
          f"{'Idle':>6} {'Efficiency':>11}")
    print(f"  {'-' * 44}")
    for r in robots:
        eff = r.goods_delivered / max(r.steps_taken, 1) * 100
        print(f"  Robot {r.robot_id:<3} {r.goods_delivered:>9} "
              f"{r.steps_taken:>6} {r.idle_steps:>6} {eff:>10.1f}%")
    print("=" * 54)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def build_parser():
    '''
    build_parser - constructs the command-line argument parser.
    Returns an argparse.ArgumentParser.
    '''
    parser = argparse.ArgumentParser(
        description="Robotic Warehouse Simulation – COMP1005/5005",
        epilog=(
            "Examples:\n"
            "  python3 warehouse.py -i\n"
            "  python3 warehouse.py -i --save-map my_map.csv\n"
            "  python3 warehouse.py -f map1.csv -p params1.csv\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-i", "--interactive", action="store_true",
                      help="Interactive mode: prompts for all parameters.")
    mode.add_argument("-f", "--map-file", metavar="MAP.csv",
                      help="Batch mode: warehouse terrain CSV file.")
    parser.add_argument("-p", "--params-file", metavar="PARAMS.csv",
                        help="Batch mode: simulation parameters CSV (required with -f).")
    parser.add_argument("--save-map", metavar="OUTPUT.csv",
                        help="Save the generated/loaded map to a CSV file.")
    return parser


def main():
    '''
    main - parses command-line arguments, sets up the simulation, and runs it.
    '''
    parser = build_parser()
    args   = parser.parse_args()

    if args.interactive:
        grid, robots, goods, params = interactive_setup()
    else:
        if not args.params_file:
            parser.error("-p / --params-file is required when using -f / --map-file.")
        grid, robots, goods, params = batch_setup(args.map_file, args.params_file)

    if args.save_map:
        save_map_csv(grid, args.save_map)

    print(f"\nStarting: {len(robots)} robot(s), {len(goods)} goods, "
          f"grid {len(grid)}x{len(grid[0])}")

    run_simulation(
        grid,
        robots,
        goods,
        max_steps  = params["max_steps"],
        step_delay = params["step_delay"],
        spawn_prob = params["spawn_prob"],
    )


if __name__ == "__main__":
    main()