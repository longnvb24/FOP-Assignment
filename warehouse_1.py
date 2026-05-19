# import nessary libraries
import numpy as np
import random
import matplotlib.pyplot as plt

# Create constants
EMPTY = 0
SHELF = 1

IDLE = 'idle'
MOVING = 'moving'
BACK = 'back'

ROBOT_COLORS = ['#e74c3c', '#3498db', '#f1c40f', '#9b59b6', 
                '#1abc9c', '#e67e22', '#2ecc71', '#34495e']


DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

class Good:
    number = 0
    def __init__ (self, row, col):
        self.row = row
        self.col = col
        Good.number += 1
        self.id = f"G{Good.number}"
        self.available = True

class Robot:
    id = 0
    def __init__ (self, row, col, grid):
        self.row = row
        self.col = col
        self.home_row = row
        self.home_col = col
        self.grid = grid
        Robot.id += 1
        self.id = Robot.id
        self.state = IDLE
        self.carrying = False
        self.target = None
        self.path = []

    def move(self, goods):
        '''
        move - moves the robot along its path
        goods: list of Good objects in the warehouse
        '''
        if self.state == IDLE:
            self.looking_for_good(goods)
        elif self.state == MOVING:
            if self.path:
                self.row, self.col = self.path.pop(0)
            self.check_destination(goods)
        elif self.state == BACK:
            if self.path:
                self.row, self.col = self.path.pop(0)
            if (self.row, self.col) == (self.home_row, self.home_col):
                self.state = IDLE
                self.carrying = None
    
    def looking_for_good(self, goods):
        '''
        looking_for_good - finds the nearest good and sets it as target

        goods: list of Good objects in the warehouse
        '''
        nearest_good = find_nearest_good(self, goods)
        if nearest_good is None:
            return
        
        pickup_cells = find_pickup_cells(self.grid, [nearest_good])

        if not pickup_cells:
            return

        if (self.row, self.col) in pickup_cells:
            path = []
        else:
            path = find_path(self.grid, self.row, self.col, pickup_cells[0])

        if not path and (self.row, self.col) not in pickup_cells:
            return
        
        nearest_good.available = False
        self.target = nearest_good
        self.state = MOVING
        self.path = path
    
    def check_destination(self, goods):
        '''
        check_destination - checks if the robot has reached its target

        goods: list of Good objects in the warehouse
        '''
        if not self.target:
            self.state = IDLE
            return
        
        if self.target not in goods:
            self.target = None
            self.state = IDLE
            self.path = []
            return
        
        pickup_cells = set(find_pickup_cells(self.grid, [self.target]))
        if (self.row, self.col) in pickup_cells and not self.path:
            goods.remove(self.target)
            self.carrying = True
            self.state = BACK
            self.target = None
            self.path = find_path(self.grid, self.row, self.col, (self.home_row, self.home_col))

def make_grid(rows, cols):
    '''
    make_grid - creates a 2D grid

    rows: number of rows (int)
    cols: number of columns (int)
    return a 2D list
    '''
    try:
        if rows <= 0 or cols <= 0:
            raise ValueError("Rows and columns must be positive integers.")
    except ValueError as e:
        print(f"Invalid grid size: {e}. Using default 12x14.")
        rows, cols = 12, 14
    return [[EMPTY for i in range(cols)] for i in range(rows)]

def add_shelves(grid):
    '''
    add_shelves - adds shelves to the grid
    
    grid: 2D list representing the warehouse grid
    return a 2D list with shelves
    '''
    rows = len(grid)
    cols = len(grid[0])
    for r in range(1, rows-1):
        for c in range(1, cols-1, 2):
            grid[r][c] = SHELF

def check_reachability(grid):
    '''
    check_reachability - check if all goods are reachable

    grid: 2D list representing the warehouse grid
    return list of reachable cells (row, col)
    '''
    rows = len(grid)
    cols = len(grid[0])

    reachable_cells = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == SHELF:
                reachable = False
                for dr, dc in DIRECTIONS:
                    if not reachable:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == EMPTY:
                            reachable = True
                            reachable_cells.append((r, c))
    return reachable_cells

def find_nearest_good(robot, goods):
    '''
    find_nearest_good - finds the nearest available good"
    
    robot: Robot object
    goods: list of Good objects

    return the nearest Good object or None if none are available
    '''

    nearest = None
    min_distance = float('inf')

    for good in goods:
        if good.available:
            distance = abs(robot.row - good.row) + abs(robot.col - good.col)
            if distance < min_distance:
                min_distance = distance
                nearest = good

    return nearest

def find_path(grid, sr, sc, goal):
    '''
    find_path - finds a path from start to goal

    grid: 2D list representing the warehouse grid
    sr: starting row
    sc: starting column
    goal: (row, col) tuple for goal position
    return a list of (row, col) tuples representing the path
    '''
    rows = len(grid)
    cols = len(grid[0])
    queue = [(sr,sc, [])]

    if (sr,sc) == goal:
        return []

    visited_cells = set()
    visited_cells.add((sr,sc))

    while queue:
        r, c, path = queue.pop(0)
        for i in DIRECTIONS:
            nr,nc = r + i[0], c + i[1]
            direc = (nr, nc)

            if 0 <= nr < rows and 0 <= nc < cols \
                and grid[nr][nc] == EMPTY \
                and direc not in visited_cells:
                new_path = path + [direc]

                if direc == goal:
                    return new_path

                visited_cells.add(direc)
                queue.append((nr, nc, new_path))
    return []

def find_pickup_cells(grid, goods):
    '''
    find_pickup_cells - returns a list of cells that allow picking up goods

    grid: 2D list representing the warehouse grid
    goods: list of Good objects
    '''
    pickup_cells = set()
    for good in goods:
        r, c = good.row, good.col
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == EMPTY:
                pickup_cells.add((nr, nc))
    return list(pickup_cells)


