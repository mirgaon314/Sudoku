import numpy as np
import pandas as pd
from utils.visualization import visualize_multiple
import random
from enum import Enum
from itertools import combinations
from itertools import permutations
from itertools import product
from collections import Counter
import json
import csv
import os
from datetime import datetime
import copy
from z3 import *

class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

def flatten_grid(grid):
    """Convert a 9x9 grid to a string."""
    return "".join(str(cell) for row in grid for cell in row)

def unflatten_grid(flat_str):
    """Convert a string back to a 9x9 grid."""
    return [list(map(int, flat_str[i*9:(i+1)*9])) for i in range(9)]

def serialize_sets(unavoidable_sets):
    """Convert unavoidable sets to a JSON string."""
    return json.dumps([[[r, c] for (r, c) in s] for s in unavoidable_sets])

def deserialize_sets(json_str):
    """Convert JSON string back to unavoidable sets."""
    # Assume json_str is like: '[[[row, col], [row, col]], [[row, col], ...]]'
    sets_list = json.loads(json_str)
    return [set(tuple(cell) for cell in s) for s in sets_list]

'''
def read_data(filename="../Sudoku/data/sudoku_data.csv"):
    puzzles = []
    solutions = []
    metadata = []
    
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzles.append(unflatten_grid(row["puzzle"]))
            solutions.append(unflatten_grid(row["solution"]))
            metadata.append({
                "difficulty": row["difficulty"],
                "clues": int(row["clues"]),
                "date": row["date_generated"],
                "unavoidable_sets": deserialize_sets(row["unavoidable_sets"])
            })
    
    return puzzles, solutions, metadata
'''

def read_data(filename="../Sudoku/data/sudoku.csv", num = 1):
    quizzes = np.zeros((num, 81), np.int32)
    solutions = np.zeros((num, 81), np.int32)
    count = 0
    for i, line in enumerate(open("../Sudoku/data/sudoku.csv", 'r').read().splitlines()[1:]):
        if(count == num): break
        count += 1
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s
    
    quizzes = quizzes.reshape((-1, 9, 9))
    solutions = solutions.reshape((-1, 9, 9))

    # empty_per_puzzle = np.sum(quizzes == 0, axis=(1, 2))
    # # build histogram
    # hist = Counter(empty_per_puzzle)
    # for empties in sorted(hist):
    #     print(f"{empties}: {hist[empties]}")
    return quizzes, solutions

def check_valid(board, row, col, num):
    # check row and column if the number exist
    for i in range(9):
        if(board[i][col] == num or board[row][i] == num):
            return False
    # check the square that this [row][col] belong
    square_row = 3 * (row // 3)
    square_col = 3 * (col // 3)
    for i in range(square_row, square_row + 3):
        for j in range(square_col, square_col + 3):
            if(board[i][j] == num):
                return False
    return True

def check_valid_board(board):
    """Check if the Sudoku board is valid."""
    for i in range(9):
        # Check rows
        row = [board[i][j] for j in range(9) if board[i][j] != 0]
        if len(row) != len(set(row)):
            return False
        # Check columns
        col = [board[j][i] for j in range(9) if board[j][i] != 0]
        if len(col) != len(set(col)):
            return False
    # Check 3x3 boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = [board[x][y] for x in range(i, i+3) for y in range(j, j+3) if board[x][y] != 0]
            if len(box) != len(set(box)):
                return False
    return True

def valid_numbers(board, row, col):
    # check row and column if the number exist
    exist = []
    to_return = [i for i in range(1,10)]
    for i in range(9):
        if(board[i][col]):
            if board[i][col] not in exist:
                exist.append(board[i][col])
        if(board[row][i]):
            if board[row][i] not in exist:
                exist.append(board[row][i])
    # check the square that this [row][col] belong
    square_row = 3 * (row // 3)
    square_col = 3 * (col // 3)
    for i in range(square_row, square_row + 3):
        for j in range(square_col, square_col + 3):
            if(board[i][j]):
                if board[i][j] not in exist:
                    exist.append(board[i][j])
    for i in range(len(exist)):
        to_return.remove(exist[i])
    return to_return

def generate_3x3_diagonal_box(board):
    for box in range(3):
        for i in range(3):
            for j in range(3):
                num_list = valid_numbers(board, box*3 + i, box*3 + j)
                random.shuffle(num_list)
                for num in num_list:
                    if check_valid(board, box*3 + i, box*3 + j, num):
                        board[box*3 + i][box*3 + j] = num

def backtrack_solve(board, return_board=False):
    """
    Backtracking solver for Sudoku.
    
    If return_board is False (default), the function returns True when a solution is found,
    modifying the board in place.
    
    If return_board is True, the function first makes a deep copy of the board,
    solves it, and returns the solved copy without altering the original board.
    """
    def _solve(b):
        for i in range(9):
            for j in range(9):
                if b[i][j] == 0:
                    num_list = valid_numbers(b, i, j)
                    random.shuffle(num_list)
                    for num in num_list:
                        if check_valid(b, i, j, num):
                            b[i][j] = num
                            _solve(b)
                            b[i][j] = 0
                    return False
        return b if return_board else True

    if return_board:
        # Create a deep copy to preserve the original board.
        board_copy = copy.deepcopy(board)
        result = _solve(board_copy)
        if result is not None:
            return result
        return board_copy
    else:
        return _solve(board)

def has_paired_values(values):
    """Check if the values are paired for size-4, size-6 sets, size-9 sets."""
    freq = Counter(values)
    if len(values) == 4:
        # For size-4 sets, there should be exactly 2 unique values, each appearing twice
        return len(freq) == 2 and all(count == 2 for count in freq.values())
    elif len(values) == 6:
        # For size-6 sets, there should be exactly 3 unique values, each appearing twice
        return (len(freq) == 3 and all(count == 2 for count in freq.values())) or (len(freq) == 2 and all(count == 3 for count in freq.values()))
    elif len(values) == 9:
        # For size-6 sets, there should be exactly 3 unique values, each appearing twice
        return len(freq) == 3 and all(count == 3 for count in freq.values())
    return False

def find_unavoidable_sets(board):

    if isinstance(board, np.ndarray):
        board = board.tolist()

    sets = []
    for band in range(3):
        # 1X2 blue prints / 1 band = 3 rows / 1 stack = 3 column/ 1X2 means 2 box in same row
        for st_1, st_2 in combinations([0, 1, 2], 2):
            for col_1, col_2 in product([0, 1, 2], repeat=2):
                # 2X2 cells  like   [1]    [2]
                #                   [2]    [1]
                for row_1, row_2 in combinations([0, 1, 2], 2):
                    # Define the 4 cells in the 1x2 blueprint
                    cells = [(band * 3 + row, st * 3 + col)
                            for row in (row_1,row_2) 
                            for st, col in zip((st_1, st_2), (col_1, col_2))]
                    values = [board[row][col] for row, col in cells]
                    if not has_paired_values(values):
                        continue
                    # Check all permutations of the values in these cells
                    for perm in permutations(values):
                        new_board = [r[:] for r in board]
                        for (row, col), value in zip(cells, perm):
                            new_board[row][col] = value
                        if check_valid_board(new_board) and new_board != board:
                            # print(f"Found unavoidable set: {cells} with permutation: {perm}")
                            sets.append(cells)
                            break
            for col_1, col_2 in product([0, 1, 2], repeat=2):
                # 3X2 cells  like   [1]    [3]
                #                   [2]    [1]
                #                   [3]    [2]
                cells = [(band * 3 + row, st * 3 + col)
                            for row in range(3) 
                            for st, col in zip((st_1, st_2), (col_1, col_2))]
                values = [board[row][col] for row, col in cells]
                if not has_paired_values(values):
                    continue
                for perm in permutations(values):
                    new_board = [r[:] for r in board]
                    for (row, col), value in zip(cells, perm):
                        new_board[row][col] = value
                    if check_valid_board(new_board) and new_board != board:
                        sets.append(cells)
                        break
        
        # 1X3 blue prints 
        for col_1, col_2, col_3 in product([0, 1, 2], repeat=3):
            # 2X3 cells  like   [1]           [2]      
            #                   [2]    [1]      
            #                          [2]    [1]
            for row_1, row_2, row_3 in permutations(combinations([0, 1, 2], 2), 3):
                # Define the 6 cells in the 1x2 blueprint
                cells = [
                    (band * 3 + r, st * 3 + col)
                    for (rows, st, col) in zip((row_1, row_2, row_3), range(3), (col_1, col_2, col_3))
                    for r in rows
                ]
                values = [board[row][col] for row, col in cells]
                # Debug: Print cells and values
                # print(f"Checking cells: {cells} with values: {values}")
                if not has_paired_values(values):
                    continue
                # Check all permutations of the values in these cells
                for perm in permutations(values):
                    new_board = [r[:] for r in board]
                    for (row, col), value in zip(cells, perm):
                        new_board[row][col] = value
                    if check_valid_board(new_board) and new_board != board:
                        # print(f"Found unavoidable set: {cells} with permutation: {perm}")
                        sets.append(cells)
                        break
        
        for col_1, col_2, col_3 in product([0, 1, 2], repeat=3):
            # 3X2 cells  like   [1]    [3]    [2]
            #                   [2]    [1]    [3]
            #                   [3]    [2]    [1]
            cells = [(band * 3 + row, st * 3 + col)
                        for row in range(3) 
                        for st, col in zip(range(3), (col_1,col_2, col_3))]
            values = [board[row][col] for row, col in cells]
            if not has_paired_values(values):
                continue
            for perm in permutations(values):
                new_board = [r[:] for r in board]
                for (row, col), value in zip(cells, perm):
                    new_board[row][col] = value
                if check_valid_board(new_board) and new_board != board:
                    sets.append(cells)
                    break
        
    # Remove duplicates by converting each list into a frozenset.
    unique_sets = set(frozenset(s) for s in sets)

    # Now, keep only the maximal (largest) sets.
    # That is, if a set A is a proper subset of some set B, then discard A.
    maximal_sets = []
    for s in unique_sets:
        if not any(s < t for t in unique_sets if s != t):
            maximal_sets.append(s)
    
    # Convert them back to sorted lists (if desired)
    maximal_sets = [sorted(list(s)) for s in maximal_sets]
    return maximal_sets

def generate_sudoku_solution():
    board = np.zeros((9,9),dtype = int)
    generate_3x3_diagonal_box(board)
    backtrack_solve(board)
    return board

def greedy_hitting_set_with_heuristics(unavoidable_sets):
    """
    Given a list of unavoidable sets (each as an iterable of cell tuples),
    compute a greedy hitting set that integrates additional heuristics.
    
    For each candidate cell, we add bonus points if its row, column, or block is
    not yet covered by the current hitting set.
    
    Returns:
        A set of cells (tuples) that form the hitting set.
    """
    # Make a working copy of unavoidable sets as sets.
    uncovered = [set(uset) for uset in unavoidable_sets]
    hitting_clues = set()
    covered_rows = set()
    covered_cols = set()
    covered_blocks = set()
    
    while uncovered:
        # Count frequency of each cell in uncovered unavoidable sets.
        freq = {}
        for uset in uncovered:
            for cell in uset:
                freq[cell] = freq.get(cell, 0) + 1
        
        best_cell = None
        best_score = -1
        # Evaluate each candidate cell with extra bonuses.
        for cell, base in freq.items():
            i, j = cell
            bonus = 0
            if i not in covered_rows:
                bonus += 1
            if j not in covered_cols:
                bonus += 1
            block = (i // 3, j // 3)
            if block not in covered_blocks:
                bonus += 1
            score = base + bonus
            if score > best_score:
                best_score = score
                best_cell = cell
        # Add best_cell to the hitting set.
        hitting_clues.add(best_cell)
        i, j = best_cell
        covered_rows.add(i)
        covered_cols.add(j)
        covered_blocks.add((i // 3, j // 3))
        # Remove all unavoidable sets that are hit by best_cell.
        uncovered = [uset for uset in uncovered if best_cell not in uset]
    return hitting_clues

def count_solutions(board, limit=2):
    """
    Count solutions using backtracking but stop as soon as the count reaches `limit`.
    Returns the number of solutions found (up to the limit).
    """
    s = Solver()
    # Create a 9x9 matrix of Int variables.
    X = [[Int(f"x_{i}_{j}") for j in range(9)] for i in range(9)]
    
    # Each cell must be between 1 and 9; add fixed-cell constraints.
    for i in range(9):
        for j in range(9):
            s.add(And(X[i][j] >= 1, X[i][j] <= 9))
            if board[i][j] != 0:
                s.add(X[i][j] == int(board[i][j]))
    
    # Rows must have distinct values.
    for i in range(9):
        s.add(Distinct(X[i]))
    
    # Columns must have distinct values.
    for j in range(9):
        s.add(Distinct([X[i][j] for i in range(9)]))
    
    # Each 3x3 block must have distinct values.
    for block_i in range(3):
        for block_j in range(3):
            block = [X[i][j] for i in range(block_i*3, block_i*3+3) 
                              for j in range(block_j*3, block_j*3+3)]
            s.add(Distinct(block))
    
    count = 0
    while count < limit and s.check() == sat:
        count += 1
        m = s.model()
        # Extract the solution.
        sol = [[m.evaluate(X[i][j]).as_long() for j in range(9)] for i in range(9)]
        # Add a constraint to block this solution.
        s.add(Or([X[i][j] != sol[i][j] for i in range(9) for j in range(9)]))
    return count



def generate_sudoku_puzzle(solution, difficulty):
    target_clues = {
        Difficulty.EASY: 38,
        Difficulty.MEDIUM: 30,
        Difficulty.HARD: 23
    }[difficulty]
    while True:
        # 1. Compute unavoidable sets.
        unavoidable_sets = find_unavoidable_sets(solution)
        # 2. Compute a greedy hitting set with extra heuristics.
        hitting_clues = greedy_hitting_set_with_heuristics(unavoidable_sets)
        
        # 3. Start with an empty board.
        board = np.zeros((9, 9), dtype=int)
        curr_clues = 0
        # Place clues from the hitting set.
        for (i, j) in hitting_clues:
            board[i][j] = solution[i][j]
            curr_clues += 1
        
        # 4. Ensure every row is covered.
        for i in range(9):
            if np.count_nonzero(board[i, :]) == 0:
                j = random.randint(0, 8)
                board[i][j] = solution[i][j]
                curr_clues += 1
        # Ensure every column is covered.
        for j in range(9):
            if np.count_nonzero(board[:, j]) == 0:
                i = random.randint(0, 8)
                board[i][j] = solution[i][j]
                curr_clues += 1
        # Ensure every 3×3 block is covered.
        for block_row in range(3):
            for block_col in range(3):
                block = board[block_row*3:(block_row+1)*3, block_col*3:(block_col+1)*3]
                if np.count_nonzero(block) == 0:
                    i = random.randint(0, 2)
                    j = random.randint(0, 2)
                    board[block_row*3 + i][block_col*3 + j] = solution[block_row*3 + i][block_col*3 + j]
                    curr_clues += 1
        
        # 5. Build a weight matrix for remaining empty cells.
        weights = np.zeros((9, 9))
        for uset in unavoidable_sets:
            for (r, c) in uset:
                weights[r][c] += 1
        # Add extra bonus weight for rows and columns with few clues.
        for i in range(9):
            if np.count_nonzero(board[i, :]) < 2:
                for j in range(9):
                    weights[i][j] += 0.5
        for j in range(9):
            if np.count_nonzero(board[:, j]) < 2:
                for i in range(9):
                    weights[i][j] += 0.5
                    
        # Sort empty cells by weight (high-to-low).
        empty_cells = [(i, j) for i in range(9) for j in range(9) if board[i][j] == 0]
        empty_cells.sort(key=lambda pos: weights[pos[0]][pos[1]], reverse=True)
        
        # 6. Fill extra clues until target clue count is reached.
        for (i, j) in empty_cells:
            if curr_clues >= target_clues:
                break
            if board[i][j] == 0:
                board[i][j] = solution[i][j]
                curr_clues += 1
                
        if count_solutions(board, limit=2) == 1:
            return board


def generate_and_save_puzzle(difficulty, filename="../Sudoku/data/sudoku_data.csv"):
    # Ensure that the target directory exists.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Generate puzzle and solution
    solution = generate_sudoku_solution()
    puzzle = generate_sudoku_puzzle(solution, difficulty)
    unavoidable_sets = find_unavoidable_sets(solution)
    
    # Prepare data: flatten_grid and serialize_sets should be defined elsewhere in your code.
    data = {
        "puzzle": flatten_grid(puzzle),
        "solution": flatten_grid(solution),
        "unavoidable_sets": serialize_sets(unavoidable_sets),
        "difficulty": difficulty.name,
        "clues": np.count_nonzero(puzzle),
        "date_generated": datetime.now().isoformat()
    }
    
    # Append to CSV in the specified file path.
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if f.tell() == 0:  # Write header only once if file is empty.
            writer.writeheader()
        writer.writerow(data)
        

if __name__ == "__main__":
    # Generate and save 10 Hard puzzles
    
    for _ in range(10):
        generate_and_save_puzzle(Difficulty.HARD)
    
    # Load and inspect data
    puzzles, solutions, metadata = read_data()
    counter = [0]
    solved = backtrack_solve(puzzles[2],return_board=True)
    
    visualize_multiple([puzzles[2],solved, solutions[2]], titles=["Puzzle", "solved", "Solution"], unavoidable_sets_list = [None, None, metadata[2]["unavoidable_sets"]])
    
    # print(puzzles[2] == solutions[2])

    '''
    print(f"Loaded {len(puzzles)} puzzles.")
    print("First puzzle's unavoidable sets:", metadata[0]["unavoidable_sets"])
    '''
    # visualize(ex_solution, title="Sudoku Solution with Unavoidable Sets Highlighted", unavoidable_sets = unavoidable_sets)


