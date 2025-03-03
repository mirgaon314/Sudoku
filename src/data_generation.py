import numpy as np
import pandas as pd
from utils.visualization import visualize
import random
from enum import Enum
from itertools import combinations
from itertools import permutations
from itertools import product
from collections import Counter

class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4

def read_data(): # Load a CSV dataset
    data_directory_path = (
        f"../data/"
    )
    data = pd.read_csv(data_directory_path + "sudoku.csv")
    puzzles = data["puzzle"].apply(lambda x: [int(c) for c in x]).tolist()
    solutions = data["solution"].apply(lambda x: [int(c) for c in x]).tolist()

    return puzzles, solutions

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

def backtrack_solve(board, find_unique=False):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                num_list = valid_numbers(board, i, j)
                random.shuffle(num_list)
                for num in num_list:
                    if check_valid(board, i, j, num):
                        board[i][j] = num
                        if backtrack_solve(board):
                            return True
                        board[i][j] = 0
                return False
    return True

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

def generate_sudoku_puzzle(solution,difficulty):
    num_clues = {
        Difficulty.EASY: 38,
        Difficulty.MEDIUM: 30,
        Difficulty.HARD: 23,
        Difficulty.EXPERT: 0  # based on the number of unavoidable sets
    }
    curr_clues = 0
    board = np.zeros((9,9),dtype = int)
    # first, use at least one element from each unavoidable sets for unique solution
    unavoid_set = find_unavoidable_sets(solution)
    for sets in unavoid_set:
        check = 0
        while check == 0:
            num = random.randint(0,len(sets)-1)
            row, col = sets[num]
            if board[row][col] == 0:
                board[row][col] = solution[row][col]
                curr_clues += 1
                check = 1

    while curr_clues < num_clues[difficulty]:
        row, col = np.random.randint(0, 9), np.random.randint(0, 9)
        if board[row][col] == 0:
            board[row][col] = solution[row][col]
            curr_clues += 1
    return board
        

if __name__ == "__main__":
    ex_solution = generate_sudoku_solution()
    ex_puzzle = generate_sudoku_puzzle(ex_solution, Difficulty.HARD)
    '''
    ex_solution = np.array([
        [6, 8, 5, 4, 3, 1, 9, 7, 2],
        [2, 9, 7, 8, 6, 5, 4, 1, 3],
        [3, 4, 1, 7, 2, 9, 5, 8, 6],
        [7, 3, 9, 2, 1, 6, 8, 4, 5],
        [1, 5, 6, 9, 4, 8, 3, 2, 7],
        [8, 2, 4, 3, 5, 7, 6, 9, 1],
        [4, 7, 3, 6, 9, 2, 1, 5, 8],
        [5, 6, 2, 1, 8, 4, 7, 3, 9],
        [9, 1, 8, 5, 7, 3, 2, 6, 4]
    ])
    unavoidable_sets = find_unavoidable_sets(ex_solution)
    # visualize(ex_solution)
    print("Unavoidable Sets:")
    for idx, u_set in enumerate(unavoidable_sets, start=1):
        print(f"Set {idx}: {u_set}")
    '''

    # visualize(ex_solution, title="Sudoku Solution with Unavoidable Sets Highlighted", unavoidable_sets = unavoidable_sets)

    visualize(ex_puzzle)

