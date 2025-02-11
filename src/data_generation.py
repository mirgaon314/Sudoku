import numpy as np
import pandas as pd
from utils.visualization import visualize
import random
from enum import Enum

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

def backtrack_solve(board, find_unique=False): # brute-force method
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                # randomize because it is also used for generating solution
                num_list = valid_numbers(board, i, j)
                random.shuffle(num_list)
                for num in num_list:
                    if check_valid(board, i, j, num):
                        board[i][j] = num
                        if backtrack_solve(board):
                            return True
                        # return into 0 when the recursion didn't work
                        board[i][j] = 0
                return False
    return True

def find_unavoidable_sets(board):
    sets = []
    combination = [[1,2], [1,3], [2,3]]
    # 1X2 blue prints / 1 band = 3 rows / 1 stack = 3 column/ 1X2 means 2 box in same row
    for box_row in range(3):
        for com in combination:
            

    return sets

def generate_sudoku_solution():
    board = np.zeros((9,9),dtype = int)
    generate_3x3_diagonal_box(board)
    backtrack_solve(board)
    return board



def generate_sudoku_puzzle(solution,difficulty):
    board = np.zeros((9,9),dtype = int)
    num_clues = {
        Difficulty.EASY: 38,
        Difficulty.MEDIUM: 30,
        Difficulty.HARD: 23,
        Difficulty.EXPERT: 0  # based on the number of unavoidable sets
    }
    curr_clues = 0

    # first, use at least one element from each unavoidable sets for unique solution
    # unavoid_set = find_unavoidable_sets(solution)
    # need to be worked

    while curr_clues < num_clues[difficulty]:
        row, col = np.random.randint(0, 9), np.random.randint(0, 9)
        if board[row][col] == 0:
            board[row][col] = solution[row][col]
            curr_clues += 1
    
    return board



if __name__ == "__main__":
    ex_solution = generate_sudoku_solution()
    ex_puzzle = generate_sudoku_puzzle(ex_solution, Difficulty.HARD)
    visualize(ex_puzzle)
    

