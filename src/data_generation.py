import numpy as np
import pandas as pd
from utils.visualization import visualize
import random

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

def backtrack_solve(board): # brute-force method
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                # randomize because it is also used for generating solution
                num_list = list[range(1,10)]
                random.shuffle(num_list)
                for num in num_list:
                    if check_valid(board, i, j, num):
                        board[i][j] = num
                        if backtrack_solve(board):
                            return True
                        # return into 0 when the recursion didn't work
                        board[i][j] = 0
                return False
    return False

def generate_sudoku_solution():
    board = np.zeros((9,9),dtype = int)
    backtrack_solve(board)
    return board

def generate_sudoku_puzzle(solution):
    board = np.copy(solution)




if __name__ == "__main__":
    ex_solution = generate_sudoku_solution()
    visualize(ex)

