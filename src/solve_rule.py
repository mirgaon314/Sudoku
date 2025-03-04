import numpy as np
import pandas as pd
import copy
from data_generation import read_data, valid_numbers, check_valid

def update_candidate(board, candidate_board):
    for row in range(0,9):
        for col in range(0,9):
            if(board[row][col] == 0):
                candidate_board[row][col].clear()
                candidate_board[row][col].extend(valid_numbers(board, row, col))
    intersection_pointing(candidate_board)
    intersection_claiming(candidate_board)
    
def intersection_pointing(candidate_board):
    '''
    if the candidate in the box is in a single row or col, the rest of the row or col in other boxes can be removed
                    8 6 () <- can be 2
                    4 3 () <- can be 2
            2       1 () 9      thus col 8 cannot have 2 except for these two
    '''
    change = True
    while change:
        change = False
        for square_row in range(0, 9, 3):
            for square_col in range(0, 9, 3):
                for candidate in range(1,10):
                    cells_with_candidate = [
                        (row, col)
                        for row in range(square_row, square_row + 3)
                        for col in range(square_col, square_col + 3)
                        if candidate in candidate_board[row][col]
                    ]
                    if len(cells_with_candidate) > 1:
                        row_c = [r for r, c in cells_with_candidate]
                        col_c = [c for r, c in cells_with_candidate]
                        if all(r == row_c[0] for r in row_c):
                            for i in range(9):
                                if not (square_col <= i < square_col + 3):
                                    if candidate in candidate_board[row_c[0]][i]:
                                        candidate_board[row_c[0]][i].remove(candidate)
                                        change = True
                            
                        if all(c == col_c[0] for c in col_c):
                            for i in range(9):
                                if not (square_row <= i < square_row + 3):
                                    if candidate in candidate_board[i][col_c[0]]:
                                        candidate_board[i][col_c[0]].remove(candidate)
                                        change = True

def intersection_claiming(candidate_board):
    '''
    if the single candidate number for the row or col is all located in a box, the other cells in that box removes the candidate number
    ex)
    1 2 3 4 5 6  7 () ()   <- 8 and 9 
                () () ()    thus no 8 or 9 in the other cells in box
                () () ()
    '''
    change = True
    while change:
        change = False
        for candidate in range(1, 10):
            for i in range(9):
                cols_ = [col for col, cell in enumerate(candidate_board[i]) if candidate in cell]
                if not cols_:
                    continue
                # Check if all these columns lie in the same 3x3 block.
                if all(3 * (cols_[0] // 3) <= c < 3 * (cols_[0] // 3) + 3 for c in cols_):
                    square_row = 3 * (i // 3)
                    square_col = 3 * (cols_[0] // 3)
                    for row in range(square_row, square_row + 3):
                        for col in range(square_col, square_col + 3):
                            if row != i and candidate in candidate_board[row][col]:
                                candidate_board[row][col].remove(candidate)
                                change = True

                rows_ = [row for row in range(9) if candidate in candidate_board[row][i]]
                if not rows_:
                    continue
                # Check if all these rows lie in the same 3x3 block.
                if all(3 * (rows_[0] // 3) <= r < 3 * (rows_[0] // 3) + 3 for r in rows_):
                    square_row = 3 * (rows_[0] // 3)
                    square_col = 3 * (i // 3)
                    for row in range(square_row, square_row + 3):
                        for col in range(square_col, square_col + 3):
                            if col != i and candidate in candidate_board[row][col]:
                                candidate_board[row][col].remove(candidate)
                                change = True

                            
def naked_pair(candidate_board):
    change = True
    while change:
        change = False
        for row in range(9):
            for col1 in range(9):
                for col2 in range(9):
                    if(col1 != col2):
                        if(candidate_board[row][col1] == candidate_board[row][col2] and len(candidate_board[row][col1])):
# need to work


def naked_single(board, candidate_board):
    change = True
    while change:
        change = False
        for row in range(9):
            for col in range(9):
                if len(candidate_board[row][col]) == 1:
                    board[row][col] = candidate_board[row][col][0]
                    candidate_board[row][col] = [0]
                    update_candidate(board, candidate_board)
                    change = True

def hidden_single(board, candidate_board):
    '''
    find the row, col, square that has only one missing spot
    if the row has only one cell that the candidate is possible, fill up with candidate
    '''
    change = True
    while change:
        change = False
        for candidate in range(1,10):
            for i in range(9):
                cols_with_candidate = [col for col, cell in enumerate(candidate_board[i]) if candidate in cell]
                if len(cols_with_candidate) == 1:
                    col = cols_with_candidate[0]
                    board[i][col] = candidate
                    candidate[i][col] = [0]
                    change = True
                    update_candidate(board, candidate_board)
                rows_with_candidate = [row for row in range (9) if candidate in candidate_board[row][i]]
                if len(rows_with_candidate) == 1:
                    row = rows_with_candidate[0]
                    board[row][i] = candidate
                    candidate[row][i] = [0]
                    change = True
                    update_candidate(board, candidate_board)
            for square_row in range(0, 9, 3):
                for square_col in range(0, 9, 3):
                    cells_with_candidate = [
                        (row, col)
                        for row in range(square_row, square_row + 3)
                        for col in range(square_col, square_col + 3)
                        if candidate in candidate_board[row][col]
                    ]
                    if len(cells_with_candidate) == 1:
                        row_c, col_c = cells_with_candidate[0]
                        board[row_c][col_c] = candidate
                        candidate_board[row_c][col_c] = [0]
                        change = True
                        update_candidate(board, candidate_board)
    
def rule_based_solver(board, return_board = False):
    def _solve(b):
        naked_single(b, candidate_board)
        hidden_single(b, candidate_board)
        return b if return_board else True

    candidate_board = [[[] for _ in range(9)] for _ in range(9)]
    for i in range(0,9):
        for j in range(0,9):
            candidate_board[i][j].append(0)
    update_candidate(board, candidate_board)

    if return_board:
        # Create a deep copy to preserve the original board.
        board_copy = copy.deepcopy(board)
        result = _solve(board_copy)
        if result is not None:
            return result
        return board_copy
    else:
        return _solve(board)

if __name__ == "__main__":
    board1 = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    board2 = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])
    rule_based_solver(board1)