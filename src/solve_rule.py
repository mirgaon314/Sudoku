import numpy as np
import pandas as pd
import copy
from data_generation import read_data, valid_numbers, check_valid
from itertools import combinations, permutations, product
from utils.visualization import visualize_multiple
from data_generation import read_data

def update_candidate(board, candidate_board):
    for row in range(0,9):
        for col in range(0,9):
            if(board[row][col] == 0):
                candidate_board[row][col].clear()
                candidate_board[row][col].extend(valid_numbers(board, row, col))
    intersection_pointing(candidate_board)
    intersection_claiming(candidate_board)
    naked_pair_triple_quadruple(candidate_board)

    
def intersection_pointing(candidate_board):
    '''
    If the candidate in the box is in a single row or col, the rest of the row or col in other boxes can be removed
                    8 6 () <- can be 2
                    4 3 () <- can be 2
            2       1 () 9      thus col 8 cannot have 2 except for these two
    '''
    change = True
    while change:
        change = False
        for square_row in range(0, 9, 3):
            for square_col in range(0, 9, 3):
                for candidate in range(1, 10):
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
    return candidate_board

def intersection_claiming(candidate_board):
    '''
    If the single candidate number for the row or col is all located in a box, the other cells in that box removes the candidate number
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
                cols_with_candidate = [col for col, cell in enumerate(candidate_board[i]) if candidate in cell]
                if len(cols_with_candidate) > 1 and all(3 * (cols_with_candidate[0] // 3) <= c < 3 * (cols_with_candidate[0] // 3) + 3 for c in cols_with_candidate):
                    square_row = 3 * (i // 3)
                    square_col = 3 * (cols_with_candidate[0] // 3)
                    for row in range(square_row, square_row + 3):
                        for col in range(square_col, square_col + 3):
                            if row != i and candidate in candidate_board[row][col]:
                                candidate_board[row][col].remove(candidate)
                                change = True

                rows_with_candidate = [row for row in range(9) if candidate in candidate_board[row][i]]
                if len(rows_with_candidate) > 1 and all(3 * (rows_with_candidate[0] // 3) <= r < 3 * (rows_with_candidate[0] // 3) + 3 for r in rows_with_candidate):
                    square_row = 3 * (rows_with_candidate[0] // 3)
                    square_col = 3 * (i // 3)
                    for row in range(square_row, square_row + 3):
                        for col in range(square_col, square_col + 3):
                            if col != i and candidate in candidate_board[row][col]:
                                candidate_board[row][col].remove(candidate)
                                change = True
    return candidate_board

def naked_pair_triple_quadruple(candidate_board):
    change = True
    while change:
        change = False

        # Check rows for naked pairs, triples, and quadruples
        for row in range(9):
            pairs = {}
            triples = {}
            quadruples = {}

            for col in range(9):
                length = len(candidate_board[row][col])
                if length == 2:
                    pair = tuple(candidate_board[row][col])
                    if pair in pairs:
                        pairs[pair].append(col)
                    else:
                        pairs[pair] = [col]
                elif length == 3:
                    triple = tuple(candidate_board[row][col])
                    if triple in triples:
                        triples[triple].append(col)
                    else:
                        triples[triple] = [col]
                elif length == 4:
                    quadruple = tuple(candidate_board[row][col])
                    if quadruple in quadruples:
                        quadruples[quadruple].append(col)
                    else:
                        quadruples[quadruple] = [col]

            for pair, cols in pairs.items():
                if len(cols) == 2:
                    for col in range(9):
                        if col not in cols:
                            if any(x in candidate_board[row][col] for x in pair):
                                candidate_board[row][col] = [x for x in candidate_board[row][col] if x not in pair]
                                change = True

            for triple, cols in triples.items():
                if len(cols) == 3:
                    for col in range(9):
                        if col not in cols:
                            if any(x in candidate_board[row][col] for x in triple):
                                candidate_board[row][col] = [x for x in candidate_board[row][col] if x not in triple]
                                change = True

            for quadruple, cols in quadruples.items():
                if len(cols) == 4:
                    for col in range(9):
                        if col not in cols:
                            if any(x in candidate_board[row][col] for x in quadruple):
                                candidate_board[row][col] = [x for x in candidate_board[row][col] if x not in quadruple]
                                change = True

        # Check columns for naked pairs, triples, and quadruples
        for col in range(9):
            pairs = {}
            triples = {}
            quadruples = {}

            for row in range(9):
                length = len(candidate_board[row][col])
                if length == 2:
                    pair = tuple(candidate_board[row][col])
                    if pair in pairs:
                        pairs[pair].append(row)
                    else:
                        pairs[pair] = [row]
                elif length == 3:
                    triple = tuple(candidate_board[row][col])
                    if triple in triples:
                        triples[triple].append(row)
                    else:
                        triples[triple] = [row]
                elif length == 4:
                    quadruple = tuple(candidate_board[row][col])
                    if quadruple in quadruples:
                        quadruples[quadruple].append(row)
                    else:
                        quadruples[quadruple] = [row]

            for pair, rows in pairs.items():
                if len(rows) == 2:
                    for row in range(9):
                        if row not in rows:
                            if any(x in candidate_board[row][col] for x in pair):
                                candidate_board[row][col] = [x for x in candidate_board[row][col] if x not in pair]
                                change = True

            for triple, rows in triples.items():
                if len(rows) == 3:
                    for row in range(9):
                        if row not in rows:
                            if any(x in candidate_board[row][col] for x in triple):
                                candidate_board[row][col] = [x for x in candidate_board[row][col] if x not in triple]
                                change = True

            for quadruple, rows in quadruples.items():
                if len(rows) == 4:
                    for row in range(9):
                        if row not in rows:
                            if any(x in candidate_board[row][col] for x in quadruple):
                                candidate_board[row][col] = [x for x in candidate_board[row][col] if x not in quadruple]
                                change = True

        # Check 3x3 blocks for naked pairs, triples, and quadruples
        for block_row in range(3):
            for block_col in range(3):
                pairs = {}
                triples = {}
                quadruples = {}
                for cell in range(9):
                    row = block_row * 3 + cell // 3
                    col = block_col * 3 + cell % 3
                    length = len(candidate_board[row][col])
                    if length == 2:
                        pair = tuple(candidate_board[row][col])
                        if pair in pairs:
                            pairs[pair].append((row, col))
                        else:
                            pairs[pair] = [(row, col)]
                    elif length == 3:
                        triple = tuple(candidate_board[row][col])
                        if triple in triples:
                            triples[triple].append((row, col))
                        else:
                            triples[triple] = [(row, col)]
                    elif length == 4:
                        quadruple = tuple(candidate_board[row][col])
                        if quadruple in quadruples:
                            quadruples[quadruple].append((row, col))
                        else:
                            quadruples[quadruple] = [(row, col)]

                for pair, cells in pairs.items():
                    if len(cells) == 2:
                        for cell in range(9):
                            row = block_row * 3 + cell // 3
                            col = block_col * 3 + cell % 3
                            if (row, col) not in cells:
                                if any(x in candidate_board[row][col] for x in pair):
                                    candidate_board[row][col] = [x for x in candidate_board[row][col] if x not in pair]
                                    change = True

                for triple, cells in triples.items():
                    if len(cells) == 3:
                        for cell in range(9):
                            row = block_row * 3 + cell // 3
                            col = block_col * 3 + cell % 3
                            if (row, col) not in cells:
                                if any(x in candidate_board[row][col] for x in triple):
                                    candidate_board[row][col] = [x for x in candidate_board[row][col] if x not in triple]
                                    change = True

                for quadruple, cells in quadruples.items():
                    if len(cells) == 4:
                        for cell in range(9):
                            row = block_row * 3 + cell // 3
                            col = block_col * 3 + cell % 3
                            if (row, col) not in cells:
                                if any(x in candidate_board[row][col] for x in quadruple):
                                    candidate_board[row][col] = [x for x in candidate_board[row][col] if x not in quadruple]
                                    change = True

    return candidate_board

def naked_single(board, candidate_board):
    for row in range(9):
        for col in range(9):
            if len(candidate_board[row][col]) == 1 and candidate_board[row][col] != [0]:
                if board[row][col] == 0:
                    board[row][col] = candidate_board[row][col][0]
                    candidate_board[row][col] = [0]
                    return

def hidden_single(board, candidate_board):
    '''
    find the row, col, square that has only one missing spot
    if the row has only one cell that the candidate is possible, fill up with candidate
    '''
    for candidate in range(1, 10):
        # Check rows for hidden singles
        for row in range(9):
            cols_with_candidate = [col for col, cell in enumerate(candidate_board[row]) if candidate in cell]
            if len(cols_with_candidate) == 1:
                col = cols_with_candidate[0]
                if board[row][col] == 0:
                    board[row][col] = candidate
                    candidate_board[row][col] = [0]
                    return
                    
        # Check columns for hidden singles
        for col in range(9):
            rows_with_candidate = [row for row in range(9) if candidate in candidate_board[row][col]]
            if len(rows_with_candidate) == 1:
                row = rows_with_candidate[0]
                if board[row][col] == 0:
                    board[row][col] = candidate
                    candidate_board[row][col] = [0]
                    return
                    
        # Check 3x3 blocks for hidden singles
        for block_row in range(0, 9, 3):
            for block_col in range(0, 9, 3):
                cells_with_candidate = [
                    (row, col)
                    for row in range(block_row, block_row + 3)
                    for col in range(block_col, block_col + 3)
                    if candidate in candidate_board[row][col]
                ]
                if len(cells_with_candidate) == 1:
                    row_c, col_c = cells_with_candidate[0]
                    if board[row_c][col_c] == 0:
                        board[row_c][col_c] = candidate
                        candidate_board[row_c][col_c] = [0]
                        return
    
def rule_based_solver(board, return_board = False):
    def _solve(b):
        change = True
        while change:
            change = False
            board_copy = copy.deepcopy(b)
            naked_single(b, candidate_board)
            if(np.any(board_copy != b)):
                update_candidate(b, candidate_board)
                change = True
                continue
            hidden_single(b, candidate_board)
            if(np.any(board_copy != b)):
                update_candidate(b, candidate_board)
                change = True
        if(np.all(b > 0)):
            return b if return_board else True
        return False

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
    puzzle, solution = read_data("../Sudoku/data/sudoku.csv",1000000)
    num = 1
    rule_based_solver(puzzle[num])
    visualize_multiple([puzzle[num], solution[num]], titles=["solved", "Solution"], unavoidable_sets_list=[None, None])
    