import matplotlib.pyplot as plt
import numpy as np

def plot_sudoku(grid, title="Sudoku", unavoidable_sets=None, ax=None):
    """
    Visualizes a Sudoku grid using matplotlib on a given axis.
    
    Args:
        grid (numpy.ndarray): A 9x9 Sudoku grid (0 represents empty cells).
        title (str): Title of the plot.
        unavoidable_sets (list of list of tuples): Each element is a list of (row, col) tuples (0-indexed)
            representing an unavoidable set.
        ax (matplotlib.axes.Axes): The axes on which to plot. If None, a new figure and axes are created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        ax.clear()  # clear the axis if reusing

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))
    ax.grid(which="both", color="black", linewidth=2)
    ax.set_title(title, fontsize=16)

    # Draw thicker lines for 3x3 subgrids.
    for i in range(0, 10, 3):
        ax.axhline(i, color="black", linewidth=4)
        ax.axvline(i, color="black", linewidth=4)

    # If unavoidable sets are provided, highlight each set with a distinct color.
    if unavoidable_sets is not None:
        colors = ['lightgray', 'lightblue', 'lightgreen', 'lightpink', 'khaki', 'wheat', 'thistle']
        for idx, u_set in enumerate(unavoidable_sets):
            color = colors[idx % len(colors)]
            for (i, j) in u_set:
                # Adjust y-coordinate since row 0 is at the top.
                rect = plt.Rectangle((j, 9 - i - 1), 1, 1, facecolor=color, alpha=0.5, edgecolor="none")
                ax.add_patch(rect)

    # Fill in the numbers (centered in each cell)
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                ax.text(j + 0.5, 8.5 - i, str(grid[i][j]),
                        fontsize=16, ha="center", va="center")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Do not call plt.show() here so we can combine multiple plots in one figure.

def visualize_multiple(boards, titles=None, unavoidable_sets_list=None):
    """
    Visualizes multiple Sudoku boards side-by-side.
    
    Args:
        boards (list of numpy.ndarray): List of 9x9 Sudoku boards.
        titles (list of str): List of titles for each board. If None, a default title is used.
        unavoidable_sets_list (list): A list of unavoidable sets corresponding to each board.
    """
    n = len(boards)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 6))
    
    # Ensure axs is iterable (even if only one board is provided)
    if n == 1:
        axs = [axs]
    
    for i, board in enumerate(boards):
        title = titles[i] if titles and i < len(titles) else f"Sudoku {i+1}"
        usets = unavoidable_sets_list[i] if unavoidable_sets_list and i < len(unavoidable_sets_list) else None
        plot_sudoku(board, title=title, unavoidable_sets=usets, ax=axs[i])
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Example puzzle and solution (replace these with your own boards as needed)
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
    
    # You can optionally pass a list of unavoidable sets for each board.
    # For this example, we'll assume there are no unavoidable sets.
    visualize_multiple([board1, board2],
                        titles=["Puzzle", "Solution"],
                        unavoidable_sets_list=[None, None])
