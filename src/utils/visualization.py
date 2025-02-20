import matplotlib.pyplot as plt
import numpy as np

def plot_sudoku(grid, title="Sudoku", unavoidable_sets=None):
    """
    Visualizes a Sudoku grid using matplotlib.

    Args:
        grid (numpy.ndarray): A 9x9 Sudoku grid (0 represents empty cells).
        title (str): Title of the plot.
        unavoidable_sets (list of list of tuples): Each element is a list of (row, col) tuples (0-indexed)
            representing an unavoidable set. Each set will be highlighted with a distinct color.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))
    ax.grid(which="both", color="black", linewidth=2)
    ax.set_title(title, fontsize=16)

    # Draw thicker lines for 3x3 subgrids
    for i in range(0, 10, 3):
        ax.axhline(i, color="black", linewidth=4)
        ax.axvline(i, color="black", linewidth=4)

    # If unavoidable sets are provided, highlight each set in a different color.
    if unavoidable_sets is not None:
        # Define a list of colors to cycle through.
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
    # plt.show()  # Caller will display the plot.

def visualize(puzzle, title="Sudoku", unavoidable_sets=None):
    """
    Visualizes a Sudoku puzzle.

    Args:
        puzzle (numpy.ndarray): A 9x9 Sudoku puzzle grid.
        title (str): Title of the plot.
        unavoidable_sets (list of list of tuples): List of unavoidable sets to highlight.
    """
    plt.figure(figsize=(6, 6))
    plot_sudoku(puzzle, title=title, unavoidable_sets=unavoidable_sets)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example puzzle and solution (replace with your data)
    puzzle = np.array([
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

    solution = np.array([
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
    # Find unavoidable sets on the solution (or puzzle, as required).
    # This returns a list of unavoidable sets (each a list of cell coordinates).
    # unavoidable_sets = find_unavoidable_sets(solution)

    # Visualize the solution with unavoidable cells highlighted.
    visualize(solution)