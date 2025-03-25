# Sudoku
https://www.csc.kth.se/utbildning/kandidatexjobb/datateknik/2012/rapport/berggren_patrik_OCH_nilsson_david_K12011.pdf 
- paper about the algorithm of sudoku
- 3 different method to solve sudoku: brute force, human logic, neural network model

# least clue for unique solution
https://www.tandfonline.com/doi/epdf/10.1080/10586458.2013.870056?src=getftr&utm_source=sciencedirect_contenthosting&getft_integrator=sciencedirect_contenthosting
- least number of clue for unique solution, 17 clues are need
- also, at least one elemet of all the unavoidable sets needs to be included for unique solution

# what is unavoidable sets
http://sudopedia.enjoysudoku.com/Deadly_Pattern.html
- Unavoidable sets are deadly patterns that they can be interchange such as

[1]    [2]                  [1]    [3]
[2]    [1]      or          [2]    [1]
                            [3]    [2]
- if at least one element is included in the puzzle, it will ensure the arrangement of the sets

# Human logic methods
https://namu.wiki/w/스도쿠/공략법#s-5

# Bolzmann machine in Pytorch
https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/


# Sudoku database
https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download


# this week work
https://medium.com/data-science/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

Implement the keras example on your own computer
Adjust that model slightly (more layers, ect…)
Format the sudoku input puzzles as 1D numpy arrays. A function that takes in the input sudoku from your data set, and gives back a 1D version