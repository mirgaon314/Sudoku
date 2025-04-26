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

# 9 million
https://www.kaggle.com/datasets/rohanrao/sudoku


# this week work
https://medium.com/data-science/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

Implement the keras example on your own computer
Adjust that model slightly (more layers, ect…)
Format the sudoku input puzzles as 1D numpy arrays. A function that takes in the input sudoku from your data set, and gives back a 1D version

https://www.heatonresearch.com/2017/06/01/hidden-layers.html
https://stackoverflow.com/questions/52485608/how-to-choose-the-number-of-hidden-layers-and-nodes


1. Write a program/function that creates minus N puzzles
2. Use 100K of N=1 puzzles and figure out a NN system that solves it
3.  Scale up to 10 million puzzles and see if that same NN system does

https://medium.com/aiguys/curriculum-learning-83b1b2221f33

Convolutional Neural Networks
cnn or gnn
44: 57
45: 12662 
46: 198940
47: 455004
48: 263913
49: 61122 
50: 7748  
51: 526   
52: 28

44: 5
45: 1291
46: 19981
47: 45218
48: 26548
49: 6139
50: 771
51: 44
52: 3
