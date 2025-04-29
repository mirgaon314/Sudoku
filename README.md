# Sudoku Puzzle Generation and Solver

## Project Overview
This project focuses on creating a Sudoku puzzle generator and developing two types of solvers: a human logic-based solver, and a neural network (NN) based solver.

## Features

### 1. Sudoku Puzzle Generator
- Developed an algorithm for Sudoku puzzle generation using backtracking to ensure a complete, valid Sudoku solution.
- Implemented logic to identify "unavoidable sets," which are groups of cells crucial for puzzle uniqueness, ensuring each puzzle includes clues to address these sets.
- Utilized a greedy hitting set algorithm with additional heuristics (covering rows, columns, and blocks effectively) to efficiently select clues necessary for puzzle uniqueness.
- Not enough to ensure the uniqueness (take too long time)

### 2. Logic-Based Solver
- Implemented various human-inspired Sudoku solving techniques, including:
  - Candidate elimination through intersection pointing and claiming strategies.
  - Identification and application of naked pairs, triples, and quadruples to reduce possibilities in rows, columns, and blocks.
  - Utilization of naked and hidden singles to logically deduce cell values efficiently.
- Ensured the solver avoids brute force, relying strictly on logical inference and constraint propagation.
- refer to https://namu.wiki/w/스도쿠/공략법#s-5

### 3. Neural Network-Based Solver
- **Fully-Connected Layers Model:**
  - Built a sequential neural network with fully-connected dense layers.
  - Implemented curriculum learning, progressively increasing puzzle difficulty by incrementally removing clues during training.
  - Achieved an accuracy of approximately 15% using 1 million sudoku database.

- **Convolutional Neural Network (CNN) Model:**
  - Developed a CNN architecture with multiple convolutional layers and dilation rates to exploit spatial dependencies in Sudoku grids.
  - Applied dropout and batch normalization for regularization and improved convergence.
  - Achieved significantly improved performance with an accuracy of 48% using 1 million sudoku database.
 
- Used datas
  - 1 million sudoku database
  - https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download
  - (graph)
 
  - 9 million sudoku database
  - https://www.kaggle.com/datasets/rohanrao/sudoku




# reference

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

https://medium.com/data-science/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
https://www.heatonresearch.com/2017/06/01/hidden-layers.html
https://stackoverflow.com/questions/52485608/how-to-choose-the-number-of-hidden-layers-and-nodes
https://medium.com/aiguys/curriculum-learning-83b1b2221f33

44: 57
45: 12662 
46: 198940
47: 455004
48: 263913
49: 61122 
50: 7748  
51: 526   
52: 28

4: 8
5: 11
6: 25
7: 72
8: 103
9: 188
10: 282
11: 456
12: 642
13: 1044
14: 1390
15: 1922
16: 2619
17: 3584
18: 4825
19: 6558
20: 8405
21: 11128
22: 14376
23: 18327
24: 23417
25: 29746
26: 37918
27: 47977
28: 59119
29: 73584
30: 90280
31: 110194
32: 133085
33: 160784
34: 193094
35: 228562
36: 269619
37: 315259
38: 364182
39: 415920
40: 470816
41: 523576
42: 573332
43: 614257
44: 642208
45: 650810
46: 635339
47: 591077
48: 517507
49: 424195
50: 314756
52: 118817
53: 56760
54: 21546
55: 6304
56: 1297
57: 182
58: 18
