from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data_generation import read_data
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np


def generate_smaller(puzzle, solution, n):
    smaller_solution = solution.copy()
    empty_positions = list(zip(*np.where(puzzle == 0)))
    
    n = min(n, len(empty_positions))
    
    chosen_indices = np.random.choice(len(empty_positions), n, replace=False)
    
    for idx in chosen_indices:
        pos = empty_positions[idx]
        smaller_solution[pos] = 0
        
    return smaller_solution


def curriculum_train(puzzle, solution, n_values, epochs_per_stage=50, batch_size=10, base_model=None, model_path=None,
                     difficulty_threshold=46, mix_ratio=0.5):
    # Prepare the original data.
    X_orig = puzzle.reshape(puzzle.shape[0], 81)
    # y_orig contains digits 1-9
    y_orig = solution.reshape(solution.shape[0], 81)
    
    # Split into training and testing (test set uses full solutions)
    X_train_orig, X_test, y_train_full, y_test_full = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )
    
    # Prepare the test set:
    # Note: We subtract 1 from the digits (1-9 becomes 0-8) for the network prediction.
    X_test = X_test  # Shape: (#test_samples, 81)
    y_test = y_test_full.reshape(-1, 81) - 1  # Test labels are complete solutions.
    
    if base_model is None:
        model = Sequential([
            Input(shape=(81,)),
            Dense(81, activation='relu'),
            Dense(27, activation='relu'),
            Dense(27, activation='relu'),
            Dense(9, activation='relu'),
            Dense(81 * 9, activation='softmax'),
            Reshape((81, 9))
        ])
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
    else:
        model = base_model
    
    # Loop over each curriculum stage (each n represents a difficulty level)
    for n in n_values:
        print(f"\n--- Training Stage: n = {n} ---")
        training_puzzles_stage = []
        training_labels_raw_stage = []  # Raw labels: digits 1-9, with 0 for removed cells.
        
        # Build the stage dataset using all training puzzles.
        for i in range(len(X_train_orig)):
            # Reshape back to a 9x9 grid.
            puzzle_2d = X_train_orig[i].reshape(9, 9)
            solution_2d = y_train_full[i].reshape(9, 9)
            empty_count = np.sum(puzzle_2d == 0)
            
            # If n is greater than the available blanks, use the full solution.
            if n > empty_count:
                modified_solution = solution_2d.copy()
            else:
                # If n is above the difficulty threshold, mix in an easier sample.
                if n > difficulty_threshold:
                    # Generate the hard sample (n removals)
                    hard_sample = generate_smaller(puzzle_2d, solution_2d, n)
                    # Generate an easier sample at the difficulty threshold.
                    if difficulty_threshold > empty_count:
                        easier_sample = solution_2d.copy()
                    else:
                        easier_sample = generate_smaller(puzzle_2d, solution_2d, difficulty_threshold)
                    # Choose which sample to use based on mix_ratio.
                    if np.random.rand() < mix_ratio:
                        modified_solution = hard_sample
                    else:
                        modified_solution = easier_sample
                else:
                    # For n below threshold, use the normal procedure.
                    modified_solution = generate_smaller(puzzle_2d, solution_2d, n)
            
            training_puzzles_stage.append(puzzle_2d)
            training_labels_raw_stage.append(modified_solution)
        
        # Convert stage data to numpy arrays.
        X_stage = np.array(training_puzzles_stage).reshape(-1, 81)
        y_stage_raw = np.array(training_labels_raw_stage).reshape(-1, 81)
        
        # Adjust labels: subtract 1 from non-blank cells so digits 1-9 become 0-8.
        # Blank cells (0) are given a dummy value (0) but their loss is ignored via the sample weights.
        y_stage = np.where(y_stage_raw != 0, y_stage_raw - 1, 0)
        
        # Train on the current stage.
        model.fit(X_stage, y_stage,
                  epochs=epochs_per_stage,
                  batch_size=batch_size)
        
        # Optionally, check the accuracy on the stage's training data.
        stage_loss, stage_acc = model.evaluate(X_stage, y_stage, verbose=0)
        print(f"Stage n = {n} training accuracy: {stage_acc*100:.2f}%")
    
    # After all stages, evaluate on the test set.
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

    if model_path is not None:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    return model

if __name__ == "__main__":
    # Load data from your CSV file.
    puzzle, solution = read_data("../Sudoku/data/sudoku.csv", 100000)
    
    # Define your curriculum stages.
    n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21, 25, 29, 33, 39, 45, 50, 55, 60]
    next_n_values = [33, 39, 45, 50, 55, 60] # 46 had 60.95%,  47 = 51.73%, 48 = 28.74, 49 = 14.85, 50 = 11%
    


    # Option 1: Train a model from scratch.
    model = curriculum_train(puzzle, solution, n_values,
                             epochs_per_stage=10, batch_size=10,
                             base_model=None,
                             model_path="C:/Users/Gaon Park/Documents/git/Sudoku/models/model_2.keras", difficulty_threshold=46, mix_ratio=0.5)
    
    # Option 2: Continue training an existing model.
    # For example, uncomment the following if you want to use a pre-saved model:
    # base_model = load_model("C:/Users/Gaon Park/Documents/git/Sudoku/models/model_1.keras")
    # model = curriculum_train(puzzle, solution, next_n_values,
    #                          epochs_per_stage=50, batch_size=10,
    #                          base_model=base_model,
    #                          model_path="C:/Users/Gaon Park/Documents/git/Sudoku/models/model_1.keras")
    # model = curriculum_train(puzzle, solution, n_values, epochs_per_stage=50, batch_size=10)
    
    # Save the trained model.
    # model.save("C:/Users/Gaon Park/Documents/git/Sudoku/models/model_1.keras")