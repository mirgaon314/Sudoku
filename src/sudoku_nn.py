from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data_generation import read_data
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np


def train(puzzle, solution):
    X = puzzle.reshape(puzzle.shape[0], 81)
    y = solution.reshape(solution.shape[0], 81) - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the Keras model.
    model = Sequential()
    model.add(Dense(81, input_shape=(81,), activation='relu'))
    model.add(Dense(27, activation='relu'))
    model.add(Dense(27, activation='relu'))
    model.add(Dense(9, activation='relu'))
    # Output layer: predict 9 classes (0-8) for each of the 81 cells.
    model.add(Dense(81 * 9, activation='softmax'))
    model.add(Reshape((81, 9)))
    
    # Compile the model using sparse categorical crossentropy.
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    # Train the model on the training data.
    model.fit(X_train, y_train, epochs=150, batch_size=10)
    
    # Evaluate the model on the test data.
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test Accuracy: %.2f' % (accuracy * 100))
    model.save("C:/Users/Gaon Park/Documents/git/Sudoku/models/model_1.keras")



if __name__ == "__main__":
    puzzle, solution = read_data("../Sudoku/data/sudoku.csv",100000) #1000000
    train(puzzle, solution)
    # model = load_model("C:/Users/Gaon Park/Documents/git/Sudoku/models/sudoku_solver_model.h5")
    # predictions = model.predict(new_data)

    # predicted_digits = np.argmax(predictions, axis=-1)
    # print(predicted_digits)