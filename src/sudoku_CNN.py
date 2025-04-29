from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data_generation import read_data
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np


def generate_smaller(puzzle, solution, n):
    smaller = solution.copy()
    blanks = list(zip(*np.where(puzzle == 0)))
    k = min(n, len(blanks))
    chosen = np.random.choice(len(blanks), k, replace=False)
    for idx in chosen:
        smaller[blanks[idx]] = 0
    return smaller


def curriculum_train(puzzle, solution, n_values = (0,44,46,48,50,52), epochs_per_stage=20, batch_size=32, base_model=None, model_path=None):
    X = puzzle.reshape(-1, 81)
    Y = solution.reshape(-1, 81)

    # train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    # test labels 1–9 → 0–8
    y_test = (Y_test - 1).clip(0)  

    # build or reuse model
    if base_model is None:
        # 1) Input: a single‑channel 9×9 “image”
        inp = layers.Input(shape=(9, 9, 1))

        # 2) Initial conv to expand to 64 features
        x = layers.Conv2D(32, (3,3), padding="same", dilation_rate= 1, use_bias=False)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, (3,3), padding="same", dilation_rate= 2, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # 4) Bottleneck conv to build bigger receptive field
        x = layers.Conv2D(128, (3,3), padding="same", dilation_rate= 4, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # x = layers.Conv2D(256, (3,3), padding="same", dilation_rate= 4, use_bias=False)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation("relu")(x)

        # 5) Dropout for regularization
        x = layers.Dropout(0.3)(x)

        # 6) Final 1×1 conv → 9 feature maps (one per digit class), softmax per cell
        x = layers.Conv2D(9, 1, padding="same", activation="softmax")(x)

        # 7) Reshape to (81 cells, 9 classes)
        out = layers.Reshape((81, 9))(x)

        model = Model(inp, out)
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=1e-3, decay=1e-5),
            metrics=["accuracy"]
        )
        model.summary()
        print("→ New model created")
    else:
        model = base_model
        print("→ Continuing from provided model")

    # sequential curriculum
    for stage, n in enumerate(n_values, 1):
        print(f"\n=== Stage {stage}/{len(n_values)}: mix up to n={n} removals ===")
        Xs, Ys = [], []

        # for each puzzle, pick one k ∈ n_values[:stage] at random
        levels = n_values[:stage+1]
        for i in range(len(X_train)):
            p = X_train[i].reshape(9,9)
            s = Y_train[i].reshape(9,9)
            # choose a difficulty level for this puzzle
            k = np.random.choice(levels)
            Ys.append(generate_smaller(p, s, k).reshape(81))
            Xs.append(p.reshape(81))

        # reshape test puzzles into 9×9×1 images
        Xs = np.vstack(Xs).astype('float32')
        Xs = Xs.reshape(-1, 9, 9, 1)
        raw_labels = np.vstack(Ys)
        # convert 1–9→0–8, blanks stay 0
        y_stage = np.where(raw_labels>0, raw_labels-1, 0)

        # train
        model.fit(Xs, y_stage,
                  epochs=epochs_per_stage,
                  batch_size=batch_size,
                  shuffle=True, verbose=1)
        loss, acc = model.evaluate(Xs, y_stage, verbose=0)
        print(f"→ Stage {stage} train acc: {acc*100:.2f}%")

    # final test eval
    X_test = X_test.reshape(-1, 9, 9, 1)
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n*** Final Test Accuracy: {acc*100:.2f}% ***")

    if model_path:
        model.save(model_path)
        print(f"→ Model saved to {model_path}")
    return model

if __name__ == "__main__":
    # Load data from your CSV file.
    puzzle, solution = read_data("../Sudoku/data/sudoku_9mil.csv", 9000000)

    # Option 1: Train a model from scratch.
    curriculum = [0, 44, 46, 48, 50, 52]
    model = curriculum_train(puzzle, solution,
                             n_values=curriculum,
                             epochs_per_stage=10,
                             batch_size=32,
                             base_model=None,
                             model_path="models/model_2.keras")

    # Option 2: Continue training an existing model.
    # For example, uncomment the following if you want to use a pre-saved model:
    # base_model = load_model("C:/Users/Gaon Park/Documents/git/Sudoku/models/model_1.keras")
    # model = curriculum_train(puzzle, solution, next_n_values,
    #                          epochs_per_stage=50, batch_size=10,
    #                          base_model=base_model,
    #                          model_path="C:/Users/Gaon Park/Documents/git/Sudoku/models/model_1.keras")
