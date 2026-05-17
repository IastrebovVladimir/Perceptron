import numpy as np
from itertools import product

from perceptron import Perceptron
from data import prepare_data


def k_fold_indices(n_samples, k=5, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    return np.array_split(indices, k)


def standardize_train_test(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def main():
    X_train_full, X_test, y_train_full, y_test, mean, std = prepare_data()

    learning_rates = [0.001, 0.01, 0.1, 0.5]
    batch_sizes = [1, 16, 32, 64]
    folds = k_fold_indices(len(X_train_full), k=5, random_state=42)

    best_score = -1.0
    best_params = None

    print("5-fold cross-validation")
    for lr, batch_size in product(learning_rates, batch_sizes):
        scores = []

        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

            X_train = X_train_full[train_idx]
            y_train = y_train_full[train_idx]
            X_val = X_train_full[val_idx]
            y_val = y_train_full[val_idx]

            X_train_std, X_val_std = standardize_train_test(X_train, X_val)

            model = Perceptron(n_features=2, random_state=42)
            model.fit(X_train_std, y_train, X_val_std, y_val, epochs=100, lr=lr, batch_size=batch_size)
            scores.append(model.accuracy(X_val_std, y_val))

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"lr={lr}, batch={batch_size}: mean_acc={mean_score:.4f}, std={std_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = (lr, batch_size)

    print(f"\nBest params: lr={best_params[0]}, batch_size={best_params[1]}, mean_acc={best_score:.4f}")

    final_model = Perceptron(n_features=2, random_state=42)
    final_model.fit(X_train_full, y_train_full, X_test, y_test, epochs=100, lr=best_params[0], batch_size=best_params[1])
    print(f"Final model test accuracy: {final_model.accuracy(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
