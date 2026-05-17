import numpy as np
import matplotlib.pyplot as plt

from data import prepare_data
from perceptron import Perceptron


def plot_losses(train_losses, val_losses):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss during training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", alpha=0.7)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_values = np.linspace(x_min, x_max, 200)

    if abs(model.w[1, 0]) > 1e-12:
        y_values = -(model.w[0, 0] * x_values + model.b) / model.w[1, 0]
        plt.plot(x_values, y_values, color="black", label="Decision boundary")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision boundary")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    X_train, X_test, y_train, y_test, mean, std = prepare_data()

    model = Perceptron(n_features=X_train.shape[1], init_mode="random", random_state=42)

    model.fit(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        lr=0.1,
        batch_size=32
    )

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    plot_losses(model.train_losses, model.val_losses)
    plot_decision_boundary(model, X_test, y_test)


if __name__ == "__main__":
    main()