import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from data import prepare_data


class MomentumPerceptron(Perceptron):
    def __init__(self, n_features, init_mode="random", random_state=42, beta=0.0):
        super().__init__(n_features, init_mode=init_mode, random_state=random_state)
        self.beta = beta
        self.velocity_w = np.zeros_like(self.w)
        self.velocity_b = 0.0

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        n_samples = X_train.shape[0]

        self.train_losses = []
        self.val_losses = []

        for _ in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                dw = (X_batch.T @ (y_pred - y_batch)) / len(X_batch)
                db = np.mean(y_pred - y_batch)

                self.velocity_w = self.beta * self.velocity_w - lr * dw
                self.velocity_b = self.beta * self.velocity_b - lr * db

                self.w += self.velocity_w
                self.b += self.velocity_b

            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append(self.compute_loss(y_val, self.forward(X_val)))

        return self


def plot_loss_curves(models):
    plt.figure(figsize=(10, 6))
    for label, model in models.items():
        epochs = np.arange(1, len(model.val_losses) + 1)
        plt.plot(epochs, model.val_losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Momentum vs SGD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    X_train, X_test, y_train, y_test, mean, std = prepare_data()

    betas = [0.0, 0.5, 0.9, 0.99]
    models = {}

    for beta in betas:
        model = MomentumPerceptron(n_features=2, beta=beta, random_state=42)
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
        label = "sgd" if beta == 0.0 else f"beta={beta}"
        models[label] = model
        print(f"{label}: test_acc={model.accuracy(X_test, y_test):.4f}")

    plot_loss_curves(models)


if __name__ == "__main__":
    main()
