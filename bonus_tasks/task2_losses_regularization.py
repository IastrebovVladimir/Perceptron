import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from data import prepare_data


class ExtendedPerceptron(Perceptron):
    def __init__(self, n_features, init_mode="random", random_state=42,
                 loss_type="cross_entropy", l2_lambda=0.0):
        super().__init__(n_features, init_mode=init_mode, random_state=random_state)
        self.loss_type = loss_type
        self.l2_lambda = l2_lambda

    def compute_loss(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)

        if self.loss_type == "hinge":
            y_signed = 2 * y_true - 1
            scores = 2 * y_pred - 1
            loss = np.mean(np.maximum(0, 1 - y_signed * scores))
        else:
            eps = 1e-12
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(
                y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            )

        return loss + self.l2_lambda * np.sum(self.w ** 2)

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

                if self.loss_type == "hinge":
                    y_signed = 2 * y_batch - 1
                    scores = 2 * y_pred - 1
                    active = (1 - y_signed * scores) > 0
                    grad_factor = np.where(active, -2 * y_signed, 0.0)
                    dw = (X_batch.T @ grad_factor) / len(X_batch)
                    db = np.mean(grad_factor)
                else:
                    dw = (X_batch.T @ (y_pred - y_batch)) / len(X_batch)
                    db = np.mean(y_pred - y_batch)

                dw += 2 * self.l2_lambda * self.w

                self.w -= lr * dw
                self.b -= lr * db

            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append(self.compute_loss(y_val, self.forward(X_val)))

        return self


def plot_loss_curves(models, title):
    plt.figure(figsize=(10, 6))
    for label, model in models.items():
        epochs = np.arange(1, len(model.val_losses) + 1)
        plt.plot(epochs, model.val_losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    X_train, X_test, y_train, y_test, mean, std = prepare_data()

    ce_model = ExtendedPerceptron(n_features=2, loss_type="cross_entropy", random_state=42)
    ce_model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    hinge_model = ExtendedPerceptron(n_features=2, loss_type="hinge", random_state=42)
    hinge_model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    print("Loss comparison")
    print(f"Cross-entropy test accuracy: {ce_model.accuracy(X_test, y_test):.4f}")
    print(f"Hinge loss test accuracy:    {hinge_model.accuracy(X_test, y_test):.4f}")

    plot_loss_curves(
        {"cross_entropy": ce_model, "hinge": hinge_model},
        "Cross-Entropy vs Hinge Loss"
    )

    lambdas = [0.0, 0.001, 0.01, 0.1]
    l2_models = {}

    print("\nL2 regularization")
    for lam in lambdas:
        model = ExtendedPerceptron(n_features=2, l2_lambda=lam, random_state=42)
        model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
        l2_models[f"lambda={lam}"] = model
        print(f"lambda={lam}: |w|={np.linalg.norm(model.w):.4f}, test_acc={model.accuracy(X_test, y_test):.4f}")

    plot_loss_curves(l2_models, "L2 Regularization Effect")


if __name__ == "__main__":
    main()
