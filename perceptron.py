import numpy as np


class Perceptron:
    def __init__(self, n_features, init_mode="random", random_state=42):
        rng = np.random.default_rng(random_state)

        if init_mode == "zeros":
            self.w = np.zeros((n_features, 1))
        elif init_mode == "large":
            self.w = rng.normal(0, 10, size=(n_features, 1))
        else:
            self.w = rng.normal(0, 0.01, size=(n_features, 1))

        self.b = 0.0
        self.train_losses = []
        self.val_losses = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        eps = 1e-12
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def accuracy(self, X, y):
        y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        n_samples = X_train.shape[0]

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

                self.w -= lr * dw
                self.b -= lr * db

            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)

            train_loss = self.compute_loss(y_train, train_pred)
            val_loss = self.compute_loss(y_val, val_pred)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

        return self