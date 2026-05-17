import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron


def standardize_train_test(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def train_test_split_manual(X, y, test_size=0.3, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def add_label_noise(y, noise_p=0.0, random_state=42):
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    n_flip = int(noise_p * len(y))
    if n_flip > 0:
        indices = rng.choice(len(y), size=n_flip, replace=False)
        y_noisy[indices] = 1 - y_noisy[indices]
    return y_noisy


def generate_gaussian_data(n_samples=500,
                           mean0=(-2, -2),
                           mean1=(2, 2),
                           cov=((1, 0), (0, 1)),
                           noise_p=0.0,
                           random_state=42):
    rng = np.random.default_rng(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0
    X0 = rng.multivariate_normal(mean0, cov, size=n0)
    X1 = rng.multivariate_normal(mean1, cov, size=n1)
    y0 = np.zeros(n0, dtype=int)
    y1 = np.ones(n1, dtype=int)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    y = add_label_noise(y, noise_p=noise_p, random_state=random_state)
    return X, y


def generate_xor_data(n_samples=500, noise_p=0.0, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-2, 2, size=(n_samples, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    y = add_label_noise(y, noise_p=noise_p, random_state=random_state)
    return X, y


def generate_circle_data(n_samples=500, radius=1.5, noise_p=0.0, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-2.5, 2.5, size=(n_samples, 2))
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = (r > radius).astype(int)
    y = add_label_noise(y, noise_p=noise_p, random_state=random_state)
    return X, y


def plot_boundary(model, X, y, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", alpha=0.7)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_values = np.linspace(x_min, x_max, 200)
    if abs(model.w[1, 0]) > 1e-12:
        y_values = -(model.w[0, 0] * x_values + model.b) / model.w[1, 0]
        plt.plot(x_values, y_values, color="black", label="Decision boundary")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_experiment(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
    X_train, X_test = standardize_train_test(X_train, X_test)

    model = Perceptron(n_features=2, random_state=42)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)

    print(f"\n{title}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    plot_boundary(model, X_test, y_test, title)


def main():
    X1, y1 = generate_gaussian_data(noise_p=0.0)
    run_experiment(X1, y1, "Linear separable Gaussian data")

    X2, y2 = generate_xor_data(noise_p=0.0)
    run_experiment(X2, y2, "XOR data")

    X3, y3 = generate_circle_data(noise_p=0.0)
    run_experiment(X3, y3, "Circle data")


if __name__ == "__main__":
    main()
