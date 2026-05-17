import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def standardize_train_test(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std[std == 0] = 1.0

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_test_std, mean, std


def prepare_data(
    n_samples=500,
    test_size=0.3,
    random_state=42
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=random_state
    )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    X_train, X_test, mean, std = standardize_train_test(X_train, X_test)

    return X_train, X_test, y_train, y_test, mean, std