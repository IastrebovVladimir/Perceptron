import numpy as np
import matplotlib.pyplot as plt

from data import prepare_data
from perceptron import Perceptron


BASE_EPOCHS = 100
BASE_LR = 0.1
BASE_BATCH_SIZE = 32
BASE_INIT_MODE = "random"
BASE_RANDOM_STATE = 42


def train_single_model(X_train, X_test, y_train, y_test,
                       lr=BASE_LR,
                       batch_size=BASE_BATCH_SIZE,
                       init_mode=BASE_INIT_MODE,
                       epochs=BASE_EPOCHS,
                       random_state=BASE_RANDOM_STATE):
    model = Perceptron(
        n_features=X_train.shape[1],
        init_mode=init_mode,
        random_state=random_state
    )

    model.fit(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size
    )

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_losses": model.train_losses,
        "val_losses": model.val_losses,
        "lr": lr,
        "batch_size": batch_size,
        "init_mode": init_mode
    }


def plot_loss_curves(results, title, label_key):
    plt.figure(figsize=(10, 6))

    for result in results:
        label = f"{label_key}={result[label_key]}"
        epochs = np.arange(1, len(result["train_losses"]) + 1)
        plt.plot(epochs, result["val_losses"], label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_results_table(results, experiment_name):
    print(f"\n{experiment_name}")
    print("-" * len(experiment_name))
    print(f"{'Parameter':<15} {'Train Accuracy':<18} {'Test Accuracy':<18}")

    for result in results:
        if "lr" in result and result["lr"] != BASE_LR:
            parameter = f"lr={result['lr']}"
        elif "batch_size" in result and result["batch_size"] != BASE_BATCH_SIZE:
            parameter = f"batch={result['batch_size']}"
        else:
            parameter = f"init={result['init_mode']}"

        print(f"{parameter:<15} {result['train_acc']:<18.4f} {result['test_acc']:<18.4f}")


def experiment_learning_rate(X_train, X_test, y_train, y_test):
    learning_rates = [0.001, 0.01, 0.5, 1.0]
    results = []

    for lr in learning_rates:
        result = train_single_model(
            X_train, X_test, y_train, y_test,
            lr=lr,
            batch_size=BASE_BATCH_SIZE,
            init_mode=BASE_INIT_MODE
        )
        results.append(result)

    plot_loss_curves(results, "Learning Rate Effect", "lr")
    print_results_table(results, "Learning Rate Experiment")
    return results


def experiment_batch_size(X_train, X_test, y_train, y_test):
    batch_sizes = [1, 16, 64, 256]
    results = []

    for batch_size in batch_sizes:
        result = train_single_model(
            X_train, X_test, y_train, y_test,
            lr=BASE_LR,
            batch_size=batch_size,
            init_mode=BASE_INIT_MODE
        )
        results.append(result)

    plot_loss_curves(results, "Batch Size Effect", "batch_size")
    print_results_table(results, "Batch Size Experiment")
    return results


def experiment_initialization(X_train, X_test, y_train, y_test):
    init_modes = ["zeros", "random", "large"]
    results = []

    for init_mode in init_modes:
        result = train_single_model(
            X_train, X_test, y_train, y_test,
            lr=BASE_LR,
            batch_size=BASE_BATCH_SIZE,
            init_mode=init_mode
        )
        results.append(result)

    plot_loss_curves(results, "Weight Initialization Effect", "init_mode")
    print_results_table(results, "Weight Initialization Experiment")
    return results


def main():
    X_train, X_test, y_train, y_test, mean, std = prepare_data()

    experiment_learning_rate(X_train, X_test, y_train, y_test)
    experiment_batch_size(X_train, X_test, y_train, y_test)
    experiment_initialization(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()