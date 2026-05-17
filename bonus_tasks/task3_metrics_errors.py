import numpy as np
import matplotlib.pyplot as plt

from data import prepare_data
from perceptron import Perceptron


def precision_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score_manual(y_true, y_pred):
    precision = precision_score_manual(y_true, y_pred)
    recall = recall_score_manual(y_true, y_pred)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def roc_curve_manual(y_true, y_scores):
    thresholds = np.unique(y_scores)[::-1]
    thresholds = np.concatenate(([1.1], thresholds, [-0.1]))

    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list)


def auc_manual(fpr, tpr):
    order = np.argsort(fpr)
    return np.trapezoid(tpr[order], fpr[order])


def plot_roc_curve(fpr, tpr, auc_value):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_misclassified_points(model, X, y):
    y_pred = model.predict(X).ravel()
    wrong = y_pred != y

    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=0.4, edgecolors="k")
    plt.scatter(X[wrong, 0], X[wrong, 1], facecolors="none", edgecolors="yellow", s=120, linewidths=2, label="Misclassified")
    plt.title("Misclassified test points")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    X_train, X_test, y_train, y_test, mean, std = prepare_data()

    model = Perceptron(n_features=2, random_state=42)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    y_pred = model.predict(X_test).ravel()
    y_scores = model.predict_proba(X_test).ravel()

    precision = precision_score_manual(y_test, y_pred)
    recall = recall_score_manual(y_test, y_pred)
    f1 = f1_score_manual(y_test, y_pred)
    fpr, tpr = roc_curve_manual(y_test, y_scores)
    auc_value = auc_manual(fpr, tpr)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc_value:.4f}")

    plot_roc_curve(fpr, tpr, auc_value)
    plot_misclassified_points(model, X_test, y_test)


if __name__ == "__main__":
    main()
