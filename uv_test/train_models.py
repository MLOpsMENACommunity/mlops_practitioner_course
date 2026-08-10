"""Train and compare multiple sklearn models on synthetic data."""

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "svm": SVC(kernel="rbf", random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
}


def main():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        print(f"{name:20s}  accuracy={results[name]['accuracy']:.4f}  f1={results[name]['f1']:.4f}")

    best = max(results, key=lambda name: results[name]["f1"])
    print(f"\nBest model: {best} (f1={results[best]['f1']:.4f})")


if __name__ == "__main__":
    main()
