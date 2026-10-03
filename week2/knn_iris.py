# Week 2: KNN on Iris (loaded from scikit-learn)
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

# try several k values
for k in [1, 3, 5, 7, 9]:
    acc = accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test))
    print(f"k={k}: accuracy={acc:.3f}")

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
pred = knn.predict(X_test)
print("\nConfusion matrix (k=5):\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=iris.target_names))

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
print("Prediction for", sample[0], "->", iris.target_names[knn.predict(sample)[0]])
