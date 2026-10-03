# Week 5: Decision Tree on Titanic + tree structure reproduction
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("titanic.csv")
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]].copy()
df["age"] = df["age"].fillna(df["age"].median())
df["sex"] = df["sex"].map({"male": 0, "female": 1})

X, y = df.drop(columns="survived"), df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42).fit(X_train, y_train)
pred = tree.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, target_names=["died", "survived"]))

# Tree structure (text)
print("Tree structure:\n", export_text(tree, feature_names=list(X.columns)))

# Tree structure (image)
plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=X.columns, class_names=["died", "survived"], filled=True, rounded=True)
plt.savefig("titanic_tree.png", dpi=130, bbox_inches="tight")
