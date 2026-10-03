# Week 6: Locally Weighted Regression on Tips (total_bill -> tip)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("tips.csv")
x, y = df["total_bill"].values, df["tip"].values
X = np.c_[np.ones_like(x), x]  # add intercept


def lwr_predict(x0, X, y, tau):
    # Gaussian kernel weights around the query point, then weighted normal equation
    w = np.exp(-((X[:, 1] - x0) ** 2) / (2 * tau ** 2))
    W = np.diag(w)
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    return np.array([1, x0]) @ theta


grid = np.linspace(x.min(), x.max(), 200)
plt.scatter(x, y, s=12, alpha=0.5, label="data")
for tau in [1, 3, 10]:
    pred = [lwr_predict(g, X, y, tau) for g in grid]
    mse = np.mean([(lwr_predict(xi, X, y, tau) - yi) ** 2 for xi, yi in zip(x, y)])
    print(f"tau={tau}: train MSE={mse:.4f}")
    plt.plot(grid, pred, label=f"tau={tau}")

# plain linear regression for comparison
b = np.polyfit(x, y, 1)
plt.plot(grid, np.polyval(b, grid), "k--", label="linear")
print("Prediction for bill=30 (tau=3): %.3f" % lwr_predict(30, X, y, 3))
plt.xlabel("total_bill"); plt.ylabel("tip"); plt.legend()
plt.title("Locally Weighted Regression"); plt.savefig("lwr_tips.png", dpi=120)
