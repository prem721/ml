# Week 4: EM algorithm (Gaussian Mixture Model) on Iris, compared with K-Means
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix

df = pd.read_csv("iris.csv")
X, y = df.drop(columns="target").values, df["target"].values

# GaussianMixture is fitted with the EM algorithm (E-step: responsibilities, M-step: update params)
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, max_iter=100).fit(X)
em_labels = gmm.predict(X)
km_labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)

print("EM converged:", gmm.converged_, "in", gmm.n_iter_, "iterations")
print("Mixing weights:", gmm.weights_.round(3))
print("Means:\n", gmm.means_.round(3))
print("\nAdjusted Rand Index vs true labels -> EM: %.3f | KMeans: %.3f"
      % (adjusted_rand_score(y, em_labels), adjusted_rand_score(y, km_labels)))
print("\nConfusion (true vs EM cluster):\n", confusion_matrix(y, em_labels))

plt.figure(figsize=(10, 4))
for i, (lab, t) in enumerate([(em_labels, "EM (GMM)"), (km_labels, "K-Means")]):
    plt.subplot(1, 2, i + 1)
    plt.scatter(X[:, 2], X[:, 3], c=lab, cmap="viridis")
    plt.title(t); plt.xlabel("petal length"); plt.ylabel("petal width")
plt.tight_layout(); plt.savefig("em_vs_kmeans.png", dpi=120)
