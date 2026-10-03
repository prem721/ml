# Week 3: K-Means on Tips dataset
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("tips.csv")
X = StandardScaler().fit_transform(df[["total_bill", "tip"]])

# elbow + silhouette to choose k
inertia, sil = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.2f}, silhouette={sil[-1]:.3f}")

km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
df["cluster"] = km.labels_
print("\nCluster sizes:\n", df["cluster"].value_counts().sort_index())
print("\nCluster means:\n", df.groupby("cluster")[["total_bill", "tip"]].mean().round(2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(2, 9), inertia, "o-"); ax[0].set_title("Elbow"); ax[0].set_xlabel("k")
ax[1].scatter(df.total_bill, df.tip, c=df.cluster, cmap="viridis")
ax[1].set_title("K-Means (k=3)"); ax[1].set_xlabel("total_bill"); ax[1].set_ylabel("tip")
plt.tight_layout(); plt.savefig("kmeans_tips.png", dpi=120)
