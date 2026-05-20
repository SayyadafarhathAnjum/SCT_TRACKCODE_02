---

### 📄 File 3: `kmeans_clustering.py` ⭐ (Main File)

```python
# ============================================================
# K-Means Clustering - Mall Customer Segmentation
# SkillCraft Technology - Task 02
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Load Dataset
# ============================================================
# Download from: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
# Save as 'Mall_Customers.csv' in the same folder

df = pd.read_csv('Mall_Customers.csv')

print("=" * 50)
print("MALL CUSTOMER DATASET - OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# STEP 2: Rename Columns for Ease
# ============================================================
df.rename(columns={
    'CustomerID': 'customer_id',
    'Genre': 'gender',
    'Age': 'age',
    'Annual Income (k$)': 'annual_income',
    'Spending Score (1-100)': 'spending_score'
}, inplace=True)

# ============================================================
# STEP 3: Exploratory Data Analysis (EDA)
# ============================================================
plt.figure(figsize=(15, 5))

# Age Distribution
plt.subplot(1, 3, 1)
sns.histplot(df['age'], kde=True, color='steelblue')
plt.title('Age Distribution')
plt.xlabel('Age')

# Annual Income Distribution
plt.subplot(1, 3, 2)
sns.histplot(df['annual_income'], kde=True, color='green')
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (k$)')

# Spending Score Distribution
plt.subplot(1, 3, 3)
sns.histplot(df['spending_score'], kde=True, color='orange')
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score (1-100)')

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150)
plt.show()
print("\n[Saved] eda_distributions.png")

# Gender Count
plt.figure(figsize=(5, 4))
sns.countplot(x='gender', data=df, palette='Set2')
plt.title('Gender Count')
plt.savefig('gender_count.png', dpi=150)
plt.show()
print("[Saved] gender_count.png")

# ============================================================
# STEP 4: Feature Selection & Scaling
# ============================================================
# Using Annual Income and Spending Score (most common approach)
X = df[['annual_income', 'spending_score']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# STEP 5: Find Optimal K using Elbow Method
# ============================================================
print("\n" + "=" * 50)
print("FINDING OPTIMAL NUMBER OF CLUSTERS (K)")
print("=" * 50)

inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    score = silhouette_score(X_scaled, km.labels_)
    silhouette_scores.append(score)
    print(f"K={k} | Inertia: {km.inertia_:.2f} | Silhouette Score: {score:.4f}")

# Elbow Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, 'bo-', markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method - Optimal K')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'rs-', markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.grid(True)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150)
plt.show()
print("\n[Saved] elbow_silhouette.png")

# ============================================================
# STEP 6: Train Final K-Means Model (K=5)
# ============================================================
OPTIMAL_K = 5
print(f"\n{'='*50}")
print(f"TRAINING K-MEANS MODEL WITH K = {OPTIMAL_K}")
print("=" * 50)

kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled)

df['cluster'] = kmeans.labels_

print(f"\nCluster Labels Assigned!")
print(f"Inertia: {kmeans.inertia_:.4f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_):.4f}")

# ============================================================
# STEP 7: 2D Cluster Visualization
# ============================================================
colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
cluster_names = [
    'Cluster 1: Low Income, Low Spend',
    'Cluster 2: Low Income, High Spend',
    'Cluster 3: Mid Income, Mid Spend',
    'Cluster 4: High Income, Low Spend',
    'Cluster 5: High Income, High Spend'
]

plt.figure(figsize=(10, 7))

for i in range(OPTIMAL_K):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(
        cluster_data['annual_income'],
        cluster_data['spending_score'],
        s=100, c=colors[i],
        label=f'Cluster {i+1}',
        edgecolors='black', linewidth=0.5, alpha=0.8
    )

# Plot centroids (inverse transform for original scale)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    s=300, c='yellow',
    marker='*', edgecolors='black',
    linewidth=1.5, label='Centroids', zorder=5
)

plt.title('Mall Customer Segmentation - K-Means Clustering (K=5)', fontsize=14, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clusters_2d.png', dpi=150)
plt.show()
print("\n[Saved] kmeans_clusters_2d.png")

# ============================================================
# STEP 8: 3D Visualization (Age + Income + Spending Score)
# ============================================================
X3D = df[['age', 'annual_income', 'spending_score']].values
X3D_scaled = StandardScaler().fit_transform(X3D)

kmeans3d = KMeans(n_clusters=OPTIMAL_K, init='k-means++', random_state=42, n_init=10)
kmeans3d.fit(X3D_scaled)
df['cluster_3d'] = kmeans3d.labels_

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(OPTIMAL_K):
    cluster_data = df[df['cluster_3d'] == i]
    ax.scatter(
        cluster_data['age'],
        cluster_data['annual_income'],
        cluster_data['spending_score'],
        s=60, c=colors[i],
        label=f'Cluster {i+1}',
        edgecolors='black', linewidth=0.3, alpha=0.8
    )

ax.set_xlabel('Age', fontsize=10)
ax.set_ylabel('Annual Income (k$)', fontsize=10)
ax.set_zlabel('Spending Score', fontsize=10)
ax.set_title('3D Customer Segmentation\n(Age + Income + Spending Score)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('kmeans_clusters_3d.png', dpi=150)
plt.show()
print("[Saved] kmeans_clusters_3d.png")

# ============================================================
# STEP 9: Cluster Analysis & Summary
# ============================================================
print("\n" + "=" * 50)
print("CLUSTER SUMMARY STATISTICS")
print("=" * 50)

cluster_summary = df.groupby('cluster')[['age', 'annual_income', 'spending_score']].agg(['mean', 'count'])
print(cluster_summary)

print("\nCustomer Count per Cluster:")
print(df['cluster'].value_counts().sort_index())

# Heatmap of cluster means
cluster_means = df.groupby('cluster')[['age', 'annual_income', 'spending_score']].mean()
plt.figure(figsize=(8, 5))
sns.heatmap(cluster_means, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': 'Mean Value'})
plt.title('Cluster Feature Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=150)
plt.show()
print("[Saved] cluster_heatmap.png")

# ============================================================
# STEP 10: Save Results
# ============================================================
df.to_csv('customer_segments.csv', index=False)
print("\n[Saved] customer_segments.csv - Customers with cluster labels")

print("\n" + "=" * 50)
print("✅ K-MEANS CLUSTERING COMPLETE!")
print("=" * 50)
print(f"  Optimal Clusters : {OPTIMAL_K}")
print(f"  Total Customers  : {len(df)}")
print(f"  Features Used    : Annual Income, Spending Score")
print(f"  Model            : K-Means++ (scikit-learn)")
print("=" * 50)