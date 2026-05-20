import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Mall Customer Segmentation")
st.markdown("**K-Means Clustering | SkillCraft Technology - Task 02**")

uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = ['customer_id', 'gender', 'age', 'annual_income', 'spending_score']

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")

    k = st.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=5)

    X = df[['annual_income', 'spending_score']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    df['Cluster'] = kmeans.labels_

    score = silhouette_score(X_scaled, kmeans.labels_)
    st.success(f"✅ Silhouette Score: {score:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Elbow Method")
        inertia = []
        for i in range(2, 11):
            km = KMeans(n_clusters=i, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)
        fig1, ax1 = plt.subplots()
        ax1.plot(range(2, 11), inertia, 'bo-')
        ax1.set_xlabel('K')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        st.pyplot(fig1)

    with col2:
        st.subheader("🎯 Cluster Plot")
        colors = ['#e74c3c','#2ecc71','#3498db','#f39c12','#9b59b6',
                  '#1abc9c','#e67e22','#e91e63','#607d8b','#795548']
        fig2, ax2 = plt.subplots()
        for i in range(k):
            c = df[df['Cluster'] == i]
            ax2.scatter(c['annual_income'], c['spending_score'],
                       s=80, c=colors[i % len(colors)], label=f'Cluster {i+1}',
                       edgecolors='black', linewidth=0.4)
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        ax2.scatter(centroids[:, 0], centroids[:, 1],
                   s=300, c='yellow', marker='*', edgecolors='black', label='Centroids')
        ax2.set_xlabel('Annual Income (k$)')
        ax2.set_ylabel('Spending Score')
        ax2.set_title(f'K-Means Clusters (K={k})')
        ax2.legend(fontsize=7)
        st.pyplot(fig2)

    st.subheader("📋 Cluster Summary")
    summary = df.groupby('Cluster')[['age','annual_income','spending_score']].mean().round(2)
    st.dataframe(summary)

    st.download_button("⬇️ Download Results CSV",
                       df.to_csv(index=False),
                       file_name="customer_segments.csv")
else:
    st.info("👆 Please upload the Mall_Customers.csv file to get started.")
    st.markdown("Download dataset from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)")