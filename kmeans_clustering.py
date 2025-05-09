import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("income.csv")

# Initial Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.title('Original Income Data')
plt.show()

# Apply KMeans before scaling
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
df['Cluster'] = y_predicted

# Plotting clusters (before scaling)
df1 = df[df.Cluster == 0]
df2 = df[df.Cluster == 1]
df3 = df[df.Cluster == 2]

plt.figure(figsize=(8, 5))
plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            color='purple', marker='*', label='Centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.title('KMeans Clustering (Before Scaling)')
plt.show()

# MinMax Scaling
scaler = MinMaxScaler()
df[['Age', 'Income($)']] = scaler.fit_transform(df[['Age', 'Income($)']])

# KMeans after scaling
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
df['Cluster'] = y_predicted

# Plotting clusters (after scaling)
df1 = df[df.Cluster == 0]
df2 = df[df.Cluster == 1]
df3 = df[df.Cluster == 2]

plt.figure(figsize=(8, 5))
plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            color='purple', marker='*', label='Centroid')
plt.xlabel('Age (scaled)')
plt.ylabel('Income ($) (scaled)')
plt.legend()
plt.title('KMeans Clustering (After Scaling)')
plt.show()

# Elbow Method to find optimal K
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_rng, sse, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Error (SSE)')
plt.title('Elbow Method to Determine Optimal K')
plt.grid(True)
plt.show()
