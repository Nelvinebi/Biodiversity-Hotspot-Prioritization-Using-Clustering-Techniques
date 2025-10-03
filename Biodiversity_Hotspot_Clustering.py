import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Generate Synthetic Biodiversity Data
np.random.seed(42)
n_points = 200  # ≥150 data points

# Spatial coordinates
x_coord = np.random.uniform(0, 100, n_points)
y_coord = np.random.uniform(0, 100, n_points)

# Ecological features
species_richness = np.random.normal(loc=120, scale=30, size=n_points)     # number of species
endemic_species = np.random.normal(loc=30, scale=10, size=n_points)       # number of endemic species
threatened_species = np.random.normal(loc=15, scale=5, size=n_points)     # number of threatened species
habitat_quality = np.random.uniform(0.4, 1.0, n_points)                    # 0 (poor) to 1 (pristine)

# Clip any negative values due to noise
species_richness = np.clip(species_richness, 10, None)
endemic_species = np.clip(endemic_species, 0, None)
threatened_species = np.clip(threatened_species, 0, None)

# Create DataFrame
df = pd.DataFrame({
    'x_coord': x_coord,
    'y_coord': y_coord,
    'species_richness': species_richness,
    'endemic_species': endemic_species,
    'threatened_species': threatened_species,
    'habitat_quality': habitat_quality
})

# 2. Scale features before clustering
features = ['species_richness', 'endemic_species', 'threatened_species', 'habitat_quality']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# 3. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['hotspot_cluster'] = kmeans.fit_predict(scaled_features)

# Optional: Assign priority labels (e.g., 0 = Low, 1 = Medium, 2 = High)
priority_labels = {0: 'Medium', 1: 'Low', 2: 'High'}
df['priority'] = df['hotspot_cluster'].map(lambda x: priority_labels.get(x, 'Unknown'))

# 4. PCA for 2D visualization (optional)
pca = PCA(n_components=2)
df[['pca1', 'pca2']] = pca.fit_transform(scaled_features)

# 5. Output Summary
print(df.head())

# 6. Visualization: Biodiversity Hotspot Clusters (PCA view)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='priority', palette='Set1', s=60)
plt.title('Biodiversity Hotspot Prioritization (PCA View)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Priority Level')
plt.tight_layout()
plt.show()

# 7. Spatial Map
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='x_coord', y='y_coord', hue='priority', palette='Set2', s=60)
plt.title('Spatial Distribution of Biodiversity Priority Areas')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.tight_layout()
plt.show()
