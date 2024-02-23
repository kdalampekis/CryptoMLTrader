import json
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt

# Load the updated embeddings
with open('embeddings_cls.json', 'r') as f:
    embeddings_dict = json.load(f)


def standardize_embeddings(embeddings_list, max_len=512):
    """Standardize the length of embeddings by padding or truncating."""
    standardized = []
    for emb in embeddings_list:
        if len(emb) > max_len:
            standardized.append(emb[:max_len])  # Truncate
        else:
            standardized.append(emb + [0] * (max_len - len(emb)))  # Pad
    return np.array(standardized)


# Clustering and Evaluation at the Class Level with KMeans and Silhouette Analysis
for class_name, sections in embeddings_dict.items():
    all_embeddings = []
    for section_emb in sections.values():
        for emb in section_emb:
            all_embeddings.extend(emb)
    all_embeddings = standardize_embeddings(all_embeddings)

    # Dynamic range for the number of clusters based on data size
    max_clusters = min(len(all_embeddings) // 5, 50)  # Limit to a maximum of 50 clusters
    range_n_clusters = range(2, max_clusters + 1)

    silhouette_scores = []
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(all_embeddings)
        silhouette_avg = silhouette_score(all_embeddings, labels)
        silhouette_scores.append(silhouette_avg)

    best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    labels = kmeans.fit_predict(all_embeddings)

    # Evaluation using Adjusted Rand Index (ARI)
    true_labels = [class_name] * len(labels)
    ari_score = adjusted_rand_score(true_labels, labels)
    print(f"Class: {class_name}, Best Number of Clusters: {best_n_clusters}, ARI: {ari_score}")

# Hierarchical Clustering at the Class Level with Different Linkage Criteria
linkage_criteria = ['ward', 'average', 'complete']
for linkage in linkage_criteria:
    hierarchical_clustering = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=linkage)
    labels = hierarchical_clustering.fit_predict(all_embeddings)
    ari_score = adjusted_rand_score(true_labels, labels)
    print(f"Linkage: {linkage}, ARI: {ari_score}")


# Function to plot Silhouette Scores
def plot_silhouette_scores(silhouette_scores, range_n_clusters, title='Silhouette Analysis'):
    plt.figure(figsize=(10, 6))
    plt.plot(list(range_n_clusters), silhouette_scores, marker='o')
    plt.title(title)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()



