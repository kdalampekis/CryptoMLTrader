import json
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Load the updated embeddings
with open('embeddings_cls.json', 'r') as f:
    embeddings_dict = json.load(f)


# Clustering and Evaluation
num_clusters_class = 10  # Adjust based on your analysis
cluster_labels_class_kmeans = {}
ari_scores_class_kmeans = {}

# Perform Clustering at the Class Level (KMeans)
# Perform Clustering at the Class Level (KMeans) with silhouette analysis
silhouette_scores_class = []

# Perform Clustering at the Class Level (KMeans) with silhouette analysis
for class_name, sections in embeddings_dict.items():
    all_embeddings = []
    for section_emb in sections.values():
        for emb in section_emb:
            if len(emb) == len(sections[list(sections.keys())[0]][0]):  # Ensure uniformity
                all_embeddings.append(emb)
    if all_embeddings:
        class_embeddings_flat = np.vstack(all_embeddings)

        # Silhouette Analysis
        silhouette_scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(class_embeddings_flat)
            silhouette_avg = silhouette_score(class_embeddings_flat, labels)
            silhouette_scores.append(silhouette_avg)
        best_n_clusters = np.argmax(silhouette_scores) + 2  # Add 2 because range starts from 2
        silhouette_scores_class.append(silhouette_scores)

        # KMeans Clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
        labels = kmeans.fit_predict(class_embeddings_flat)
        cluster_labels_class_kmeans[class_name] = labels

        # Evaluate Agreement at the Class Level (KMeans) using Adjusted Rand Index (ARI)
        true_labels_class = [class_name] * len(labels)
        ari = adjusted_rand_score(true_labels_class, labels)
        ari_scores_class_kmeans[class_name] = ari

# Plot silhouette scores for each class
plt.figure(figsize=(10, 6))
for i, silhouette_scores in enumerate(silhouette_scores_class):
    plt.plot(range(2, 11), silhouette_scores, label=f'Class {i+1}')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Class Level Clustering')
plt.legend()
plt.show()


# Initialize dictionaries to store clustering labels and ARI scores
cluster_labels_section_kmeans = {}
ari_scores_section_kmeans = {}
num_clusters_section = 10

# Perform Clustering at the Section/Division Level (KMeans) with silhouette analysis
silhouette_scores_section = []
for class_name, sections in embeddings_dict.items():
    class_cluster_labels = {}
    class_silhouette_scores = {}
    class_ari_scores = {}
    for section_name, embeddings in sections.items():
        uniform_embeddings = [emb for emb in embeddings if len(emb) == len(embeddings[0])]
        if uniform_embeddings:
            section_embeddings_flat = np.vstack(uniform_embeddings)

            # Silhouette Analysis
            silhouette_scores = []
            for n_clusters in range(2, 11):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(section_embeddings_flat)
                silhouette_avg = silhouette_score(section_embeddings_flat, labels)
                silhouette_scores.append(silhouette_avg)
            best_n_clusters = np.argmax(silhouette_scores) + 2  # Add 2 because range starts from 2
            class_silhouette_scores[section_name] = silhouette_scores

            # KMeans Clustering with optimal number of clusters
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
            labels = kmeans.fit_predict(section_embeddings_flat)
            class_cluster_labels[section_name] = labels

            # Evaluate Agreement at the Section Level (KMeans) using Adjusted Rand Index (ARI)
            true_labels_section = [section_name] * len(labels)
            ari = adjusted_rand_score(true_labels_section, labels)
            class_ari_scores[section_name] = ari

    silhouette_scores_section.append(class_silhouette_scores)
    cluster_labels_section_kmeans[class_name] = class_cluster_labels
    ari_scores_section_kmeans[class_name] = class_ari_scores

# Plot silhouette scores for each section
plt.figure(figsize=(10, 6))
for i, class_silhouette_scores in enumerate(silhouette_scores_section):
    for section_name, silhouette_scores in class_silhouette_scores.items():
        plt.plot(range(2, 11), silhouette_scores, label=f'Class {i + 1}, Section {section_name}')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Section/Division Level Clustering')
plt.legend()
plt.show()

# Perform Hierarchical Clustering at the Class Level
num_clusters_class_hierarchical = 4  # Adjust based on your analysis
cluster_labels_class_hierarchical = {}
ari_scores_class_hierarchical = {}
# Perform Hierarchical Clustering at the Class Level
all_embeddings_hierarchical = []
for class_emb in embeddings_dict.values():
    for section_emb in class_emb.values():
        # Ensure uniformity in embeddings length within each section
        uniform_embeddings = [emb for emb in section_emb if len(emb) == len(section_emb[0])]
        if uniform_embeddings:
            # Stack embeddings vertically
            all_embeddings_hierarchical.extend(uniform_embeddings)

# Convert to 2D NumPy array
all_embeddings_hierarchical = np.vstack(all_embeddings_hierarchical)

hierarchical_clustering = AgglomerativeClustering(n_clusters=num_clusters_class_hierarchical)
labels_hierarchical = hierarchical_clustering.fit_predict(all_embeddings_hierarchical)

# Process cluster labels and ARI scores
start_index = 0
for class_name, sections in embeddings_dict.items():
    num_embeddings = sum(len(section_emb) for section_emb in sections.values() if len(section_emb) > 0)
    end_index = start_index + num_embeddings
    cluster_labels_class_hierarchical[class_name] = labels_hierarchical[start_index:end_index]

    true_labels_class = [class_name] * num_embeddings
    ari = adjusted_rand_score(true_labels_class, labels_hierarchical[start_index:end_index])
    ari_scores_class_hierarchical[class_name] = ari

    start_index = end_index

# Silhouette Method
silhouette_scores = []
for k in range(2, 11):  # Try different numbers of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(all_embeddings_hierarchical)  # Use hierarchical clustering embeddings
    silhouette_scores.append(silhouette_score(all_embeddings_hierarchical, labels))

# Elbow Method
inertia = []
for k in range(1, 11):  # Try different numbers of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(all_embeddings_hierarchical)  # Use hierarchical clustering embeddings
    inertia.append(kmeans.inertia_)

# Plot Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Analysis')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Plot Elbow Curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Reporting Results
print("Class Level Clustering Results (KMeans):")
for class_name, ari in ari_scores_class_kmeans.items():
    print(f"Class: {class_name}, Adjusted Rand Index (ARI): {ari}")

print("Section/Division Level Clustering Results (KMeans):")
for class_name, sections in ari_scores_section_kmeans.items():
    print(f"Class: {class_name}")
    for section_name, ari in sections.items():
        print(f"  Section: {section_name}, Adjusted Rand Index (ARI): {ari}")

print("\nClass Level Clustering Results (Hierarchical):")
for class_name, ari in ari_scores_class_hierarchical.items():
    print(f"Class: {class_name}, Adjusted Rand Index (ARI): {ari}")
