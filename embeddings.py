from transformers import BertModel, BertTokenizer
import json
import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering

# Download pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the data from the JSON file
with open('roget_words.json', 'r') as f:
    data = json.load(f)

# Dictionary to store embeddings
embeddings_dict = {}

# Process each class and section
for class_name, sections in data.items():
    class_embeddings = {}
    max_sequence_length = 0  # Track the maximum sequence length
    for section_name, word_lists in sections.items():
        section_embeddings = []
        for word_list in word_lists:
            # Tokenize words
            tokens = tokenizer(word_list, padding=True, truncation=True, return_tensors='pt')
            # Get BERT embeddings
            with torch.no_grad():
                outputs = model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Aggregate embeddings (mean pooling)
            section_embeddings.append(embeddings.tolist())  # Convert embeddings to list and append
        # Update max_sequence_length
        max_sequence_length = max(max_sequence_length, len(section_embeddings))
        # Store section embeddings in the dictionary
        class_embeddings[section_name] = section_embeddings
    # Pad or truncate section_embeddings to max_sequence_length
    for section_name, embeddings_list in class_embeddings.items():
        while len(embeddings_list) < max_sequence_length:
            embeddings_list.append([0] * len(embeddings_list[0]))  # Pad with zeros
        class_embeddings[section_name] = embeddings_list[:max_sequence_length]  # Truncate if necessary
    # Store class embeddings in the dictionary
    embeddings_dict[class_name] = class_embeddings

print("Data processed successfully!")

# Save embeddings to a JSON file
with open('embeddings.json', 'w') as f:
    json.dump(embeddings_dict, f)

print("Embeddings saved successfully!")

# Step 1: Perform Clustering at the Class Level (KMeans)
num_clusters_class = 6  # Number of clusters for each class
cluster_labels_class_kmeans = {}
for class_name, sections in embeddings_dict.items():
    class_embeddings = [emb for section_emb in sections.values() for emb_list in section_emb for emb in emb_list]
    kmeans = KMeans(n_clusters=num_clusters_class, random_state=42)
    labels = kmeans.fit_predict(class_embeddings)
    cluster_labels_class_kmeans[class_name] = labels

print("Class Level Clustering (KMeans) completed successfully!")

# Step 2: Perform Clustering at the Section/Division Level (KMeans)
num_clusters_section = 6  # Number of clusters for each section
cluster_labels_section_kmeans = {}
for class_name, sections in embeddings_dict.items():
    class_cluster_labels = {}
    for section_name, embeddings in sections.items():
        embeddings_concatenated = [emb for emb_list in embeddings for emb in emb_list]
        kmeans = KMeans(n_clusters=num_clusters_section, random_state=42)
        labels = kmeans.fit_predict(embeddings_concatenated)
        class_cluster_labels[section_name] = labels
    cluster_labels_section_kmeans[class_name] = class_cluster_labels

print("Section/Division Level Clustering (KMeans) completed successfully!")

# Step 3: Perform Hierarchical Clustering at the Class Level
num_clusters_class_hierarchical = 6  # Number of clusters for each class (hierarchical)
cluster_labels_class_hierarchical = {}
all_embeddings_hierarchical = []
for class_name, sections in embeddings_dict.items():
    class_embeddings = [emb for section_emb in sections.values() for emb_list in section_emb for emb in emb_list]
    all_embeddings_hierarchical.extend(class_embeddings)
hierarchical_clustering = AgglomerativeClustering(n_clusters=num_clusters_class_hierarchical)
labels = hierarchical_clustering.fit_predict(all_embeddings_hierarchical)
start_index = 0
for class_name, sections in embeddings_dict.items():
    end_index = start_index + len(sections) * len(next(iter(sections.values())))
    cluster_labels_class_hierarchical[class_name] = labels[start_index:end_index]
    start_index = end_index

print("Class Level Clustering (Hierarchical) completed successfully!")

# Step 4: Evaluate Agreement at the Class Level (KMeans)
# Use Adjusted Rand Index (ARI) or Normalized Mutual Information (NMI)
ari_scores_class_kmeans = {}
for class_name, labels in cluster_labels_class_kmeans.items():
    true_labels_class = [class_name] * len(labels)  # Assuming each word belongs to its respective class
    ari = adjusted_rand_score(true_labels_class, labels)
    ari_scores_class_kmeans[class_name] = ari

# Step 5: Evaluate Agreement at the Section/Division Level (KMeans)
# Use Adjusted Rand Index (ARI) or Normalized Mutual Information (NMI)
ari_scores_section_kmeans = {}
for class_name, sections in cluster_labels_section_kmeans.items():
    class_ari_scores = {}
    for section_name, labels in sections.items():
        true_labels_section = [section_name] * len(labels)  # Assuming each word belongs to its respective section
        ari = adjusted_rand_score(true_labels_section, labels)
        class_ari_scores[section_name] = ari
    ari_scores_section_kmeans[class_name] = class_ari_scores

# Step 6: Evaluate Agreement at the Class Level (Hierarchical)
# Use Adjusted Rand Index (ARI) or Normalized Mutual Information (NMI)
ari_scores_class_hierarchical = {}
for class_name, labels in cluster_labels_class_hierarchical.items():
    true_labels_class = [class_name] * len(labels)  # Assuming each word belongs to its respective class
    ari = adjusted_rand_score(true_labels_class, labels)
    ari_scores_class_hierarchical[class_name] = ari

# Step 7: Report Results
print("Class Level Clustering Results (KMeans):")
for class_name, ari in ari_scores_class_kmeans.items():
    print(f"Class: {class_name}")
    print("  Adjusted Rand Index (ARI):", ari)
    print()

print("Section/Division Level Clustering Results (KMeans):")
for class_name, sections in ari_scores_section_kmeans.items():
    print(f"Class: {class_name}")
    for section_name, ari in sections.items():
        print(f"  Section: {section_name}")
        print("    Adjusted Rand Index (ARI):", ari)
    print()

print("Class Level Clustering Results (Hierarchical):")
for class_name, ari in ari_scores_class_hierarchical.items():
    print(f"Class: {class_name}")
    print("  Adjusted Rand Index (ARI):", ari)
    print()
