import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# Load embeddings
with open('embeddings_cls.json', 'r') as f:
    embeddings_dict = json.load(f)


# Flatten the embeddings dictionary and collect class and section labels
embeddings = []
class_labels = []
section_labels = []

expected_dim = 1024

# Iterate through the dictionary to collect embeddings and their labels
for class_name, sections in embeddings_dict.items():
    for section_name, section_embeddings in sections.items():
        for embedding in section_embeddings:
            # Assuming each 'embedding' is a list of floats representing the embedding vector
            embeddings.append(embedding)  # Add the embedding to the list
            class_labels.append(class_name)  # Add the corresponding class label
            section_labels.append(section_name)  # Add the corresponding section label


# Convert lists to NumPy arrays
X = np.array([np.mean(emb, axis=0) if len(emb) > 0 else np.zeros(expected_dim) for emb in embeddings])
y_class = np.array(class_labels)  # Convert the list of class labels to a NumPy array
y_section = np.array(section_labels)  # Convert the list of section labels to a NumPy array


# Encode class and section labels
class_le = LabelEncoder()
section_le = LabelEncoder()
y_class_encoded = class_le.fit_transform(y_class)
y_section_encoded = section_le.fit_transform(y_section)

# Splitting the dataset
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded)
X_train_section, X_test_section, y_train_section, y_test_section = train_test_split(X, y_section_encoded, test_size=0.2, random_state=42, stratify=y_section_encoded)

# Function to train and evaluate a model
def train_evaluate_model(X_train, X_test, y_train, y_test, target_names):
    # Compute class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # Model Training with Hyperparameter Tuning
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(RandomForestClassifier(class_weight=class_weights_dict), param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train, y_train)

    # Best model evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Best Model Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Train and evaluate the models
print("Model for Class Prediction:")
train_evaluate_model(X_train_class, X_test_class, y_train_class, y_test_class, class_le.classes_)

print("\nModel for Section/Division Prediction:")
train_evaluate_model(X_train_section, X_test_section, y_train_section, y_test_section, section_le.classes_)
