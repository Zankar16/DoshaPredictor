import csv
import math
import random
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Load dataset from CSV file
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read header
        for row in reader:
            dataset.append(row)
    return headers, dataset

# Convert categorical features to numerical encoding
def encode_categorical_data(headers, dataset):
    categorical_columns = headers[:-1]  # Exclude the target column (Dosha)
    numerical_data = []
    label_encodings = defaultdict(dict)
    
    # One-hot encoding for categorical features
    one_hot_encoded_data = []
    one_hot_encodings = defaultdict(dict)

    # Create encoding maps for categorical values
    for col_idx in range(len(categorical_columns)):
        unique_values = sorted(set(row[col_idx] for row in dataset))  # Ensure consistent ordering
        encoding = {val: idx for idx, val in enumerate(unique_values)}
        label_encodings[categorical_columns[col_idx]] = encoding

    # Encode dataset
    for row in dataset:
        numerical_row = []
        
        # One-hot encode categorical columns
        for i, col in enumerate(categorical_columns):
            encoded_vector = [0] * len(label_encodings[col])
            encoded_vector[label_encodings[col][row[i]]] = 1
            numerical_row.extend(encoded_vector)

        # Encode dosha label
        dosha_label = row[-1]  # Keep Dosha as a string
        numerical_row.append(dosha_label)
        
        numerical_data.append(numerical_row)

    return numerical_data, label_encodings, one_hot_encodings, ["vata", "pitta", "kapha"]

# Compute Euclidean distance
def euclidean_distance(point1, point2):
    return math.sqrt(sum((point1[d] - point2[d]) ** 2 for d in range(len(point1))))

# K-Means Clustering
def k_means_clustering(data, num_clusters, max_iterations=100):
    data_points = np.array([row[:-1] for row in data])  # Exclude labels for clustering

    if len(data_points) == 0:
        raise ValueError("⚠️ No training data available for K-Means!")

    # Initialize random centroids
    random_indices = np.random.choice(len(data_points), num_clusters, replace=False)
    centroids = data_points[random_indices]

    for _ in range(max_iterations):
        clusters = [[] for _ in range(num_clusters)]
        for point in data_points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)

        # Compute new centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                new_centroids.append(random.choice(data_points))

        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids

    return centroids

# Compute probability of test point belonging to each cluster
def compute_probabilities(test_point, centroids, alpha=1.0):
    probabilities = []
    total_weight = 0

    for centroid in centroids:
        weight = math.exp(-alpha * euclidean_distance(test_point, centroid))
        probabilities.append(weight)
        total_weight += weight

    # Normalize probabilities
    probabilities = [p / total_weight for p in probabilities]
    return probabilities

# Train Random Forest Classifier on cluster probabilities
def train_random_forest(train_probs, train_labels):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(train_probs, train_labels)
    return rf_model

# Predict dominant clusters (convert indices to Dosha names)
def predict_dominant_clusters(probabilities, dosha_names, threshold=0.3):
    dominant_indices = [idx for idx, prob in enumerate(probabilities) if prob >= threshold]
    if not dominant_indices:
        dominant_indices = [np.argmax(probabilities)]  # Ensure at least one cluster is selected
    return "+".join([dosha_names[idx] for idx in dominant_indices])  # Convert to Dosha name string

# 🚀 **Main Execution**
filename = "dataset.csv"  # Replace with your dataset file
num_clusters = 3  # Vata, Pitta, Kapha

# Step 1: Load dataset
headers, dataset = load_csv(filename)

# Step 2: Convert to numerical format
numerical_data, label_encodings, dosha_names = encode_categorical_data(headers, dataset)

# Convert Dosha to string before splitting
for row in numerical_data:
    row[-1] = str(row[-1])  # Preserve multi-label dosha

# **5-Fold Cross Validation**
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(numerical_data)):
    print(f"\n🔹 Fold {fold + 1}/{5}:")

    train_data = [numerical_data[i] for i in train_idx]
    test_data = [numerical_data[i] for i in test_idx]

    # Filter **only single-label** dosha training data
    single_label_train_data = [row for row in train_data if "," not in row[-1]]

    # Train K-Means model
    centroids = k_means_clustering(single_label_train_data, num_clusters)

    # Step 6: Compute probabilities for training data
    train_probs = []
    train_labels = []
    for train_instance in single_label_train_data:
        train_point = train_instance[:-1]
        train_probs.append(compute_probabilities(train_point, centroids))
        train_labels.append(train_instance[-1])

    # Train Random Forest model
    rf_model = train_random_forest(train_probs, train_labels)

    # Compute probabilities for test data and predict dominant clusters
    test_probs = []
    actual_labels = []
    predicted_clusters = []

    for test_instance in test_data:
        test_point = test_instance[:-1]
        test_prob = compute_probabilities(test_point, centroids)
        test_probs.append(test_prob)
        actual_labels.append(test_instance[-1])

        # Get dominant clusters as string (multi-labels joined with "+")
        predicted_cluster = predict_dominant_clusters(test_prob, dosha_names, threshold=0.3)
        predicted_clusters.append(predicted_cluster)

    # Calculate accuracy (if predicted dosha matches one of the actual doshas)
    correct_predictions = sum(1 for i in range(len(test_data)) if any(dosha in predicted_clusters[i] for dosha in actual_labels[i].split("+")))
    accuracy = (correct_predictions / len(test_data)) * 100
    all_accuracies.append(accuracy)

    print(f"✅ Fold {fold+1} Accuracy: {accuracy:.2f}%\n")
    print("-" * 40)

# Compute overall average accuracy across folds
avg_accuracy = sum(all_accuracies) / len(all_accuracies)
print(f"\n🎯 **Final Average Accuracy Across 5 Folds: {avg_accuracy:.2f}%**")
