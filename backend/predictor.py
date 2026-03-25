import csv
import math
import random
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import json
import sys
import os

class DoshaPredictor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.headers, self.dataset = self.load_csv(dataset_path)
        self.numerical_data, self.label_encodings, self.dosha_names = self.encode_categorical_data(self.headers, self.dataset)
        self.num_clusters = 3
        self.centroids = None
        self.rf_model = None
        self.train()

    def load_csv(self, filename):
        dataset = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                dataset.append(row)
        return headers, dataset

    def encode_categorical_data(self, headers, dataset):
        categorical_columns = headers[:-1]
        numerical_data = []
        label_encodings = defaultdict(dict)
        
        for col_idx in range(len(categorical_columns)):
            unique_values = sorted(set(row[col_idx] for row in dataset))
            encoding = {val: idx for idx, val in enumerate(unique_values)}
            label_encodings[categorical_columns[col_idx]] = encoding

        for row in dataset:
            numerical_row = []
            for i, col in enumerate(categorical_columns):
                encoded_vector = [0] * len(label_encodings[col])
                encoded_vector[label_encodings[col][row[i]]] = 1
                numerical_row.extend(encoded_vector)
            
            dosha_label = str(row[-1])
            numerical_row.append(dosha_label)
            numerical_data.append(numerical_row)

        return numerical_data, label_encodings, ["vata", "pitta", "kapha"]

    def euclidean_distance(self, point1, point2):
        return math.sqrt(sum((point1[d] - point2[d]) ** 2 for d in range(len(point1))))

    def k_means_clustering(self, data, num_clusters, max_iterations=100):
        data_points = np.array([row[:-1] for row in data])
        if len(data_points) == 0:
            raise ValueError("No training data available")

        random_indices = np.random.choice(len(data_points), num_clusters, replace=False)
        centroids = data_points[random_indices]

        for _ in range(max_iterations):
            clusters = [[] for _ in range(num_clusters)]
            for point in data_points:
                distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(point)

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

    def compute_probabilities(self, test_point, centroids, alpha=1.0):
        probabilities = []
        total_weight = 0
        for centroid in centroids:
            weight = math.exp(-alpha * self.euclidean_distance(test_point, centroid))
            probabilities.append(weight)
            total_weight += weight
        return [p / total_weight for p in probabilities]

    def train(self):
        # Use single-label data for clustering
        single_label_data = [row for row in self.numerical_data if "," not in row[-1]]
        self.centroids = self.k_means_clustering(single_label_data, self.num_clusters)
        
        train_probs = []
        train_labels = []
        for instance in single_label_data:
            train_point = instance[:-1]
            train_probs.append(self.compute_probabilities(train_point, self.centroids))
            train_labels.append(instance[-1])

        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(train_probs, train_labels)

    def predict(self, feature_input):
        # feature_input: dictionary of {header: value}
        numerical_row = []
        for header in self.headers[:-1]:
            val = feature_input.get(header)
            encoded_vector = [0] * len(self.label_encodings[header])
            if val in self.label_encodings[header]:
                encoded_vector[self.label_encodings[header][val]] = 1
            numerical_row.extend(encoded_vector)
            
        probs = self.compute_probabilities(numerical_row, self.centroids)
        # We can also use rf_model.predict([probs])[0] 
        pred = self.rf_model.predict([probs])[0]
        return pred

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        sys.exit(1)
        
    try:
        input_data = json.loads(sys.argv[1])
        # Use relative path from backend/ to dataset.csv
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset.csv')
        predictor = DoshaPredictor(dataset_path)
        prediction = predictor.predict(input_data)
        print(json.dumps({"dosha": prediction}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
