import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Load dataset
def load_csv(filename):
    return pd.read_csv(filename)

# Encode categorical features (excluding the target column)
def encode_features(df):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target (Dosha)

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X).toarray()
    
    return X_encoded, y, encoder  # Keep y as original text (multi-labels)

# Encode labels without SMOTE
def encode_labels(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return y_encoded, label_encoder

# Train KMeans where each cluster represents one of the doshas
def train_kmeans(X_train, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    kmeans.fit(X_train)
    return kmeans

# Get probability features for doshas from KMeans
def get_dosha_probabilities(kmeans, X_data):
    distances = kmeans.transform(X_data)  # Compute distances to cluster centers
    max_distance = np.max(distances, axis=1, keepdims=True)
    dosha_probabilities = 1 - (distances / max_distance)  # Convert to probability-like values
    return dosha_probabilities  # Columns correspond to (Vata, Pitta, Kapha)

# Train RandomForestClassifier with dosha probability features
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predict multi-label Doshas based on probability differences
def predict_multi_label(model, X_test, label_encoder, threshold_percentage=0.85):
    probabilities = model.predict_proba(X_test)
    dosha_classes = model.classes_

    # Convert numerical classes back to text labels
    dosha_classes_str = label_encoder.inverse_transform(dosha_classes)

    predicted_labels = []
    for prob_row in probabilities:
        max_prob = max(prob_row)
        selected_doshas = [
            dosha_classes_str[i]  # Convert back to text labels
            for i, prob in enumerate(prob_row)
            if prob >= max_prob * threshold_percentage  # Select based on threshold
        ]
        predicted_labels.append("+".join(selected_doshas))  # Join labels correctly
    
    return predicted_labels, probabilities

# Evaluate multi-label accuracy using probability-based partial credit
def evaluate_multi_label_with_prob(y_actual, y_predicted, y_probabilities, dosha_classes):
    correct_predictions = 0
    total_partial_score = 0
    total_predictions = len(y_actual)

    for i in range(total_predictions):
        actual_doshas = set(y_actual[i].split('+'))
        predicted_doshas = set(y_predicted[i].split('+'))
        predicted_probs = dict(zip(dosha_classes, y_probabilities[i]))

        if actual_doshas == predicted_doshas:
            correct_predictions += 1
        else:
            actual_probs = [predicted_probs[d] for d in actual_doshas if d in predicted_probs]
            predicted_probs_selected = [predicted_probs[d] for d in predicted_doshas if d in predicted_probs]
            
            if actual_probs and predicted_probs_selected:
                prob_similarity = sum(min(ap, pp) for ap, pp in zip(sorted(actual_probs, reverse=True), 
                                                                     sorted(predicted_probs_selected, reverse=True)))
                total_partial_score += prob_similarity

    exact_match_accuracy = (correct_predictions / total_predictions) * 100
    partial_match_accuracy = ((correct_predictions + total_partial_score) / total_predictions) * 100  

    return exact_match_accuracy, partial_match_accuracy

# ===== MAIN EXECUTION =====
# Load and preprocess dataset
filename = "dataset.csv"
df = load_csv(filename)
X, y, encoder = encode_features(df)
y_encoded, label_encoder = encode_labels(y)  # No SMOTE, just label encoding

# Train KMeans (clusters represent doshas)
kmeans = train_kmeans(X, num_clusters=3)

# Get Dosha probability features
dosha_probs_train = get_dosha_probabilities(kmeans, X)

# Combine original features with dosha probabilities
X_train_final = np.hstack((X, dosha_probs_train))

# Train Random Forest on combined features
model = train_random_forest(X_train_final, y_encoded)

# Load validation dataset
validation_filename = "validation.csv"
df_val = load_csv(validation_filename)
X_val, y_val = df_val.iloc[:, :-1], df_val.iloc[:, -1]

# Encode validation dataset
X_val_encoded = encoder.transform(X_val).toarray()

# Get dosha probabilities for validation set
dosha_probs_val = get_dosha_probabilities(kmeans, X_val_encoded)

# Combine validation features with dosha probabilities
X_val_final = np.hstack((X_val_encoded, dosha_probs_val))

# Predict multi-label outputs
y_val_pred, y_val_probabilities = predict_multi_label(model, X_val_final, label_encoder, threshold_percentage=0.85)

# Evaluate multi-label predictions using probability-based scoring
exact_acc, partial_acc = evaluate_multi_label_with_prob(y_val, y_val_pred, y_val_probabilities, model.classes_)

# Print results
comparison_df = pd.DataFrame({
    'Actual Dosha': y_val,
    'Predicted Dosha': y_val_pred
})

print(comparison_df)
print(f"\nExact Multi-Label Accuracy: {exact_acc:.2f}%")
print(f"Partial Credit Multi-Label Accuracy: {partial_acc:.2f}%")
