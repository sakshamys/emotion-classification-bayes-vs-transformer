from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

# Test data
X_test = dataset["test"]["text"]
y_test = dataset["test"]["label"]

# Dataset emotion names
emotion_names = dataset["test"].features["label"].names

print(f"Test samples: {len(X_test)}")

# ---------------------------------------
# Load Transformer
# ---------------------------------------

print("\nLoading Transformer model...")

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

print("Transformer loaded successfully.")

# ---------------------------------------
# Make predictions
# ---------------------------------------

print("\nPredicting emotions...")

start_time = time.time()

results = classifier(
    list(X_test),
    batch_size=16,
    truncation=True
)

prediction_time = time.time() - start_time

# ---------------------------------------
# Convert Transformer labels
# ---------------------------------------

# Transformer labels:
# anger, disgust, fear, joy, neutral, sadness, surprise

# Our dataset labels:
# sadness, joy, love, anger, fear, surprise

# There is no direct "love" label in the Transformer.
# We therefore map the Transformer emotions to the
# closest available dataset classes for comparison.

label_mapping = {
    "sadness": "sadness",
    "joy": "joy",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise"
}

# Keep track of predictions
y_pred = []
unknown_predictions = []

for result in results:

    predicted_emotion = result["label"]

    if predicted_emotion in label_mapping:
        mapped_emotion = label_mapping[predicted_emotion]
    else:
        # disgust and neutral do not exist in our dataset
        # We mark them as unknown for now.
        mapped_emotion = None
        unknown_predictions.append(predicted_emotion)

    y_pred.append(mapped_emotion)

# ---------------------------------------
# Remove predictions that cannot be mapped
# ---------------------------------------

valid_indices = [
    i for i, prediction in enumerate(y_pred)
    if prediction is not None
]

y_test_valid = [
    emotion_names[y_test[i]]
    for i in valid_indices
]

y_pred_valid = [
    y_pred[i]
    for i in valid_indices
]

# ---------------------------------------
# Evaluation
# ---------------------------------------

accuracy = accuracy_score(
    y_test_valid,
    y_pred_valid
)

print("\n==============================")
print("TRANSFORMER RESULTS")
print("==============================")

print(f"Total test samples: {len(X_test)}")
print(f"Evaluated samples: {len(valid_indices)}")
print(f"Skipped samples: {len(X_test) - len(valid_indices)}")

print(f"\nPrediction time: {prediction_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}")

# Show skipped labels
if unknown_predictions:
    print("\nTransformer labels not present in dataset:")

    from collections import Counter

    skipped_counts = Counter(unknown_predictions)

    for label, count in skipped_counts.items():
        print(f"{label}: {count}")

# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test_valid,
        y_pred_valid,
        labels=emotion_names,
        target_names=emotion_names,
        zero_division=0
    )
)

# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(
    y_test_valid,
    y_pred_valid,
    labels=emotion_names
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=emotion_names,
    yticklabels=emotion_names
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Transformer Confusion Matrix")

plt.tight_layout()
plt.show()