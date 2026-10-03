from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

# Load dataset
dataset = load_dataset("dair-ai/emotion")

# Get training labels
labels = dataset["train"]["label"]

# Count labels
label_counts = Counter(labels)

# Emotion names
emotion_names = dataset["train"].features["label"].names

# Prepare data
emotions = [emotion_names[i] for i in sorted(label_counts)]
counts = [label_counts[i] for i in sorted(label_counts)]

# Create bar chart
plt.figure(figsize=(8, 5))

plt.bar(emotions, counts)

plt.title("Emotion Distribution in Training Dataset")
plt.xlabel("Emotion")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.show()