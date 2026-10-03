from datasets import load_dataset
from collections import Counter

# Load dataset
dataset = load_dataset("dair-ai/emotion")

# Get training labels
labels = dataset["train"]["label"]

# Count each emotion
label_counts = Counter(labels)

# Get emotion names
emotion_names = dataset["train"].features["label"].names

print("Emotion distribution in training data:\n")

for label_id, count in sorted(label_counts.items()):
    print(f"{label_id} - {emotion_names[label_id]}: {count}")

# Calculate text lengths
text_lengths = [len(text.split()) for text in dataset["train"]["text"]]

print("\nText length information:")
print(f"Shortest text: {min(text_lengths)} words")
print(f"Longest text: {max(text_lengths)} words")
print(f"Average length: {sum(text_lengths) / len(text_lengths):.2f} words")