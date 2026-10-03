from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("dair-ai/emotion")

# Emotion labels
labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

rows = []

# Export train, validation and test data
for split in ["train", "validation", "test"]:

    for i in range(len(dataset[split])):

        rows.append({
            "split": split,
            "text": dataset[split][i]["text"],
            "label": dataset[split][i]["label"],
            "emotion": labels[dataset[split][i]["label"]]
        })

# Create DataFrame
df = pd.DataFrame(rows)

# Save CSV
df.to_csv(
    "emotion_dataset_all_sentences.csv",
    index=False
)

print("====================================")
print("Dataset exported successfully!")
print("Total sentences:", len(df))
print("File: emotion_dataset_all_sentences.csv")
print("====================================")