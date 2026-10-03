from datasets import load_dataset

print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

print("\nDataset information:")
print(dataset)

print("\nFirst 5 training examples:")

for i in range(5):
    print(dataset["train"][i])

print("\nEmotion label mapping:")

print(dataset["train"].features["label"])