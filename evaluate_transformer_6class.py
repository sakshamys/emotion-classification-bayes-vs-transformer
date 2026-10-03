from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

X_test = dataset["test"]["text"]
y_test = dataset["test"]["label"]

emotion_names = dataset["test"].features["label"].names

print("\nEmotion labels:")
for i, name in enumerate(emotion_names):
    print(i, "->", name)

print("\nNumber of test samples:", len(X_test))


# --------------------------------------------------
# 2. Load Transformer model
# --------------------------------------------------

print("\nLoading Transformer model...")

classifier = pipeline(
    "text-classification",
    model="Sreekant13/distilbert-emotion"
)

print("Transformer loaded successfully!")


# --------------------------------------------------
# 3. Make predictions
# --------------------------------------------------

print("\nMaking predictions...")

start_time = time.time()

results = classifier(
    list(X_test),
    batch_size=16,
    truncation=True
)

end_time = time.time()

inference_time = end_time - start_time

print("Prediction completed!")


# --------------------------------------------------
# 4. Convert predicted labels to numbers
# --------------------------------------------------

label_to_id = {
    name: i for i, name in enumerate(emotion_names)
}

y_pred = []

for result in results:

    predicted_label = result["label"]

    if predicted_label in label_to_id:
        y_pred.append(label_to_id[predicted_label])

    elif predicted_label.startswith("LABEL_"):
        label_number = int(predicted_label.split("_")[1])
        y_pred.append(label_number)

    else:
        raise ValueError(
            f"Unknown label returned by model: {predicted_label}"
        )


# --------------------------------------------------
# 5. Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("TRANSFORMER RESULTS")
print("======================================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Inference Time: {inference_time:.2f} seconds")


# --------------------------------------------------
# 6. Classification report
# --------------------------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=emotion_names,
        zero_division=0
    )
)


# --------------------------------------------------
# 7. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)


# --------------------------------------------------
# 8. Plot confusion matrix
# --------------------------------------------------

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

plt.savefig("transformer_confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved as:")
print("transformer_confusion_matrix.png")