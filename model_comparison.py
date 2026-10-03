import matplotlib.pyplot as plt
import os

# Final project results
models = ["Naive Bayes", "Transformer"]

accuracy = [69.40, 92.30]
macro_f1 = [44.00, 87.70]
weighted_f1 = [63.00, 92.33]

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

x = range(len(models))
width = 0.25

plt.figure(figsize=(9, 6))

plt.bar(
    [i - width for i in x],
    accuracy,
    width,
    label="Accuracy"
)

plt.bar(
    x,
    macro_f1,
    width,
    label="Macro F1"
)

plt.bar(
    [i + width for i in x],
    weighted_f1,
    width,
    label="Weighted F1"
)

plt.xticks(list(x), models)
plt.ylabel("Score (%)")
plt.xlabel("Model")
plt.title("Naive Bayes vs Transformer Performance")
plt.ylim(0, 100)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Model comparison graph saved successfully.")