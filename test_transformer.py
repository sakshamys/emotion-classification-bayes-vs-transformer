from transformers import pipeline

print("Loading Transformer model...")

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

print("Model loaded successfully!\n")

# Test sentences
texts = [
    "I am extremely happy today!",
    "I feel very sad and lonely.",
    "I am really angry about what happened.",
    "I am scared about the situation.",
    "I love spending time with my family.",
    "Wow, I did not expect that!"
]

print("Predictions:\n")

for text in texts:
    result = classifier(text)[0]

    print(f"Text: {text}")
    print(f"Emotion: {result['label']}")
    print(f"Confidence: {result['score']:.4f}")
    print("-" * 50)