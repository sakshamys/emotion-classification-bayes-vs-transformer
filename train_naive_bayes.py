from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import time

print("Loading dataset...")

# Load dataset
dataset = load_dataset("dair-ai/emotion")

# Get training and test data
X_train = dataset["train"]["text"]
y_train = dataset["train"]["label"]

X_test = dataset["test"]["text"]
y_test = dataset["test"]["label"]

print("Dataset loaded.")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ---------------------------------------
# Step 1: Convert text into TF-IDF
# ---------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed.")
print(f"Number of features: {X_train_tfidf.shape[1]}")

# ---------------------------------------
# Step 2: Train Naive Bayes
# ---------------------------------------

print("\nTraining Multinomial Naive Bayes...")

start_time = time.time()

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds.")

# ---------------------------------------
# Step 3: Make predictions
# ---------------------------------------

print("\nMaking predictions...")

start_time = time.time()

y_pred = model.predict(X_test_tfidf)

prediction_time = time.time() - start_time

# ---------------------------------------
# Step 4: Evaluate model
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("NAIVE BAYES RESULTS")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Prediction time: {prediction_time:.4f} seconds")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=dataset["test"].features["label"].names
    )
)