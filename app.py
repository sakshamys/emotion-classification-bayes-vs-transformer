import streamlit as st
import pandas as pd

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from transformers.pipelines import pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Emotion Classification",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# EMOTION LABELS
# ============================================================

emotion_names = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]


# ============================================================
# TITLE
# ============================================================

st.title("😊 Emotion Classification")
st.subheader("Naive Bayes vs Transformer")

st.write(
    "Enter a sentence below and compare the emotion predicted "
    "by a traditional Naive Bayes classifier and a modern Transformer model."
)


# ============================================================
# LOAD AND TRAIN NAIVE BAYES
# ============================================================

@st.cache_resource
def load_naive_bayes():

    dataset = load_dataset("dair-ai/emotion")

    X_train = dataset["train"]["text"]
    y_train = dataset["train"]["label"]

    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Naive Bayes
    model = MultinomialNB()

    model.fit(
        X_train_tfidf,
        y_train
    )

    return vectorizer, model


# ============================================================
# LOAD TRANSFORMER
# ============================================================

@st.cache_resource
def load_transformer():

    classifier = pipeline(
        "text-classification",
        model="Sreekant13/distilbert-emotion"
    )

    return classifier


# ============================================================
# LOAD MODELS
# ============================================================

with st.spinner("Loading models..."):

    vectorizer, nb_model = load_naive_bayes()

    transformer = load_transformer()


st.success("Models loaded successfully!")


# ============================================================
# USER INPUT
# ============================================================

st.markdown("### Enter your text")

text = st.text_area(
    "Type a sentence expressing an emotion:",
    placeholder="Example: I am extremely happy because I got selected!",
    height=120
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("🔍 Predict Emotion", type="primary"):

    if text.strip() == "":

        st.warning("Please enter some text first.")

    else:

        # ----------------------------------------------------
        # NAIVE BAYES PREDICTION
        # ----------------------------------------------------

        text_tfidf = vectorizer.transform([text])

        nb_prediction = nb_model.predict(text_tfidf)[0]

        nb_probabilities = nb_model.predict_proba(text_tfidf)[0]

        nb_confidence = max(nb_probabilities) * 100

        nb_emotion = emotion_names[nb_prediction]


        # ----------------------------------------------------
        # TRANSFORMER PREDICTION
        # ----------------------------------------------------

        transformer_result = transformer(
            text,
            truncation=True
        )[0]

        transformer_emotion = transformer_result["label"]

        transformer_confidence = (
            transformer_result["score"] * 100
        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.markdown("---")

        st.header("Prediction Results")


        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # NAIVE BAYES
        # ----------------------------------------------------

        with col1:

            st.subheader("🧠 Naive Bayes")

            st.metric(
                "Predicted Emotion",
                nb_emotion.upper()
            )

            st.metric(
                "Confidence",
                f"{nb_confidence:.2f}%"
            )


        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        with col2:

            st.subheader("🤖 Transformer")

            st.metric(
                "Predicted Emotion",
                transformer_emotion.upper()
            )

            st.metric(
                "Confidence",
                f"{transformer_confidence:.2f}%"
            )


        # ====================================================
        # AGREEMENT
        # ====================================================

        st.markdown("---")

        if nb_emotion == transformer_emotion:

            st.success(
                f"✅ Both models predict: **{nb_emotion.upper()}**"
            )

        else:

            st.info(
                "⚠️ The two models produced different predictions."
            )


        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        st.markdown("---")

        st.header("Model Comparison")

        comparison_data = {
            "Model": [
                "Naive Bayes",
                "Transformer"
            ],

            "Prediction": [
                nb_emotion,
                transformer_emotion
            ],

            "Confidence": [
                f"{nb_confidence:.2f}%",
                f"{transformer_confidence:.2f}%"
            ],

            "Test Accuracy": [
                "69.40%",
                "92.30%"
            ]
        }

        comparison_df = pd.DataFrame(
            comparison_data
        )

        st.table(comparison_df)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Project")

    st.write(
        "This project compares a traditional "
        "machine learning approach with a modern "
        "Transformer-based approach for emotion "
        "classification."
    )

    st.markdown("### Emotions")

    for emotion in emotion_names:

        st.write(f"• {emotion.capitalize()}")

    st.markdown("### Models")

    st.write("**Traditional:**")

    st.write("TF-IDF + Multinomial Naive Bayes")

    st.write("**Modern:**")

    st.write("DistilBERT Transformer")

    st.markdown("### Dataset")

    st.write("DAIR-AI Emotion Dataset")

    st.markdown("### Test Accuracy")

    st.write("Naive Bayes: **69.40%**")

    st.write("Transformer: **92.30%**")