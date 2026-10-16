import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src is on sys.path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_text

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    layout="wide",
)

MODEL_PATH = BASE_DIR / "models" / "tfidf_logreg.joblib"


@st.cache_resource
def load_model():
    """Loads and caches the trained TF-IDF + Logistic Regression pipeline."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def compute_word_contributions(pipeline, cleaned_text: str):
    """
    Computes local feature contributions for the given text:
    contribution = tfidf_weight * logistic_regression_coefficient
    Positive contribution pushes prediction toward Positive;
    Negative contribution pushes prediction toward Negative.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    # Transform input text into TF-IDF vector
    tfidf_vector = vectorizer.transform([cleaned_text])
    coo = tfidf_vector.tocoo()

    if coo.nnz == 0:
        return [], []

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]

    positive_contributions = []
    negative_contributions = []

    for col_idx, tfidf_val in zip(coo.col, coo.data):
        word = str(feature_names[col_idx])
        weight = float(tfidf_val * coefficients[col_idx])

        if weight > 0:
            positive_contributions.append((word, weight))
        elif weight < 0:
            negative_contributions.append((word, abs(weight)))

    # Sort in descending order of influence
    positive_contributions.sort(key=lambda x: x[1], reverse=True)
    negative_contributions.sort(key=lambda x: x[1], reverse=True)

    return positive_contributions[:6], negative_contributions[:6]


def main():
    st.title("Sentiment Analysis — Movie Reviews")
    st.markdown(
        "**TF-IDF + Logistic Regression Pipeline** trained on the official 50,000 IMDB movie reviews benchmark."
    )

    pipeline = load_model()
    if pipeline is None:
        st.error(
            "Model checkpoint not found at `models/tfidf_logreg.joblib`. "
            "Please run `python src/train.py` first to train and save the model."
        )
        return

    # Sample reviews dropdown
    samples = {
        "Custom Input": "",
        "Positive Review": "An absolute masterpiece of cinema! The performances were brilliant, emotional, and captivating from start to finish.",
        "Negative Review": "This was an utter waste of time. The plot was completely incoherent and the acting was wooden and terrible.",
        "Nuanced / Negation Review": "The movie was not good despite the high budget. I did not like the pacing and never felt connected to the characters.",
        "Mixed Review": "The cinematography and music were wonderful, but the dialogue and screenplay were painfully bad.",
    }

    selected_sample = st.selectbox("Quick-load a sample review or enter your own below:", list(samples.keys()))
    default_text = samples[selected_sample] if selected_sample != "Custom Input" else ""

    user_review = st.text_area(
        "Enter movie review:",
        value=default_text,
        height=140,
        placeholder="Type or paste a movie review here...",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    if predict_clicked or user_review.strip():
        if not user_review.strip():
            st.warning("Please enter a review before analyzing.")
            return

        cleaned_review = clean_text(user_review)
        if not cleaned_review:
            st.warning("The input contains no readable words after cleaning.")
            return

        # Inference
        probs = pipeline.predict_proba([cleaned_review])[0]
        prob_neg, prob_pos = probs[0], probs[1]
        pred_label = "Positive" if prob_pos >= 0.5 else "Negative"
        confidence = prob_pos if pred_label == "Positive" else prob_neg

        st.divider()

        # Display Prediction Cards
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if pred_label == "Positive":
                st.success("### Predicted Sentiment: **Positive**")
            else:
                st.error("### Predicted Sentiment: **Negative**")

        with col_res2:
            st.metric(
                label=f"Confidence ({pred_label})",
                value=f"{confidence * 100:.1f}%",
            )
            st.progress(float(confidence))

        st.caption(f"Estimated probabilities — Positive: {prob_pos * 100:.2f}% | Negative: {prob_neg * 100:.2f}%")

        # Display Top Contributing Words
        st.subheader("Explainability: Top Contributing Words")
        st.write(
            "Feature attribution via element-wise product of TF-IDF value and Logistic Regression coefficient ($w_i \\cdot x_i$)."
        )

        pos_words, neg_words = compute_word_contributions(pipeline, cleaned_review)

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("##### Words Pushing Positive")
            if pos_words:
                df_pos = pd.DataFrame(pos_words, columns=["Word / Bigram", "Positive Weight"])
                df_pos["Positive Weight"] = df_pos["Positive Weight"].round(4)
                st.dataframe(df_pos, use_container_width=True, hide_index=True)
            else:
                st.info("No positive vocabulary features identified.")

        with col_neg:
            st.markdown("##### Words Pushing Negative")
            if neg_words:
                df_neg = pd.DataFrame(neg_words, columns=["Word / Bigram", "Negative Weight"])
                df_neg["Negative Weight"] = df_neg["Negative Weight"].round(4)
                st.dataframe(df_neg, use_container_width=True, hide_index=True)
            else:
                st.info("No negative vocabulary features identified.")

    # How it works section
    with st.expander("How It Works (Methodology & Architecture)"):
        st.markdown("""
        1. **Text Preprocessing**: 
           - HTML tags stripped (spaces preserved to prevent gluing words).
           - Lowercased and sanitized to alphabetical characters.
        2. **TF-IDF Vectorization (`ngram_range=(1,2)`)**:
           - Extracts unigrams and bigrams up to 10,000 features.
           - **No stop-word removal**: Sklearn's default stop list strips crucial negations (`not`, `no`, `never`). Retaining them allows bigrams like `"not good"` to be recognized as negative units.
        3. **Logistic Regression Classifier**:
           - Computes class probabilities via the sigmoid function: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$.
           - Linear coefficients directly explain which tokens drive the prediction.
        """)


if __name__ == "__main__":
    main()
