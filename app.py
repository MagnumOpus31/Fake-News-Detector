import streamlit as st
import joblib
from pathlib import Path

from src.preprocessing import preprocess_text


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="TruthLens | Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0f14;
        color: #f5f7fa;
    }

    /* Hide Streamlit default elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }

    .hero-icon {
        font-size: 2.8rem;
        margin-bottom: 0.3rem;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -2px;
        line-height: 1.1;
        color: #f5f7fa;
    }

    .hero-title span {
        color: #6ea8fe;
    }

    .hero-subtitle {
        color: #8d98a7;
        font-size: 1rem;
        max-width: 600px;
        margin: 0.8rem auto 0 auto;
        line-height: 1.6;
    }

    /* Info cards */
    .info-card {
        background: #111720;
        border: 1px solid #202936;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        height: 100%;
    }

    .info-number {
        font-size: 1.4rem;
        font-weight: 700;
        color: #6ea8fe;
    }

    .info-label {
        color: #8d98a7;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 0.7rem;
    }

    /* Text area */
    textarea {
        background-color: #111720 !important;
        color: #f5f7fa !important;
        border: 1px solid #2a3442 !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    textarea:focus {
        border-color: #6ea8fe !important;
        box-shadow: 0 0 0 1px #6ea8fe !important;
    }
    
    textarea::placeholder {
        color:#7f8a99 !important;
        opacity: 1 !important;
    }

    /* Predict button */
    .stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 10px;
        border: none;
        background: #6ea8fe;
        color: #07111f;
        font-size: 1rem;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #8ab9ff;
        transform: translateY(-1px);
    }

    /* Result cards */
    .result-card {
        margin-top: 1.5rem;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        background: #111720;
    }

    .fake-result {
        border: 1px solid #7f1d1d;
    }

    .real-result {
        border: 1px solid #166534;
    }

    .result-label {
        font-size: 0.85rem;
        color: #8d98a7;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.4rem;
    }

    .score {
        color: #9aa4b2;
        font-size: 0.9rem;
        margin-top: 0.6rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #687383;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #202936;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load model and vectorizer
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def clear_article():
    st.session_state["article_input"] = ""

# --------------------------------------------------
# Hero section
# --------------------------------------------------

st.markdown(
    "<h1 style='text-align: center; font-size: 3.2rem; "
    "margin-bottom: 0;'>"
    "Truth<span style='color:#6ea8fe;'>Lens</span>"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color:#8d98a7; "
    "font-size:1rem; margin-top:0.8rem;'>"
    "AI-powered analysis for identifying potentially "
    "misleading news content."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("")
# --------------------------------------------------
# Model information
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">99.27%</div>
        <div class="info-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">SVM</div>
        <div class="info-label">Best Model</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">NLP</div>
        <div class="info-label">Technology</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Article input
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Analyze a news article</div>',
    unsafe_allow_html=True
)

article = st.text_area(
    "News Article",
    key="article_input",
    height=260,
    placeholder=(
        "Paste the full news article here...\n\n"
        "For better results, provide the complete article "
        "rather than just the headline."
    ),
    label_visibility="collapsed"
)

if article:
    word_count = len(article.split())
    character_count = len(article)

    st.caption(
        f"{word_count:,} words · {character_count:,} characters"
    )
else:
    st.caption("Paste an article above to begin analysis.")

# --------------------------------------------------
# Prediction
# --------------------------------------------------

# --------------------------------------------------
# Prediction
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    analyze_clicked = st.button(
        "Analyze Article",
        use_container_width=True
    )

with col2:
    st.button(
        "Clear Article",
        use_container_width=True,
        on_click=clear_article
    )




if analyze_clicked:

    if not article.strip():

        st.warning("Please enter a news article before analyzing.")

    else:

        # Apply preprocessing
        cleaned_text = preprocess_text(article)

        # Convert to TF-IDF
        text_tfidf = vectorizer.transform([cleaned_text])

        # Make prediction
        prediction = model.predict(text_tfidf)[0]

        # Get SVM decision score
        score = model.decision_function(text_tfidf)[0]

        # Display result
        if prediction == 0:

            st.error("🚨 FAKE NEWS DETECTED")

            st.caption(
                f"Model decision score: {score:.4f}"
            )

        else:

            st.success("✓ THIS ARTICLE APPEARS TO BE REAL")

            st.caption(
                f"Model decision score: {score:.4f}"
            )
# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("""
<div class="footer">
    TruthLens · Fake News Detection using NLP & Machine Learning
    <br><br>
    Predictions are model-based classifications and should not be
    considered definitive verification of factual accuracy.
</div>
""", unsafe_allow_html=True)