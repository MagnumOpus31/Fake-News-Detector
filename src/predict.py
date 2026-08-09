import joblib
from pathlib import Path

from preprocessing import preprocess_text

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths to saved model and vectorizer
MODEL_PATH = BASE_DIR / "models" / "svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Load trained model and TF-IDF vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_news(text):
    # Apply the same preprocessing used during training
    cleaned_text = preprocess_text(text)

    # Convert cleaned text into TF-IDF features
    text_tfidf = vectorizer.transform([cleaned_text])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]

    if prediction == 0:
        label = "FAKE"
    else:
        label = "REAL"

    return label


if __name__ == "__main__":
    article = input("Enter news article: ")

    result = predict_news(article)

    print("\nPrediction:", result)