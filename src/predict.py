import joblib
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths to saved model and vectorizer
MODEL_PATH = BASE_DIR / "models" / "svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Load trained model and TF-IDF vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_news(text):
    # Convert article text into TF-IDF features
    text_tfidf = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    score=model.decision_function(text_tfidf)[0]

    if prediction == 0:
        label="FAKE"
    else:
        label="REAL"
    return label, score


if __name__ == "__main__":
    article = input("Enter news article: ")

    result, score = predict_news(article)

    print("\nPrediction:", result)
    print("Decision score:", score) 