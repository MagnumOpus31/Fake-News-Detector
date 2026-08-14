import joblib
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import preprocess_text


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

FAKE_DATA_PATH = BASE_DIR / "data" / "raw" / "Fake.csv"
REAL_DATA_PATH = BASE_DIR / "data" / "raw" / "True.csv"


# --------------------------------------------------
# Load model and vectorizer
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

fake_df = pd.read_csv(FAKE_DATA_PATH)
real_df = pd.read_csv(REAL_DATA_PATH)

fake_df["label"] = 0
real_df["label"] = 1

df = pd.concat([fake_df, real_df], ignore_index=True)
df["text"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
)

df["text"] = df["text"].apply(preprocess_text)

# --------------------------------------------------
# Duplicate Article Analysis
# --------------------------------------------------

print("\n" + "=" * 50)
print("           DATASET DUPLICATE ANALYSIS")
print("=" * 50)

total_articles = len(df)

duplicate_articles = df["text"].duplicated().sum()

unique_articles = df["text"].nunique()

print(f"\nTotal articles  : {total_articles}")
print(f"Unique articles : {unique_articles}")
print(f"Duplicate rows  : {duplicate_articles}")

if duplicate_articles > 0:
    print("\n⚠️ Duplicate articles detected.")
else:
    print("\n✓ No duplicate articles detected.")


# --------------------------------------------------
# Prepare text
# --------------------------------------------------

df["text"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
)

df["text"] = df["text"].apply(preprocess_text)
# --------------------------------------------------
# Remove duplicate articles
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["text"],
    keep="first"
).reset_index(drop=True)

after_duplicates = len(df)

print("\nDuplicate-free dataset:")
print(f"Articles before removing duplicates : {before_duplicates}")
print(f"Articles after removing duplicates  : {after_duplicates}")
print(f"Duplicates removed                 : {before_duplicates - after_duplicates}")


# --------------------------------------------------
# Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# --------------------------------------------------
# Convert test data to TF-IDF
# --------------------------------------------------

X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)


print("\n" + "=" * 50)
print("           TRUTHLENS MODEL EVALUATION")
print("=" * 50)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nClassification Report:")
print("-" * 50)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["FAKE", "REAL"]
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

print("Confusion Matrix:")
print("-" * 50)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\nMatrix layout:")
print("[[Fake predicted as Fake, Fake predicted as Real]")
print(" [Real predicted as Fake, Real predicted as Real]]")


print("\n" + "=" * 50)