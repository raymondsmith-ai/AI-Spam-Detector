import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.preprocess import clean_text


def train_model():

    # Step 1: Load the real-world dataset
    print("\nLoading SMS Spam Collection dataset...")

    data_path = Path("data/SMSSpamCollection")

    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "text"]
    )

    print(f"Original messages: {len(df)}")

    # Step 2: Remove missing values
    df = df.dropna(subset=["label", "text"])

    # Step 3: Clean the messages
    print("\nCleaning text messages...")

    df["text"] = df["text"].apply(clean_text)

    # Remove empty messages
    df = df[df["text"] != ""]

    # Remove duplicate messages
    df = df.drop_duplicates(subset=["text"])

    print(f"Messages after cleaning: {len(df)}")

    # Step 4: Separate features and labels
    X = df["text"]
    y = df["label"]

    # Step 5: Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")

    # Step 6: Create machine learning pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB(alpha=0.1))
    ])

    # Step 7: Train the model
    print("\nTraining AI Spam Detector...")

    model.fit(X_train, y_train)

    # Step 8: Evaluate the model
    print("\nEvaluating model performance...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    ))

    # Step 9: Save trained model
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)

    model_file = models_path / "spam_model.pkl"

    joblib.dump(model, model_file)

    print(f"\nModel saved successfully: {model_file}")


if __name__ == "__main__":
    train_model()