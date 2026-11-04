import joblib

from pathlib import Path

from src.preprocess import clean_text


# Location of our trained AI model
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_model.pkl"


def load_model():
    """
    Load our previously trained machine learning model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python -m src.train first."
        )

    model = joblib.load(MODEL_PATH)

    return model


def predict_message(message, model=None):
    """
    Predict whether a message is spam or ham.
    """

    # Check that the message contains text
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a valid message.")

    # Load the trained model if one wasn't provided
    if model is None:
        model = load_model()

    # Clean the message
    cleaned_message = clean_text(message)

    # Make a prediction
    prediction = model.predict([cleaned_message])[0]

    # Get probability estimates
    probabilities = model.predict_proba([cleaned_message])[0]

    # Find the probability assigned to the predicted class
    class_index = list(model.classes_).index(prediction)

    confidence = probabilities[class_index]

    return {
        "message": message,
        "prediction": prediction,
        "confidence": round(float(confidence) * 100, 2)
    }


def main():

    print("\n================================")
    print("       AI SPAM DETECTOR")
    print("================================")

    try:
        model = load_model()
    except FileNotFoundError as error:
        print(error)
        return

    while True:

        print("\nEnter a message to analyze.")
        print("Type 'exit' to quit.")

        message = input("\nMessage: ")

        if message.lower().strip() == "exit":
            print("\nGoodbye!")
            break

        try:
            result = predict_message(message, model)

            print("\n---------- RESULT ----------")

            if result["prediction"] == "spam":
                print("Classification: SPAM")
            else:
                print("Classification: LEGITIMATE")

            print(f"Model confidence: {result['confidence']}%")

            print("----------------------------")

        except ValueError as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()