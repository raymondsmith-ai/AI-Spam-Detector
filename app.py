import streamlit as st

from src.predict import load_model, predict_message


# Configure the application
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🤖",
    layout="centered"
)


# Application title
st.title("AI Spam Detector")

st.write(
    "An intelligent spam detection application "
    "powered by machine learning."
)

st.divider()


# Load the trained model
@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()

except FileNotFoundError:
    st.error(
        "Trained model not found. "
        "Please run python -m src.train first."
    )
    st.stop()


# Message input
st.subheader("Analyze a Message")

message = st.text_area(
    "Enter a text message:",
    placeholder="Type or paste your message here...",
    height=150
)


# Analyze button
if st.button("Analyze Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a message before analyzing.")

    else:
        result = predict_message(message, model)

        prediction = result["prediction"]
        confidence = result["confidence"]

        st.divider()

        st.subheader("Analysis Results")

        if prediction == "spam":

            st.error("SPAM DETECTED")

            st.write(
                "This message contains patterns "
                "associated with spam."
            )

        else:

            st.success("LEGITIMATE MESSAGE")

            st.write(
                "This message appears to be legitimate."
            )

        st.metric(
            label="Model Confidence",
            value=f"{confidence}%"
        )

        st.progress(confidence / 100)


# Application information
st.divider()

st.subheader("About This Project")

st.write(
    "This application uses Natural Language Processing "
    "(NLP) and machine learning to classify SMS messages."
)

st.write(
    "**Machine Learning Algorithm:** Multinomial Naive Bayes"
)

st.write(
    "**Text Vectorization:** TF-IDF"
)

st.write(
    "**Training Dataset:** SMS Spam Collection"
)

st.caption(
    "Predictions are statistical estimates and may be incorrect. "
    "Model confidence is not a guarantee of accuracy."
)