import streamlit as st
import pickle
import gzip
import warnings
warnings.filterwarnings("ignore")

#loading the model_steps file where model, label encoder and TF-IDF Vectorizer are stored
def load_model():
    with gzip.open(r'notebook/optimized_model_steps.pkl.gz', 'rb') as file:
        data = pickle.load(file)
    return data
data = load_model()

best_model = data["best_model"]
label_encoder = data["le"]
vectorizer = data["vectorizer"]

def is_numeric(input_str):
    try:
        float(input_str)
        return True
    except ValueError:
        return False

def show_predict_page():
    st.title("Medium App Reviews Sentiment Analysis")

    st.subheader("This tool analyzes the sentiments expressed in reviews of the [Medium](https://play.google.com/store/apps/details?id=com.medium.reader) app on the Google Play Store.")

    st.markdown("Trained on data which is available on [Kaggle](https://www.kaggle.com/datasets/raqhea/medium-app-reviews-from-google-play-store/).")

    text = st.text_input("Enter your review and press submit")
    ok = st.button("Submit")

    if ok:
        # Check if input is blank
        if text.strip() == "":
            st.error("Please enter some text to predict its sentiment.")
        # Check if input is numeric
        elif is_numeric(text):
            st.error("Please enter text instead of numeric value.")
        else:
            input_data = vectorizer.transform([text])
            predictions = best_model.predict(input_data)
            predicted_sentiments = label_encoder.inverse_transform(predictions)

            st.success(f"The predicted sentiment of the review is {predicted_sentiments[0].lower()}.")

    st.write("##### You can find the full codebase & project specifics on my [GitHub](https://github.com/jaideep156/medium-reviews-sentiment-analysis).")