import streamlit as st
import joblib
import warnings
warnings.filterwarnings("ignore")

#loading the steps file where model, label encoder and TF-IDF Vectorizer are stored
data = joblib.load('notebook\steps.pkl')

best_model = data["best_model"]
label_encoder = data["le"]
vectorizer = data["vectorizer"]

def show_predict_page():
    st.title("Medium App Reviews Sentiment Analysis")

    st.subheader("This is a sentiment analyzer of reviews of the [Medium](https://play.google.com/store/apps/details?id=com.medium.reader) app from Google Play Store")

    st.markdown("The dataset along with data dictionary can be found on [Kaggle](https://www.kaggle.com/datasets/raqhea/medium-app-reviews-from-google-play-store/).")

    text = st.text_input("Enter your review and press submit")
    ok = st.button("Submit")

    if ok:
        input_data = vectorizer.transform([text])
        predictions = best_model.predict(input_data)
        predicted_sentiments = label_encoder.inverse_transform(predictions)

        st.write('The predicted sentiment of the above review is:',predicted_sentiments[0].lower())

    st.write("##### You can find the full codebase & project specifics on my [GitHub](https://github.com/jaideep156/medium-reviews-sentiment-analysis).")