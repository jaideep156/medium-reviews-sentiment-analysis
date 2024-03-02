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
    
    text = st.text_input("Enter the review and press the button below")
    ok = st.button("Submit")

    if ok:
        input_data = vectorizer.transform([text])
        predictions = best_model.predict(input_data)
        predicted_sentiments = label_encoder.inverse_transform(predictions)

        st.write('The predicted sentiment of the above review is:',predicted_sentiments[0].lower())
