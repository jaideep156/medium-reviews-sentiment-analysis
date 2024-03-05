## This file loads the best model from notebook/model_steps.pkl.gz and uses it to make predictions on new data
## Open notebook/sentiment-analysis.ipynb for a detailed walkthrough
import pickle
import gzip

def load_model():
    with gzip.open(r'notebook\model_steps.pkl.gz', 'rb') as file:
        data = pickle.load(file)
    return data
data = load_model()

best_model = data["best_model"]
label_encoder = data["le"]
vectorizer = data["vectorizer"]

def predict_new_data(text):
    input_data = vectorizer.transform([text])
    predictions = best_model.predict(input_data)
    predicted_sentiments = label_encoder.inverse_transform(predictions)

    print("The predicted sentiment of your review is:",predicted_sentiments[0].lower())

predict_new_data("LOVE THIS THANK YOU!")