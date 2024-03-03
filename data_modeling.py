from data_ingestion import read_data
from data_preprocessing import data_preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

filename = './data/dataset.csv'
df = read_data(filename)
df = data_preprocessing(df)

def encoding(df):
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

def preparation(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def prediction():
    #Loading the best RANDOM FOREST model saved in 'notebook/pkl'
    #For in depth: follow 'notebook/sentiment-analysis.ipynb'
    # it has the following hyperparamters {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}

    data = joblib.load('notebook\steps.pkl')
    best_model = data["best_model"]
    label_encoder = data["le"]
    vectorizer = data["vectorizer"]

    input_data = vectorizer.transform([text])
    predictions = best_model.predict(input_data)
    predicted_sentiments = label_encoder.inverse_transform(predictions)

    print("The predicted sentiment of the above review is:",predicted_sentiments[0].lower())

text = "Love this app!!"
prediction()