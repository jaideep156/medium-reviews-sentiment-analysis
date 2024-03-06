from src.data_ingestion import read_data
from src.data_preprocessing import data_preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

filename = './data/dataset.csv'
df = read_data(filename)
df = data_preprocessing(df)

def preparation(df):
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return vectorizer, label_encoder, X_train, y_train

def model_building(X_train, y_train):
    model = RandomForestClassifier(max_depth= None, min_samples_leaf= 1, min_samples_split= 5, n_estimators= 100) 
    #These are the best parameters according to hyperparameter tuning as seen in 'notebook/sentiment-analysis.ipynb'
    model.fit(X_train, y_train)
    return model

def prediction(text):
    input_data = vectorizer.transform([text])
    predictions = model.predict(input_data)
    predicted_sentiments = label_encoder.inverse_transform(predictions)
    print("The predicted sentiment of the above review is:",predicted_sentiments[0].lower())

vectorizer, label_encoder, X_train, y_train = preparation(df)
model = model_building(X_train, y_train)