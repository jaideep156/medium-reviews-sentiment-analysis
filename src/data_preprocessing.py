from src.data_ingestion import read_data
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

filename = './data/dataset.csv'
df = read_data(filename)

def text_cleaning(text):
    return re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', '', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

#Combining everything into a master function
def data_preprocessing(df):
    df['review'] = df['review'].apply(text_cleaning)
    df['review'] = df['review'].apply(remove_stopwords)
    return df

df = data_preprocessing(df)