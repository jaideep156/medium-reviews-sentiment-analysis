# Sentiment analysis of [Medium](https://play.google.com/store/apps/details?id=com.medium.reader) app reviews from Google Play Store

This project embarks on a journey to analyze sentiment patterns within [Medium](https://play.google.com/store/apps/details?id=com.medium.reader) app reviews from Google Play Store using natural language processing (NLP) techniques and machine learning algorithms. 

## Data
The dataset and the complete data dictionary can be found on [Kaggle](https://www.kaggle.com/datasets/raqhea/medium-app-reviews-from-google-play-store/).
## Dependencies

- [pandas](https://pandas.pydata.org/docs/index.html) to manipulate the data.
- [scikit-learn (version 1.2.2)](https://scikit-learn.org/1.2/whats_new/v1.2.html#) for machine learning tasks.
- [matplotlib](https://matplotlib.org/) for visualizations.
- [NLTK](https://www.nltk.org/) for NLP tasks like tokenizations, removing stop words, etc. 
- [imblearn](https://imbalanced-learn.org/stable/install.html) to handle class imbalance.
- [streamlit](https://streamlit.io/) to host the app. 

I am specifically using ```scikit-learn``` version ```1.2.2``` in this project due to a bug which is discussed [here](https://discuss.streamlit.io/t/valueerror-node-array-from-the-pickle-has-an-incompatible-dtype/46682/6).

## Methodology

The data is first fetched from [Kaggle](https://www.kaggle.com/datasets/raqhea/medium-app-reviews-from-google-play-store/) and then we remove irrelevant columns ```reviewId```, ```repliedAt```, etc. You can follow the [Jupyter Notebook](https://github.com/jaideep156/medium-reviews-sentiment-analysis/blob/main/notebook/sentiment-analysis.ipynb) for a detailed walkthrough.

### Defining the pipeline
Data acquisition -> Preprocessing -> Train-Test split -> Model building -> Hyperparameter tuning the model.

### Preprocessing
After basic exploratory data analysis steps like checking null values and data types of the columns, there is a huge class imbalance in the `sentiment` column with `POSITIVE` having `39982`, `NEGATIVE` having `5863` & `NEUTRAL` having `7198` values.

This was handled using `RandomUnderSampler` from `imblearn` making all classes with an equal number of observations, i.e. `5863`.

Next, we clean the text before giving it to the model by removing numbers, special characters & stop words. 

Now, we make a `LabelEncoder` from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) to encode the `sentiment` column values. 

Finally, the last step before train-test split is converting text into numbers using [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

### Train-Test Split

Did a 70-30 train-test split with 70% in the training set and 30% for testing set using this [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) library

### Model building

Initially, I built a basic [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) model and carried out the predictions.

### Evaluation metrics used:
- [Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
- [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
- [F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

### Hyperparameter Tuning the model

The existing random forest classifier is hyper-parameter tuned using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) with the following grid:

```
{
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```
### Results
Next, the model was retrained with the best parameters and the new evaluation metrics are as follows: 
``` 
Precision = 0.8
Recall = 0.8
F1 score = 0.8 
Accuracy = 0.79
``` 

### Saving the model

Finally, we save the `model`, `label_encoder`, and `vectorizer` using `joblib` as follows:

```
data = 
{
    "best_model": best_rf_model, 
    "le": label_encoder, 
    "vectorizer": vectorizer
}

joblib.dump(data, 'steps.pkl')

```

## Run Locally

Clone the project

```bash
  git clone https://github.com/jaideep156/medium-reviews-sentiment-analysis.git
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit run app.py
```
### OR 

simply run these two lines of code in your command line:
```bash
  pip install -r requirements.txt
```

```bash
  python main.py
```
`main.py` contains the whole code. The data will be fetched, preprocessed, model will be built (with the best parameters shown above) and the sentiment of the text you enter in the file will be predicted.