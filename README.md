# Sentiment analysis of [Medium](https://play.google.com/store/apps/details?id=com.medium.reader) app reviews from Google Play Store

This project embarks on a journey to analyze sentiment patterns within [Medium](https://play.google.com/store/apps/details?id=com.medium.reader) app reviews from Google Play Store using natural language processing (NLP) techniques and machine learning algorithms. 

## To access the live version of the app, click [here](https://medium-sentiment-analysis.streamlit.app/).

## Data
The dataset and the complete data dictionary can be found on [Kaggle](https://www.kaggle.com/datasets/raqhea/medium-app-reviews-from-google-play-store/).
## Dependencies

- [pandas](https://pandas.pydata.org/docs/index.html) to manipulate the data.
- [scikit-learn (version 1.2.2*)](https://scikit-learn.org/1.2/whats_new/v1.2.html#) for machine learning tasks.
- [matplotlib](https://matplotlib.org/) for visualizations.
- [NLTK](https://www.nltk.org/) for NLP tasks like tokenizations, removing stop words, etc. 
- [streamlit cloud](https://streamlit.io/cloud) to deploy the app. 

*I am specifically using ```scikit-learn``` version ```1.2.2``` in this project due to a bug which is discussed [here](https://discuss.streamlit.io/t/valueerror-node-array-from-the-pickle-has-an-incompatible-dtype/46682/6).

## Methodology

The data is first fetched from [Kaggle](https://www.kaggle.com/datasets/raqhea/medium-app-reviews-from-google-play-store/) and then we remove irrelevant columns ```reviewId```, ```repliedAt```, etc. You can follow the [Jupyter Notebook](https://github.com/jaideep156/medium-reviews-sentiment-analysis/blob/main/notebook/sentiment-analysis.ipynb) for a detailed walkthrough.

### Defining the pipeline
Data acquisition -> Preprocessing -> Train-Test split -> Model building -> Hyperparameter tuning the model -> Saving the model -> Predicting new data using the saved model.

### Preprocessing
After basic exploratory data analysis steps like checking null values and data types of the columns, there is a huge class imbalance in the `sentiment` column with `POSITIVE` having `39982`, `NEGATIVE` having `5863` & `NEUTRAL` having `7198` values.

Why not simply balance the classes using balancing techniques like `RandomUnderSampler` or `RandomOverSampler`?

- Discarding data can lead to information loss & adding random data to training is unethical.
- The model might not generalize well to unseen data with the original imbalance.

Next, we clean the text before giving it to the model by removing numbers, special characters & stop words. 

Now, we make a `LabelEncoder` from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) to encode the `sentiment` column values. 

Finally, the last step before train-test split is converting text into numbers using [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

### Train-Test Split

Did a 70-30 train-test split with 70% in the training set and 30% for testing set using this [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) library

### Model building

Initially, I built a basic [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) model and carried out the predictions.

### Evaluation metrics used:
- [Macro F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

I have specifically chosen Macro F1 score for the following reasons:
- Balances Performance: It considers the performance of the model on both the majority (Positive) and minority classes (Negative and Neutral), providing a more comprehensive picture of its effectiveness.
- Identifies Overfitting to Majority Class: A high macro F1 score suggests that the model is performing well on all classes, not just the majority class.
- Focusing on only Accuracy can be misleading as classifier could simply predict "Positive" for every instance and achieve a very high accuracy (around 80%). However, this wouldn't reflect the model's ability to accurately classify the minority classes (Negative and Neutral) that are also crucial.

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
F1 score = 0.75 
Accuracy = 0.87
Confusion matrix = 
[[ 1092   148   507]
 [  160  1235   795]
 [  127   312 11537]]
``` 

### Saving the model

Finally, we train the `best_rf_model` on training data again & save it along with `label_encoder`, & `vectorizer` in [`optimized_model_steps.pkl.gz`](https://github.com/jaideep156/medium-reviews-sentiment-analysis/blob/main/notebook/optimized_model_steps.pkl.gz) using `pickle` and `gzip` as follows since it is a large file:

```
data = 
{
    "best_model": best_rf_model, 
    "le": label_encoder, 
    "vectorizer": vectorizer
}

with gzip.open('optimzed_model_steps.pkl.gz', 'wb') as file:
    pickle.dump(data, file)

```

## Run Locally

Clone the project

```bash
  git clone https://github.com/jaideep156/medium-reviews-sentiment-analysis.git
```

Go to the project directory

```bash
  cd medium-reviews-sentiment-analysis
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
Clone the project

```bash
  git clone https://github.com/jaideep156/medium-reviews-sentiment-analysis.git
```
Go to the project directory

```bash
  cd medium-reviews-sentiment-analysis
```
Install dependencies
```bash
  pip install -r requirements.txt
```
and finally,
```bash
  python main.py
```
[`main.py`](https://github.com/jaideep156/medium-reviews-sentiment-analysis/blob/main/main.py) contains the whole code. The data will be fetched, preprocessed, model will be built (with the best parameters shown above) and the sentiment of the text you enter in the `main.py` will be predicted.

### OR 
If you don't want to rebuild the model everytime after each input,
follow this process in your command line AFTER building the best model from [`notebook/sentiment-analysis.ipynb`](https://github.com/jaideep156/medium-reviews-sentiment-analysis/blob/main/notebook/sentiment-analysis.ipynb) and save it as `optimized_model_steps.pkl.gz`:

Clone the project

```bash
  git clone https://github.com/jaideep156/medium-reviews-sentiment-analysis.git
```
Go to the project directory

```bash
  cd medium-reviews-sentiment-analysis
```
Install dependencies
```bash
  pip install -r requirements.txt
```
and finally,
```bash
  python src/model_loader.py
```
This loads the pre-existing best random forest model from `notebooks/optimized_model_steps.pkl.gz` and carries out the predictions.

## Deployment
This code has been deployed using [Streamlit Community Cloud](https://streamlit.io/cloud) and the file is [`app.py`](https://github.com/jaideep156/medium-reviews-sentiment-analysis/blob/main/app.py)

To run the project locally, follow [these](https://github.com/jaideep156/medium-reviews-sentiment-analysis?tab=readme-ov-file#run-locally) steps as mentioned above.