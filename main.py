from src.data_ingestion import read_data
from src.data_preprocessing import data_preprocessing
from src.data_modeling import preparation,model_building,prediction

filename = './data/dataset.csv'
df = read_data(filename)
df = data_preprocessing(df)
vectorizer, label_encoder, X_train, y_train = preparation(df)
model = model_building(X_train, y_train)

prediction("Hate this! Uninstalling!") #Outputs: The predicted sentiment of the above review is: negative