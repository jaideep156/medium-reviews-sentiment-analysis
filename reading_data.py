import pandas as pd

def read_data(filename):
    df = pd.read_csv(filename)
    columns_to_remove = ['reviewId','score','thumbsUpCount','reviewCreatedVersion','at','replyContent','repliedAt', 'predicted_category','appVersion']
    df.drop(columns=columns_to_remove, inplace=True)
    df = df.rename(columns={'content': 'review'})

    return df

filename = './data/dataset.csv'
df = read_data(filename)