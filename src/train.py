import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    # read the tarin data with folds
    df = pd.read_csv('../input/mnist_train_folds.csv')

    # training data is where kfold is not equal to provided fold 
    # also note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold data is equal to provided data
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from the dataframe and convert it to a numpy
    # array by using .values
    # target is label column in the dataframe
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values
