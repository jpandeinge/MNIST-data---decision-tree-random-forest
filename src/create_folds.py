import pandas as pd
from sklearn import  model_selection

if __name__ == '__main__':
    
    # path of mnist train csv file
    df_train = pd.read_csv('../input/mnist_train.csv')
    
    # create a new column of kfolds and fill it with -1
    df_train['kfold'] = -1

    # reshuffle the dataset  and randomize the data
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    # initiate the kfold model selection from 
    kf = model_selection.KFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df_train)):
        df_train.loc[val_, 'kfold'] = fold

    # save the new csv with kfod column
    df_train.to_csv('../input/mnist_train_folds.csv', index=False)
