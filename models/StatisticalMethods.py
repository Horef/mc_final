from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import fold_split

def cross_validation(data: pd.DataFrame, model, folds: int = 5) -> (list, int):
    """
    Perform cross validation on the given data using the given model
    :param data: data to perform cross validation on
    :param model: model to use
    :param folds: number of folds
    :return: list of MSEs for each number of neighbors, best number of neighbors
    """
    # splitting the data
    data_folds = fold_split(data, folds=folds)

    # doing the cross validation
    mses_list = []
    for i in tqdm(range(len(data_folds)), desc='Cross Validation'):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i + 1:])
        train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
        test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

        # training the model
        model.fit(X=train_X, y=train_y)
        # predicting the labels
        predictions = model.predict(test_X)
        # calculating the average error
        mse = ((predictions - test_y) ** 2).mean()
        mses_list.append(mse)
        # print(f'Mean Squared Error for the Feed Forward Network model is: {mse:.2f}')

    return mses_list