import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Used to verify that the data follows a specified format
    :param data: data to be cleaned
    :return: cleaned data
    """
    # removing rows with missing values
    data = data.dropna()
    # removing duplicate rows
    data = data.drop_duplicates()
    # resetting the index
    data = data.reset_index(drop=True)
    # in the 'sureness' column, capping all values to be between 0 and 100
    data['sureness'] = data['sureness'].clip(0, 100)

    return data

def train_test_split(data: pd.DataFrame, test_ratio: float) -> (pd.DataFrame, pd.DataFrame):
    """
    Used to split the data into training and testing sets
    :param data: data to be split
    :param test_ratio: ratio of data to be used for testing
    :return: training and testing sets
    """
    # shuffling the data
    data = data.sample(frac=1, random_state=3).reset_index(drop=True)
    # calculating the index to split the data
    split_index = int(len(data) * (1 - test_ratio))
    # splitting the data
    train_data = data[:split_index]
    test_data = data[split_index:]

    return train_data, test_data

def fold_split(data: pd.DataFrame, folds: int = 5) -> [pd.DataFrame]:
    """
    Split the data into folds
    :param data: data to split
    :param folds: number of folds
    :return: list of dataframes
    """
    # creating a copy of the data
    data = data.copy()

    # calculating the number of rows in each fold
    fold_size = len(data) // folds
    # shuffling the data
    data = data.sample(frac=1, random_state=3).reset_index(drop=True)
    # splitting the data into folds
    folded_data = [data.iloc[i*fold_size:(i+1)*fold_size] for i in range(folds)]

    return folded_data
