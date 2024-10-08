import numpy as np
import pandas as pd
import data_preprocessing as dp
import matplotlib.pyplot as plt
from models import KNearestNeighbors as knn
from models import FeedForwardNetwork as ffn
from sklearn.linear_model import LinearRegression
from models import DecisionTree as dt
from models.Baseline import Baseline
from tqdm import tqdm
from models.FNNFusedKNN import FNNFusedKNN

def test_different_models():
    # loading the data
    data = pd.read_pickle('labels/full_data.pkl')
    # cleaning the data
    data = dp.clean_data(data)
    # splitting the data
    data_folds = dp.fold_split(data, folds=5)

    print('Baseline Model')
    # trying the baseline model
    # doing the cross validation
    mses_list = []
    l1_list = []
    for i in range(len(data_folds)):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i + 1:])
        train_X, train_y = train_data.drop(columns=['sureness']), train_data['sureness']
        test_X, test_y = test_data.drop(columns=['sureness']), test_data['sureness']

        # training the model
        model = Baseline()
        model.fit(X=train_X, y=train_y)
        # predicting the labels
        predictions = model.predict(test_X)
        # calculating the average error
        mse = ((predictions - test_y) ** 2).mean()
        l1 = np.abs(predictions - test_y).mean()
        mses_list.append(mse)
        l1_list.append(l1)

        print(f'Mean Squared Error for the Baseline model on fold {i} is: {mse:.2f} and the L1 is: {l1:.2f}')

    # printing the average mse for the baseline model
    avg_mse = np.array(mses_list).mean()
    avg_l1 = np.array(l1_list).mean()
    print(f'Average MSE for the Baseline model is: {avg_mse:.2f} and the average L1 is: {avg_l1:.2f}')

    print('\n\nK-Nearest Neighbors Model')
    # trying different knn models
    # doing the cross validation
    mses_list = []
    l1_list = []
    for i in range(len(data_folds)):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i + 1:])
        train_X, train_y = train_data.drop(columns=['sureness']), train_data['sureness']
        test_X, test_y = test_data.drop(columns=['sureness']), test_data['sureness']

        max_neighbors = len(train_X)
        knn_models = [knn.KNN(n_neighbors=n_n, metric='minkowski') for n_n in range(1, max_neighbors + 1)]

        mses = []
        l1s = []
        for model, n_n in zip(knn_models, range(1, max_neighbors + 1)):
            # training the model
            model.fit(X=train_X, y=train_y)
            # predicting the labels
            predictions = model.predict(test_X)
            # calculating the average error
            mse = ((predictions - test_y) ** 2).mean()
            l1 = np.abs(predictions - test_y).mean()
            mses.append(mse)
            l1s.append(l1)
            # print(f'Mean Squared Error for the KNN model with num of neighbors {n_n} is: {mse:.2f}')
        mses_list.append((mses, max_neighbors))
        l1_list.append((l1s, max_neighbors))

    # plotting the average errors
    plt.figure(figsize=(10, 6))
    for i in range(len(data_folds)):
        plt.plot(range(1, mses_list[i][1] + 1), mses_list[i][0], label=f'Fold {i + 1}')
    # plotting the average error over all folds
    avg_mses = np.array([mses[0] for mses in mses_list]).mean(axis=0)
    avg_l1s = np.array([l1[0] for l1 in l1_list]).mean(axis=0)
    plt.plot(range(1, mses_list[0][1] + 1), avg_mses, label='Average', color='black', linestyle='--')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('KNN Model Performance')
    plt.savefig('plots/knn_performance.png')
    # printing the average mse for each number of neighbors
    print('Average MSE for each number of neighbors:')
    for i, mse in enumerate(avg_mses):
        print(f'Number of neighbors: {i + 1}, Average MSE: {mse:.2f}, Average L1: {avg_l1s[i]:.2f}')
    # printing the best number of neighbors
    best_n_neighbors = np.argmin(avg_mses) + 1
    print(f'The best number of neighbors is: {best_n_neighbors},\n'
          f'with an average MSE of: {avg_mses[best_n_neighbors - 1]:.2f} and an average L1 of: {avg_l1s[best_n_neighbors - 1]:.2f}')


    print('\n\nFeed Forward Network Model')
    # doing the cross validation
    mses_list = []
    l1_list = []
    for i in range(len(data_folds)):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i + 1:])
        train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
        test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

        # training the model
        model = ffn.FFN(hidden_size=9, num_layers=3)
        model.fit(X=train_X, y=train_y, printouts=False)
        # predicting the labels
        predictions = model.predict(test_X)
        # calculating the average error
        mse = ((predictions - test_y) ** 2).mean()
        l1 = np.abs(predictions - test_y).mean()
        mses_list.append(mse)
        l1_list.append(l1)

        print(
            f'Mean Squared Error for the Feed Forward Network model on fold {i} is: {mse:.2f} and the L1 is: {l1:.2f}')

    # printing the average mse for the feed forward network model
    avg_mse = np.array(mses_list).mean()
    avg_l1 = np.array(l1_list).mean()
    print(f'Average MSE for the Feed Forward Network model is: {avg_mse:.2f} and the average L1 is: {avg_l1:.2f}')


    print('\n\nLinear Regression Model')
    print('Cross Validation')
    # trying the linear regression model
    # doing the cross validation
    mses_list = []
    l1_list = []
    for i in range(len(data_folds)):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i + 1:])
        train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
        test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

        # training the model
        model = LinearRegression()
        model.fit(train_X, train_y)
        # predicting the labels
        predictions = np.clip(model.predict(test_X), 0, 100)
        # calculating the average error
        mse = ((predictions - test_y) ** 2).mean()
        mses_list.append(mse)
        l1 = np.abs(predictions - test_y).mean()
        l1_list.append(l1)

        print(f'Mean Squared Error for the Linear Regression model on fold {i} is: {mse:.2f} and the L1 is: {l1:.2f}')
        print(f'Coefficients for fold {i}: {model.coef_}\n')

    # printing the average mse for the linear regression model
    avg_mse = np.array(mses_list).mean()
    avg_l1 = np.array(l1_list).mean()
    print(f'Average MSE for the Linear Regression model is: {avg_mse:.2f} and the average L1 is: {avg_l1:.2f}')


    print('\n\nDecision Tree Model')
    # trying the decision tree model
    # doing the cross validation
    mses_list = []
    l1_list = []
    for i in range(len(data_folds)):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i + 1:])
        train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
        test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

        # training the model
        max_depth = 20
        models = [dt.DecisionTree(max_depth=depth) for depth in range(1, max_depth + 1)]

        mses = []
        l1s = []
        for model, depth in zip(models, range(1, max_depth + 1)):
            model.fit(X=train_X, y=train_y)
            predictions = model.predict(test_X)
            mse = ((predictions - test_y) ** 2).mean()
            l1 = np.abs(predictions - test_y).mean()
            mses.append(mse)
            l1s.append(l1)
            # print(f'Mean Squared Error for the Decision Tree model with max depth {depth} is: {mse:.2f}')
        mses_list.append((mses, max_depth))
        l1_list.append((l1s, max_depth))

    # plotting the average errors
    plt.figure(figsize=(10, 6))
    for i in range(len(data_folds)):
        plt.plot(range(1, mses_list[i][1] + 1), mses_list[i][0], label=f'Fold {i + 1}')
    # plotting the average error over all folds
    avg_mses = np.array([mses[0] for mses in mses_list]).mean(axis=0)
    avg_l1s = np.array([l1[0] for l1 in l1_list]).mean(axis=0)
    plt.plot(range(1, mses_list[0][1] + 1), avg_mses, label='Average', color='black', linestyle='--')
    plt.xlabel('Max Depth')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Decision Tree Model Performance')
    plt.savefig('plots/decision_tree_performance.png')
    # printing the average mse for each depth
    print('Average MSE for each max depth:')
    for i, mse in enumerate(avg_mses):
        print(f'Max Depth: {i + 1}, Average MSE: {mse:.2f}, Average L1: {avg_l1s[i]:.2f}')
    # printing the best depth
    best_max_depth = np.argmin(avg_mses) + 1
    print(f'The best max depth is: {best_max_depth},\n'
          f'with an average MSE of: {avg_mses[best_max_depth - 1]:.2f} and an average L1 of: {avg_l1s[best_max_depth -
                                                                                                      1]:.2f}')

    # plotting the tree for a single fold
    test_data = data_folds[4]
    train_data = pd.concat(data_folds[:4] + data_folds[4 + 1:])
    train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
    test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

    model = dt.DecisionTree(max_depth=1)
    model.fit(X=train_X, y=train_y)
    model.plot_tree()

def create_FFN_data():
    # loading the data, training a feed forward network model on all data
    # later making predictions on the whole dataset, and saving the data with the predictions
    print('Loading the data')
    data = pd.read_pickle('labels/full_data.pkl')
    # cleaning the data
    data = dp.clean_data(data)

    print('Training the Feed Forward Network model')
    # training the model
    model = ffn.FFN(hidden_size=9, num_layers=3)
    model.fit(X=data.drop(columns=['sureness']).to_numpy(), y=data['sureness'].to_numpy(), printouts=False)
    # predicting the labels
    predictions = model.predict(data.drop(columns=['sureness']).to_numpy())
    data['sureness'] = predictions
    data.to_pickle('labels/ffn_data.pkl')
    print('Data with predictions saved')

if __name__ == '__main__':
    test_different_models()
    # create_FFN_data()
