import numpy as np
import pandas as pd
import data_preprocessing as dp
import matplotlib.pyplot as plt
from models import KNearestNeighbors as knn
from models import FeedForwardNetwork as ffn
from sklearn.linear_model import LinearRegression
from models import DecisionTree as dt
from tqdm import tqdm

if __name__ == '__main__':
    # loading the data
    data = pd.read_pickle('labels/full_data.pkl')
    # cleaning the data
    data = dp.clean_data(data)
    # splitting the data
    data_folds = dp.fold_split(data, folds=5)

    print('K-Nearest Neighbors Model')
    # trying different knn models
    # doing the cross validation
    mses_list = []
    for i in tqdm(range(len(data_folds)), desc='Cross Validation'):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i+1:])
        train_X, train_y = train_data.drop(columns=['sureness']), train_data['sureness']
        test_X, test_y = test_data.drop(columns=['sureness']), test_data['sureness']

        max_neighbors = len(train_X)
        knn_models = [knn.KNN(n_neighbors=n_n) for n_n in range(1, max_neighbors)]

        mses = []
        for model, n_n in zip(knn_models, range(1, max_neighbors)):
            # training the model
            model.fit(X=train_X, y=train_y)
            # predicting the labels
            predictions = model.predict(test_X)
            # calculating the average error
            mse = ((predictions - test_y) ** 2).mean()
            mses.append(mse)
            #print(f'Mean Squared Error for the KNN model with num of neighbors {n_n} is: {mse:.2f}')
        mses_list.append((mses, max_neighbors))

    # plotting the average errors
    plt.figure(figsize=(10, 6))
    for i in range(len(data_folds)):
        plt.plot(range(1, mses_list[i][1]), mses_list[i][0], label=f'Fold {i+1}')
    # plotting the average error over all folds
    avg_mses = np.array([mses[0] for mses in mses_list]).mean(axis=0)
    plt.plot(range(1, mses_list[0][1]), avg_mses, label='Average', color='black', linestyle='--')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('KNN Model Performance')
    plt.savefig('plots/knn_performance.png')
    # printing the average mse for each number of neighbors
    print('Average MSE for each number of neighbors:')
    for i, mse in enumerate(avg_mses):
        print(f'Number of neighbors: {i+1}, Average MSE: {mse:.2f}')
    # printing the best number of neighbors
    best_n_neighbors = np.argmin(avg_mses) + 1
    print(f'The best number of neighbors is: {best_n_neighbors}, with an average MSE of: {avg_mses[best_n_neighbors-1]:.2f}')

    # # testing a simple feed forward network model
    # test_data = data_folds[0]
    # train_data = pd.concat(data_folds[:0] + data_folds[0 + 1:])
    # train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
    # test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()


    print('\n\nFeed Forward Network Model')
    # training the model
    epoch_num = 150
    # model = ffn.FFN(num_layers=3, hidden_size=4)
    # losses, mses_train, mses_test = model.fit_n_test(X_train=train_X, y_train=train_y, X_test=test_X, y_test=test_y,
    #                                                  epochs=epoch_num, lr=0.01)
    # # plotting the loss, mse for training and testing data over the epochs
    # # loss is on one graph, mse for training and testing data is on another graph
    # plt.subplots(1, 2, figsize=(15, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(epoch_num), losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss over Epochs')
    # plt.subplot(1, 2, 2)
    # plt.plot(range(epoch_num), mses_train, label='Training Data')
    # plt.plot(range(epoch_num), mses_test, label='Testing Data')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE')
    # plt.title('MSE over Epochs')
    # plt.legend()
    # plt.savefig('plots/ffn_performance.png')
    # print(f'Mean Squared Error for the Feed Forward Network model is: {mses_test[-1]:.2f}')

    # trying different feed forward network models
    # doing the cross validation
    mses_list = []
    for i in tqdm(range(len(data_folds)), desc='Cross Validation'):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i+1:])
        train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
        test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

        # training the model
        model = ffn.FFN(hidden_size=10, num_layers=3)
        model.fit(X=train_X, y=train_y, printouts=False)
        # predicting the labels
        predictions = model.predict(test_X)
        # calculating the average error
        mse = ((predictions - test_y) ** 2).mean()
        mses_list.append(mse)
        #print(f'Mean Squared Error for the Feed Forward Network model is: {mse:.2f}')

    # printing the average mse for the feed forward network model
    avg_mse = np.array(mses_list).mean()
    print(f'Average MSE for the Feed Forward Network model is: {avg_mse:.2f}')

    print('\n\nLinear Regression Model')
    print('Cross Validation')
    # trying the linear regression model
    # doing the cross validation
    mses_list = []
    for i in tqdm(range(len(data_folds)), desc='Cross Validation'):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i+1:])
        train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
        test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

        # training the model
        model = LinearRegression()
        model.fit(train_X, train_y)
        # predicting the labels
        predictions = np.clip(model.predict(test_X),0, 100)
        # calculating the average error
        mse = ((predictions - test_y) ** 2).mean()
        mses_list.append(mse)
        #print(f'Mean Squared Error for the Linear Regression model is: {mse:.2f}')

    # printing the average mse for the linear regression model
    avg_mse = np.array(mses_list).mean()
    print(f'Average MSE for the Linear Regression model is: {avg_mse:.2f}')

    print('\nSingle Fold Testing')
    # testing the linear regression model on a single fold, printing the coefficients, mse
    test_data = data_folds[4]
    train_data = pd.concat(data_folds[:4] + data_folds[4 + 1:])
    train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
    test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

    model = LinearRegression()
    model.fit(train_X, train_y)
    predictions = np.clip(model.predict(test_X), 0, 100)
    mse = ((predictions - test_y) ** 2).mean()
    print(f'Mean Squared Error for the Linear Regression model is: {mse:.2f}')
    print(f'Coefficients: {model.coef_}')

    
    # # plotting the predictions against the true values
    # plt.figure(figsize=(10, 6))
    # plt.scatter(test_y, predictions)
    # plt.plot([0, 100], [0, 100], color='black', linestyle='--')
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # plt.title('True Values vs Predictions')
    # plt.savefig('plots/lin_reg_predictions.png')
    # # plotting the residuals
    # residuals = predictions - test_y
    # plt.figure(figsize=(10, 6))
    # plt.scatter(predictions, residuals)
    # plt.plot([0, 100], [0, 0], color='black', linestyle='--')
    # plt.xlabel('Predictions')
    # plt.ylabel('Residuals')
    # plt.title('Predictions vs Residuals')
    # plt.savefig('plots/lin_reg_residuals.png')

    print('\n\nDecision Tree Model')
    # trying the decision tree model
    # doing the cross validation
    mses_list = []
    for i in tqdm(range(len(data_folds)), desc='Cross Validation'):
        # getting the train and test data
        test_data = data_folds[i]
        train_data = pd.concat(data_folds[:i] + data_folds[i+1:])
        train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
        test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

        # training the model
        model = dt.DecisionTree(max_depth=5)
        model.fit(X=train_X, y=train_y)
        # predicting the labels
        predictions = model.predict(test_X)
        # calculating the average error
        mse = ((predictions - test_y) ** 2).mean()
        mses_list.append(mse)
        #print(f'Mean Squared Error for the Decision Tree model is: {mse:.2f}')

    # printing the average mse for the decision tree model
    avg_mse = np.array(mses_list).mean()
    print(f'Average MSE for the Decision Tree model is: {avg_mse:.2f}')

    # plotting the tree for a single fold
    test_data = data_folds[4]
    train_data = pd.concat(data_folds[:4] + data_folds[4 + 1:])
    train_X, train_y = train_data.drop(columns=['sureness']).to_numpy(), train_data['sureness'].to_numpy()
    test_X, test_y = test_data.drop(columns=['sureness']).to_numpy(), test_data['sureness'].to_numpy()

    model = dt.DecisionTree(max_depth=5)
    model.fit(X=train_X, y=train_y)
    model.plot_tree()
