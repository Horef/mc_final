import numpy as np
from torch import nn
import torch

class FFN(nn.Module):
    """
    Implementation of a Feed Forward Neural Network for regression.
    """

    def __init__(self, input_size:int = 4, hidden_size:int = 10, output_size:int = 1, num_layers:int = 2, dropout=0.5):
        """
        Used to initialize the Feed Forward Neural Network.
        :param input_size: number of input features.
        :param hidden_size: number of neurons in the hidden layer.
        :param output_size: number of output features.
        :param num_layers: number of hidden layers.
        :param dropout: dropout rate.
        """
        super(FFN, self).__init__()
        # setting the seed for reproducibility
        torch.manual_seed(3)

        # creating the layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(hidden_size, output_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Used to perform the forward pass.
        :param x: input data.
        :return: output data.
        """
        # if x is not a tensor, convert it to a tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        for layer in self.layers:
            x = layer(x)
        return self.sigmoid(x)*100

    def predict(self, x):
        """
        Used to predict the output for the given input.
        :param x: input data.
        :return: predicted output.
        """
        # if x is not a tensor, convert it to a tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        return self.forward(x).detach().numpy()

    def fit(self, X, y, epochs=100, lr=0.01, printouts=True):
        """
        Used to train the model.
        :param X: training data.
        :param y: training labels.
        :param epochs: number of epochs.
        :param lr: learning rate.
        :param printouts: whether to print the loss or not.
        :return: nothing.
        """
        # if X and/or y is not a tensor, convert it to a tensor
        X, y = self.check_input_type([X, y])
        y = torch.transpose(y.unsqueeze(0), 0, 1)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                if printouts:
                    print(f'Epoch {epoch}, loss: {loss.item():.3f}')
        if printouts:
            print('Training finished!')

    def fit_n_test(self, X_train, y_train, X_test, y_test, epochs:int=100, lr=0.01, printouts=True):
        """
        Used to train the model and test it on the test data.
        :param X_train: training data.
        :param y_train: training labels.
        :param X_test: testing data.
        :param y_test: testing labels.
        :param epochs: number of epochs.
        :param lr: learning rate.
        :param printouts: whether to print the loss or not.
        :return: nothing.
        """
        # if X_train, y_train, X_test, and/or y_test is not a tensor, convert it to a tensor
        X_train, y_train, X_test, y_test = self.check_input_type([X_train, y_train, X_test, y_test])

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        losses = []
        mses_train = []
        mses_test = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                if printouts:
                    print(f'Epoch {epoch}, loss: {loss.item():.3f}')
            losses.append(loss.item())

            # calculating the mse for the training data
            mses_train.append(((output - y_train) ** 2).mean().item())
            # calculating the mse for the testing data
            self.train(False)
            output_test = self.forward(X_test)
            mses_test.append(((output_test - y_test) ** 2).mean().item())
            self.train(True)

        if printouts:
            print('Training finished!')

        return losses, mses_train, mses_test

    def check_input_type(self, datasets:list) -> list:
        """
        Used to check if the input data is a tensor or not.
        :param datasets: list of datasets to check.
        :return: list of datasets as tensors.
        """
        return [torch.tensor(data, dtype=torch.float32) if not torch.is_tensor(data) else data for data in datasets]