from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Implementation of a Decision Tree model
class DecisionTree:
    def __init__(self, max_depth=5, random_state=3):
        """
        Used to initialize the Decision Tree model.
        :param max_depth: maximum depth of the tree.
        :param random_state: random state to use.
        """
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

    def fit(self, X, y):
        """
        Used to train the Decision Tree model.
        :param X:
        :param y:
        :return:
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Used to predict the labels for the given data.
        :param X:
        :return:
        """
        return self.model.predict(X)

    def plot_tree(self, name='decision_tree.png'):
        """
        Used to plot the decision tree.
        :return:
        """
        plt.figure(figsize=(30, 20))
        plot_tree(self.model, filled=True, rounded=True)
        plt.savefig('plots/decision_tree.png')