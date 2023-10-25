# Helper functions to be used throughout notebooks in repo.

import matplotlib.pyplot as plt



def plot_predictions(X_train=X_train,
                     y_train=y_train,
                     X_test=X_test,
                     y_test=y_test,
                     predictions=None):
    """
    Function to plot data & predictions

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param predictions:
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, y_train, c='b', s=6, label="Train")
    plt.scatter(X_test, y_test, c='y', s=6, label="Test")
    if predictions is not None:
        plt.scatter(X_test, predictions, c='r', s=6, label="Predictions")
    plt.legend(prop={"size": 14})
