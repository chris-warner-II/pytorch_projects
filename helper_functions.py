# Helper functions to be used throughout notebooks in repo.

import matplotlib.pyplot as plt


def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions=None):
    """
    Function to plot training and test data vs. labels for linear regression dataset
    as well as test data vs. model predictions.

    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param predictions:
    :return: None
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=15, label="Train")
    plt.scatter(test_data, test_labels, c='g', s=15, label="Test")
    plt.xlabel('Data')
    plt.ylabel('Label')
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=15, marker='x', label="Predictions")
    plt.legend(prop={"size": 14})


def plot_loss(epoch, loss, test_loss=None,y_scale='linear'):
    """

    :param epoch:
    :param loss:
    :param test_loss:
    :param y_scale
    :return: None
    """
    plt.figure(figsize=(10,7))
    plt.plot(epoch, loss, c='b', label='Train')
    if test_loss is not None:
        plt.plot(epoch, test_loss, c='r', label='Test')
    plt.yscale(y_scale)
    plt.legend(prop={"size": 14})



