# Helper functions to be used throughout notebooks in repo.

import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions=None):
    """
    For Linear Regression, creates scatter plot of 1D data (independent variable) vs. 1D labels (dependent variable) .
        Plots a). training data v. label,
              b). test data v. label, and
              c). test data vs model predictions (if provided).

    :args:
        :train_data:   1D vector of independent variables (X's) in training data
        :train_labels: 1D vector of dependent variables (y's) in training data
        :test_data:    1D vector of independent variables (X's) in test data
        :test_labels:  1D vector of dependent variables (y's) in test data
        :predictions:  1D vector of model predictions (y's) for test data

    :returns: None
    """

    # Put all data on cpu for plotting - Device Agnostic Code.
    train_data, train_labels = train_data.to("cpu"), train_labels.to("cpu")
    test_data, test_labels = test_data.to("cpu"), test_labels.to("cpu")
    if predictions is not None:
        predictions = predictions.to("cpu")

    plt.scatter(train_data, train_labels, c='b', s=15, label="Train")
    plt.scatter(test_data, test_labels, c='g', s=15, label="Test")
    plt.xlabel('Data')
    plt.ylabel('Label')
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=15, marker='x', label="Preds")
    plt.legend(prop={"size": 8})



def plot_loss(epoch, loss, test_loss=None, acc=None, test_acc=None, y_scale='linear'):
    """
    Plot training loss and test loss against epoch during training loop.

    :args:
        :epoch: 1D vector of training epoch on x-axis
        :loss: 1D vector of training loss on y-axis
        :test_loss: 1D vector of loss on test data on y-axis
        :y_scale: 'linear' or 'log' for scale of y-axis

    :return: None
    """

    plt.plot(epoch, loss, c='b', linestyle='-', label='Train Loss')
    if test_loss is not None:
        plt.plot(epoch, test_loss, c='r', linestyle='-', label='Test Loss')
    if acc is not None:
        plt.plot(epoch, acc, c='b', linestyle='--', label='Train Accuracy')
    if test_acc is not None:
        plt.plot(epoch, test_acc, c='r', linestyle='--', label='Test Accuracy')
    plt.yscale(y_scale)
    plt.ylabel('Loss | Accuracy')
    plt.xlabel('Training Epoch')
    plt.legend(prop={"size": 14})


def accuracy_fn(label_pred, label_true):
    """
    Compute accuracy for binary classification task.
        Accuracy = (# correct predictions) / (# total predictions)
    :args:
        :label_pred: - model predicted labels
        :label_true: - ground truth labels

    :return:
        :accuracy: - fraction of model predictions that match ground truth
    """
    correct = (label_pred == label_true).sum()
    return correct / len(label_true)


def scatter_2D_class(train_data, train_labels, test_data, test_labels):
    """
    For multi-class classification, make two scatter subplots of 2D data
    (one for test data, one for train data) color coding the data by different classes.

    Args:
        train_data: 2D vector of (x,y) position of scatter points for training data
        train_labels: 1D vector of class label for each data point in training set
        test_data: 2D vector of (x,y) position of scatter points for test data
        test_labels: 1D vector of class label for each data point in test set

    Returns: None
    """

    # Put all data on cpu for plotting - Device Agnostic Code.
    train_data, train_labels = train_data.to("cpu"), train_labels.to("cpu")
    test_data, test_labels = test_data.to("cpu"), test_labels.to("cpu")

    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.scatter(train_data[:,0],train_data[:,1] , c=train_labels)
    plt.title("Train")

    # Plot accuracy
    plt.subplot(1,2,2)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels)
    plt.title("Test")



def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())