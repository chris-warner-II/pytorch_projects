# pytorch model definitions to be used throughout notebooks in the repo:

import torch
from torch import nn

from helper_functions import accuracy_fn

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def train_step_multi_classification(data, labels, model, loss_fn, optimizer):
    """

    :param data:
    :param labels:
    :param model:
    :param loss_fn:
    :param optimizer:
    :return: loss, acc
    """

    model.train()

    y_logits = model(data)

    loss = loss_fn(y_logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_prob = torch.softmax(y_logits, dim=1)
    y_pred = torch.argmax(y_prob, dim=1)

    acc = accuracy_fn(y_pred,labels)


    return loss, acc


def test_step_multi_classification(data, labels, model, loss_fn):
    """

    :param data:
    :param labels:
    :param model:
    :param loss_fn:
    :return:
    """

    model.eval()

    with torch.inference_mode():
        y_logits = model(data)

    loss = loss_fn(y_logits, labels)

    y_prob = torch.softmax(y_logits, dim=1)
    y_pred = torch.argmax(y_prob, dim=1)

    acc = accuracy_fn(y_pred, labels)

    return loss, acc


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def train_step_bin_classification(data, labels, model, loss_fn, optimizer):
    """

    :param data:
    :param labels:
    :param model:
    :param loss_fn:
    :param optimizer:
    :return: loss, acc
    """
    model.train()

    y_logit = model(data)
    y_prob = torch.sigmoid(y_logit)
    y_pred = torch.round(y_prob)

    loss = loss_fn(y_logit, labels)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    acc = accuracy_fn(y_pred, labels)

    return loss, acc


def test_step_bin_classification(data, labels, model, loss_fn):
    """

    :param data:
    :param labels:
    :param model:
    :param loss_fn:
    :return:
    """

    model.eval()

    with torch.inference_mode():
        y_logit = model(data)

    y_prob = torch.sigmoid(y_logit)
    y_pred = torch.round(y_prob)

    loss = loss_fn(y_logit, labels)

    acc = accuracy_fn(y_pred, labels)

    return loss, acc



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def train_step_regression(data, labels, model, loss_fn, optimizer):
    """
    Define the training step to be used within training loop.
    Predict responses to train_data.
    Calculate loss between predictions & train_labels.
    Perform backprop with loss.
    Perform gradient descent.
    :return: loss
    """

    model.train()

    preds = model(data)
    loss = loss_fn(preds, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def test_step_regression(data, labels, model, loss_fn):
    """
    Define test step to be used to compute test_loss.
    Predict responses to test_data with model params fixed.
    Calculate loss
    :return: test_loss
    """

    model.eval()

    with torch.inference_mode():
        preds = model(data)

    loss = loss_fn(preds, labels)

    return loss

def nonlin_type(nonlin_str):

    if nonlin_str is None:
        nonlin = nn.Identity()
    elif nonlin_str == 'relu':
        nonlin = nn.ReLU()
    elif nonlin_str == 'sigmoid':
        nonlin = nn.Sigmoid()
    elif nonlin_str == 'tanh':
        nonlin = nn.Tanh()
    else:
        print(f"Dont recognize nonlinearity {nonlin_str}. Setting nonlin to nn.Identity")
        nonlin = nonlin = nn.Identity()

    return nonlin

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class LinearRegressionModelV0(nn.Module):
    """
    A Linear Regression model with 'in_dim' weight and 1 bias parameter to fit a line in
    'in_dim' dimensions.
    """
    def __init__(self,in_dim):
        super().__init__()
        self.linear_layer1 = nn.Linear(in_features=in_dim,
                                       out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer1(x)

class LinearRegressionModelV1(nn.Module):
    """
    A Linear Regression model with 3 linear layers (input, hidden & output) and pointwise
    output non-linearities after input and hidden layers.
    """
    def __init__(self,in_dim, hid_dim, nl_type):
        super().__init__()
        self.linear_layer1 = nn.Linear(in_features=in_dim, out_features=hid_dim)
        self.linear_layer2 = nn.Linear(in_features=hid_dim, out_features=hid_dim)
        self.linear_layer3 = nn.Linear(in_features=hid_dim, out_features=1)

        if nl_type == 'sigmoid':
            self.nl = nn.Sigmoid()
        elif nl_type == 'relu':
            self.nl = nn.ReLU()
        elif nl_type == 'tanh':
            self.nl = nn.Tanh()
        else:
            raise NameError('nl_type')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.nl( self.linear_layer1(x) )
        z = self.nl( self.linear_layer2(z) )
        return self.linear_layer3(z)


class MultiClassificationModelV0(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8, nonlin=None):
        """
        Initializes Model for Multi-class Classification.

        Args:
            input_features (int): number input features to model
            output_features (int): number input features to model (number of output classes)
            hidden_units (int): number of hidden units between layers, default 8

        Returns:

        Example:

        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nonlin_type(nonlin),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nonlin_type(nonlin),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)