# pytorch model definitions to be used throughout notebooks in the repo:

import torch
from torch import nn


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

    return(loss)


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

    return(loss)



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