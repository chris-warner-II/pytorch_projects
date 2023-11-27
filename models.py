# pytorch model definitions to be used throughout notebooks in the repo:

import torch
from torch import nn

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def train_step_multi_classification(data: torch.Tensor,
                                    labels: torch.Tensor,
                                    model: nn.Module,
                                    loss_fn: nn.Module,
                                    accuracy_fn,
                                    optimizer: torch.optim.Optimizer):
    """
    For multiclass classification problem, perform training step inside training loop.

    :args:
        :data: - training data - [NxD] torch.Tensor where D is dimension and N is number of data points
        :labels: - training labels - [Nx1] torch.Tensor
        :model: - torch model that will learn params on training data - nn.Module
        :loss_fn: - loss function used to compare model predictions to ground truth labels
            to perform backprop & gradient descent. nn.Module
            Note: for binary classification problems, use are nn.BCELoss() or nn.BCEWithLogitsLoss().
        :accuracy_fn: - function to compute accuracy of model predictions relative to labels.
        :optimizer: - torch.optim.Optimizer to perform gradient descent step. (ex. SGD or Adam)

    :returns:
        :loss: scalar value inside a torch.Tensor that is loss accumulated across all data points.
        :acc: scalar value inside a torch.Tensor that is accuracy accumulated across all data points.
    """

    # 1. Put model in training mode to track gradients
    model.train()

    # 2. Forward Pass: Make prediction with model
    y_logits = model(data)

    # 3. Calculate the loss / error between model predictions and ground truth labels
    loss = loss_fn(y_logits, labels)

    # 4. Optimizer zero grad
    optimizer.zero_grad()

    # 5. Backpropagate loss through the network
    loss.backward()

    # 6. Perform gradient descent using optimizer.
    optimizer.step()

    # 7. Convert model prediction y_logits -> probability -> predictions.
    y_prob = torch.softmax(y_logits, dim=1)
    y_pred = torch.argmax(y_prob, dim=1)

    # 8. Use model predictions and labels to compute accuracy
    acc = accuracy_fn(y_pred,labels)

    return loss, acc


def test_step_multi_classification(data: torch.Tensor,
                                   labels: torch.Tensor,
                                   model: nn.Module,
                                   loss_fn: nn.Module,
                                   accuracy_fn):

    """
    For multiclass classification problem, perform test step inside training loop.

    :args:
        :data: - training data - [NxD] torch.Tensor where D is dimension and N is number of data points
        :labels: - training labels - [Nx1] torch.Tensor
        :model: - torch model that will learn params on training data - nn.Module
        :loss_fn: - loss function used to compare model predictions to ground truth labels
            to perform backprop & gradient descent. nn.Module
            Note: for binary classification problems, use are nn.BCELoss() or nn.BCEWithLogitsLoss().
        :accuracy_fn: - function to compute accuracy of model predictions relative to labels.

    :returns:
        :loss: scalar value inside a torch.Tensor that is loss accumulated across all data points.
        :acc: scalar value inside a torch.Tensor that is accuracy accumulated across all data points.
    """

    # 1. Set model in eval mode
    model.eval()

    # 2. Forward pass: Run test data through model
    with torch.inference_mode():
        y_logits = model(data)

    # 3. Calculate loss between model output and labels.
    loss = loss_fn(y_logits, labels)

    # 4. Convert model output logits -> probability -> prediction
    y_prob = torch.softmax(y_logits, dim=1)
    y_pred = torch.argmax(y_prob, dim=1)

    # 5. Compute accuracy of model predictions, comparing them to labels.
    acc = accuracy_fn(y_pred, labels)

    return loss, acc


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def train_step_bin_classification(data: torch.Tensor,
                                  labels: torch.Tensor,
                                  model: nn.Module,
                                  loss_fn: nn.Module,
                                  accuracy_fn,
                                  optimizer: torch.optim.Optimizer):
    """
    For binary classification problem, perform training step inside training loop.

    :args:
        :data: - training data - [NxD] torch.Tensor where D is dimension and N is number of data points
        :labels: - training labels - [Nx1] torch.Tensor
        :model: - torch model that will learn params on training data - nn.Module
        :loss_fn: - loss function used to compare model predictions to ground truth labels
            to perform backprop & gradient descent. nn.Module
            Note: for binary classification problems, use are nn.BCELoss() or nn.BCEWithLogitsLoss().
        :accuracy_fn: - function to compute accuracy of model predictions relative to labels.
        :optimizer: - torch.optim.Optimizer to perform gradient descent step. (ex. SGD or Adam)

    :returns:
        :loss: scalar value inside a torch.Tensor that is loss accumulated across all data points.
        :acc: scalar value inside a torch.Tensor that is accuracy accumulated across all data points.
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


def test_step_bin_classification(data: torch.Tensor,
                                 labels: torch.Tensor,
                                 model: nn.Module,
                                 loss_fn: nn.Module,
                                 accuracy_fn):
    """
    For binary classification problem, perform test step inside training loop.

    :args:
        :data: - training data - [NxD] torch.Tensor where D is dimension and N is number of data points
        :labels: - training labels - [Nx1] torch.Tensor
        :model: - torch model that will learn params on training data - nn.Module
        :loss_fn: - loss function used to compare model predictions to ground truth labels
            to perform backprop & gradient descent. nn.Module
            Note: for binary classification problems, use are nn.BCELoss() or nn.BCEWithLogitsLoss().
        :accuracy_fn: - function to compute accuracy of model predictions relative to labels.

    :returns:
        :loss: scalar value inside a torch.Tensor that is loss accumulated across all data points.
        :acc: scalar value inside a torch.Tensor that is accuracy accumulated across all data points.
    """

    # 1. Set model in eval mode
    model.eval()

    # 2. Make predictions on test_data from model
    with torch.inference_mode():
        y_logit = model(data)

    # 3. Calculate loss between model outputs and ground truth labels (use logits because loss_fn was BCEWithLogitsLoss)
    loss = loss_fn(y_logit, labels)

    # 4. Squash model outputs ranging from -inf to inf, to be between 0 & 1 and then round to get prediction
    y_prob = torch.sigmoid(y_logit)
    y_pred = torch.round(y_prob)

    # 5. Calculate accuracy between model predictions and labels
    acc = accuracy_fn(y_pred, labels)

    return loss, acc



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def train_step_regression(data: torch.Tensor,
                          labels: torch.Tensor,
                          model: nn.Module,
                          loss_fn: nn.Module,
                          optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """
    Define the training step to be used within training loop for training a regression model.
    Model trains on all data points before taking single gradient descent step for each epoch.

    :args:
        :data: - training data - [DxN] torch.Tensor where D is dimension and N is number of data points
        :labels: - training labels - [1xN] torch.Tensor
        :model: - torch model that will learn params on training data - nn.Module
        :loss_fn: - loss function used to compare model predictions to ground truth labels
            to perform backprop & gradient descent. nn.Module
            Note: for regression problems, common loss functions used are nn.L1Loss() or nn.MSELoss().
        :optimizer: optimizer to perform gradient descent step. torch.optim.Optimizer

    :returns:
        :loss: scalar value inside a torch.Tensor that is loss accumulated across all data points.
    """

    # 1. Set model in train mode.
    model.train()

    # 2. Predict responses to train_data.
    preds = model(data)

    # 3. Calculate loss between predictions & train_labels.
    loss = loss_fn(preds, labels)

    # 4. Optimizer zero grad.
    optimizer.zero_grad()

    # 5. Perform backprop with loss.
    loss.backward()

    # 6. Perform gradient descent.
    optimizer.step()

    return loss


def test_step_regression(data: torch.Tensor,
                         labels: torch.Tensor,
                         model: nn.Module,
                         loss_fn: nn.Module) -> torch.Tensor:
    """
    Define test step to be used to compute loss on test data without updating model parameters.

    :args:
        :data: - test data - [DxN] torch.Tensor where D is dimension and N is number of data points
        :labels: - test labels - [1xN] torch.Tensor
        :model: - torch model that will learn params on training data - nn.Module
        :loss_fn: - loss function used to compare model predictions to ground truth labels
            to perform backprop & gradient descent. nn.Module
            Note: for regression problems, common loss functions used are nn.L1Loss() or nn.MSELoss().

    :returns:
        :loss: scalar value inside a torch.Tensor that is loss accumulated across all data points.
    """

    # 1. Set model in evaluation mode
    model.eval()

    # 2. Make predictions with model with torch context manager in inference_mode.
    with torch.inference_mode():
        preds = model(data)

    # 3. Calculate loss between ground truth labels and model predictions.
    loss = loss_fn(preds, labels)

    return loss

def nonlin_type(nonlin_str):
    """
    Function to return a non-linear nn.Module from a user provided string input.
    Can add more non-linearities to this.
    :args:
        :nonlin_str: String indicating non-linearity.
    :returns:
        :nonlin: torch nn.Module() for the prescribed non-linearity.
    """

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
    Defines class for A Linear Regression model with a single linear layer with
    'in_dim' weights and 1 bias parameter to fit a line in 'in_dim' dimensions.

    :args:
        :in_dim: - number of dimensions for independent variable (X)
        :x: - independent variables. [NxD] torch.Tensor

    :returns:
        :y_logit: - model predictions for 1D dependent variable (y). [1xD] torch.Tensor
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear_layer1 = nn.Linear(in_features=in_dim,
                                       out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer1(x)

class LinearRegressionModelV1(nn.Module):
    """
    Defines class for a Linear Regression model with 3 linear layers and pointwise
    output non-linearities after input and hidden layers.

    :args:
        :in_dim: - number of dimensions for independent variable (X)
        :hid_dim: - number of hidden nodes in 2nd linear layer
        :nl_type: str indicating non-linearity type (relu, sigmoid, tanh, etc.)
        :x: - independent variables. [NxD] torch.Tensor

    :returns:
        :y_logit: - model predictions for 1D dependent variable (y). [1xD] torch.Tensor
    """
    def __init__(self, in_dim: int, hid_dim: int, nl_type: str):
        super().__init__()
        self.linear_layer1 = nn.Linear(in_features=in_dim, out_features=hid_dim)
        self.linear_layer2 = nn.Linear(in_features=hid_dim, out_features=hid_dim)
        self.linear_layer3 = nn.Linear(in_features=hid_dim, out_features=1)

        self.nl = nonlin_type(nl_type)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.nl( self.linear_layer1(x) )
        z = self.nl( self.linear_layer2(z) )
        return self.linear_layer3(z)


class MultiClassificationModelV0(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, hid_dim:int=8, nonlin=None):
        """
        Defines class of Model for Multi-class Classification that has linear layers with the
        potential for pointwise non-linearity activation functions between the layers.

        :args:
            :in_dim (int): number input features to model
            :out_dim (int): number input features to model (number of output classes)
            :hid_dim (int): number of hidden units between layers, default 8
            :nonlin (str): indicates type of non-linearity after layers 1 & 2, default None
            :x: - input to model

        :returns:
            :y_logit: model output. Class predictions. Must be passed through softmax & argmax to become predictions.

        Example:

        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hid_dim),
            nonlin_type(nonlin),
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nonlin_type(nonlin),
            nn.Linear(in_features=hid_dim, out_features=out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)