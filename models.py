# pytorch model definitions to be used throughout notebooks in the repo:

import torch
from torch import nn

from tqdm.auto import tqdm         # for progress bar

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

def train_step_batch_multi_classification(model: nn.Module,
                                           dataloader: torch.utils.data.DataLoader,
                                           optimizer: torch.optim.Optimizer,
                                           loss_fn: nn.Module,
                                           accuracy_fn,
                                           device: torch.device = None,
                                          verbose_model:bool = False):
    """
    Training step to train a multiclass classification model. Is called within training loop
    over epochs and a loop over batches in dataloader as well.

    :args:
        :model: - torch model that will learn params on training data - nn.Module
        :dataloader: - DataLoader iterable from torch.utils.data to feed in data by the batch.
        :optimizer: - optimizer to perform gradient descent step. torch.optim.Optimizer
        :loss_fn: - loss function used to compare model logits to ground truth labels
            to perform backprop & gradient descent. nn.Module
            Note: for multiclass classification problems, loss functions is nn.CrossEntropyLoss().
        :accuracy_fn: -  proportion of model predicted classes that match ground truth
        :device: - device that code is run on ("cuda" or "cpu")

    :returns:
        :train_loss: - average loss per batch across all batches in training dataset
        :train_acc:  - average accuracy per batch across all batches in training dataset
    """

    # Put model in training mode to track gradients and update weights
    model.train()

    train_loss, train_acc = 0, 0

    # Loop through batches
    for batch, (X, y_labels) in enumerate(dataloader):
        # Put data on target device
        X, y_labels = X.to(device), y_labels.to(device)

        # Forward pass
        y_logits = model(X,verbose=verbose_model)

        # calculate loss
        loss = loss_fn(y_logits, y_labels)
        train_loss += loss

        # The Meat
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute accuracy
        y_probs = torch.softmax(y_logits, dim=1)
        y_preds = torch.argmax(y_probs, dim=1)
        acc = accuracy_fn(y_preds, y_labels)
        train_acc += acc

    # Get average loss, acc per batch.
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    # Print whats happening
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.4f}")

    return train_loss, train_acc








# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def test_step_batch_multi_classification(model: nn.Module,
                                          dataloader: torch.utils.data.DataLoader,
                                          loss_fn: nn.Module,
                                          accuracy_fn,
                                          device: torch.device = None):
    """
    Compute loss and accuracy on test data without updating model params for multiclass
    classification model. Called within training loop.

    :args:
        :model: - torch model that will learn params on training data - nn.Module
        :dataloader: - DataLoader iterable from torch.utils.data to feed in data by the batch.
        :loss_fn: - loss function used to compare model logits to ground truth labels
            to perform backprop & gradient descent. nn.Module
            Note: for multiclass classification problems, loss functions is nn.CrossEntropyLoss().
        :accuracy_fn: -  proportion of model predicted classes that match ground truth
        :device: - device that code is run on ("cuda" or "cpu")

    :returns:
        :test_loss: - average loss per batch across all batches in test dataset
        :test_acc:  - average accuracy per batch across all batches in test dataset
    """

    # Put model in eval mode
    model.eval()

    test_loss, test_acc = 0, 0

    # Put on inference mode context manager
    with torch.inference_mode():
        for batch, (X, y_labels) in enumerate(dataloader):
            # Put data on device
            X, y_labels = X.to(device), y_labels.to(device)

            # Forward pass
            y_logits = model(X)

            # Compute loss
            loss = loss_fn(y_logits, y_labels)
            test_loss += loss

            # compute accuracy
            y_probs = torch.softmax(y_logits, dim=1)
            y_preds = torch.argmax(y_probs, dim=1)
            acc = accuracy_fn(y_preds, y_labels)
            test_acc += acc

        # Get average loss, acc per batch.
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    # Print whats happening
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.4f}")

    return test_loss, test_acc






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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def eval_model_multi_classification(model: nn.Module,
                                    dataloader: torch.utils.data.DataLoader,
                                    loss_fn: nn.Module,
                                    accuracy_fn,
                                    device: torch.device = None):
    """Returns dictionary containing results of model predicting on dataloader

    :args:
        :model: - torch model that will learn params on training data - nn.Module
        :dataloader: - DataLoader iterable from torch.utils.data to feed in data by the batch.
        :loss_fn: - loss function used to compare model logits to ground truth labels
            to perform backprop & gradient descent. nn.Module
            Note: for multiclass classification problems, loss functions is nn.CrossEntropyLoss().
        :accuracy_fn: -  proportion of model predicted classes that match ground truth
        :device: - device that code is run on ("cuda" or "cpu")

    :returns: - dictionary containing...
        "model_name" - name of model
        "model_loss" - average loss per batch
        "model_acc" - average accuracy per batch

    """

    print(f"Eval model {model.__class__.__name__} on dataset {dataloader.dataset}")

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y_labels in tqdm(dataloader):
            # Put data on device (device agnostic code)
            X, y_labels = X.to(device), y_labels.to(device)

            # Make predictions
            y_logits = model(X)

            # Accumulate loss and accuracy per batch
            loss += loss_fn(y_logits, y_labels)
            acc += accuracy_fn(y_logits.argmax(dim=1), y_labels)

        # Divide loss and accuracy by number of batches to get average per batch.
        loss /= len(dataloader)
        acc /= len(dataloader)

        return {"model_name": model.__class__.__name__,
                "model_loss": loss.item(),
                "model_acc": acc.item()}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def make_predictions_multi_classification(model: nn.Module,
                                          data: list):
    """
        Function to output list of prediction probability tensors, one for each sample in data.
        Predictions made by model.

        :args:
            :model: - trained model to use to make predictions
            :data: - list of input images

        :returns:
            :pred_probs: - list of vector of prediction probabilities for all classes for each input image
    """

    pred_probs = []

    model.eval()
    with torch.inference_mode():
        for img in data:
            # Forward pass, model outputs raw logits
            y_logit = model(img.unsqueeze(0))

            # Convert logits -> prediction probabilities
            y_prob = y_logit.softmax(dim=1)

            pred_probs.append(y_prob)

    return pred_probs


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

class SingleLayerLinearModel(nn.Module):
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class ThreeLayerModel(nn.Module):
    """
    Defines class for a Linear Regression model with 3 linear layers and pointwise
    output non-linearities after input and hidden layers.
    Note: this model is functionally identical to and interchangeable with ThreeLayerNonlinModel2.

    :args:
        :in_dim: - number of dimensions for independent variable (X)
        :hid_dim: - number of hidden nodes in 2nd linear layer
        :out_dim: - number of outputs, default = 1 for regression
        :nl_type: str indicating non-linearity type (relu, sigmoid, tanh, etc.)
        :x: - independent variables. [NxD] torch.Tensor

    :returns:
        :y_logit: - model predictions for 1D dependent variable (y). [1xD] torch.Tensor
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int = 1, nl_type: str = None):
        super().__init__()
        self.linear_layer1 = nn.Linear(in_features=in_dim, out_features=hid_dim)
        self.linear_layer2 = nn.Linear(in_features=hid_dim, out_features=hid_dim)
        self.linear_layer3 = nn.Linear(in_features=hid_dim, out_features=out_dim)

        self.nl = nonlin_type(nl_type)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.nl( self.linear_layer1(x) )
        z = self.nl( self.linear_layer2(z) )
        return self.linear_layer3(z)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class ThreeLayerModel2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hid_dim: int = 8, nl_type: str = None):
        """
        Defines class of Model for Multi-class Classification that has linear layers with the
        potential for pointwise non-linearity activation functions between the layers.
        Note: this model is functionally identical to and interchangeable with ThreeLayerNonlinModel.

        :args:
            :in_dim (int): number input features to model
            :out_dim (int): number input features to model (number of output classes)
            :hid_dim (int): number of hidden units between layers, default 8
            :nl_type (str): indicates type of non-linearity after layers 1 & 2, default None
            :x: - input to model

        :returns:
            :y_logit: model output. Class predictions. Must be passed through softmax & argmax to become predictions.

        Example:

        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hid_dim),
            nonlin_type(nl_type),
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nonlin_type(nl_type),
            nn.Linear(in_features=hid_dim, out_features=out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class TinyVGG_CIFAR(nn.Module):
    """
    Replicate TinyVGG from CNN Explainer website - https://poloclub.github.io/cnn-explainer/
    To apply it to multiclass classification on CIFAR dataset.

    Network has 2 convolution blocks (Conv2d -> Relu -> Conv2d -> Relu -> MaxPool)
        followed by 1 classifier block, which is just a linear layer.

    :args:
        :input_shape: (int) -
        :hidden_units: (int) -
        :output_shape: (int) -
        :x: (Tensor) - input image of shape [batch, colorchans, height, width]
        :verbose: (bool) - Flag to enter verbose mode and print out shape of data after each block

    :returns:
        :y_logits: (Tensor) - raw outputs from model must be converted into probs (by softmax) and preds (by argmax)
    """

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        #
        super().__init__()
        #
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        #
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        #
        self.classifier_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 8 * 8,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:

        x = self.conv_block1(x)
        if verbose: print(f"After conv_block1, shape is: {x.shape}")
        x = self.conv_block2(x)
        if verbose: print(f"After conv_block2, shape is: {x.shape}")
        x = self.classifier_block(x)
        if verbose: print(f"After classifier_block, shape is: {x.shape}")

        return x

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #