import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from pdb import set_trace


show_plots=False

# Plotting
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# CIFAR
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
	])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
	transform = transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
	transform = transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
	shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
	shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

examples = iter(train_loader)
images, labels = examples.next()

# show images
if show_plots: imshow(torchvision.utils.make_grid(images))


# Set up CNN layers
conv1 = nn.Conv2d( 3, 6, 5 ) # input_size=color chan, output_size, kernel_size
pool = nn.MaxPool2d( 2, 2 ) # kernel_size, stride 
conv2 = nn.Conv2d( 6, 16, 5 ) #



# Send images through CNN and look at sizes after each layer.
print(f'Image size = {images.shape}')
x = conv1(images)
print(f'After conv1, size = {x.shape}')
if show_plots: imshow(torchvision.utils.make_grid(x[:,:3,:,:]))
x = pool(x)
print(f'After pool1, size = {x.shape}')
if show_plots: imshow(torchvision.utils.make_grid(x[:,:3,:,:]))
x = conv2(x)
print(f'After conv1, size = {x.shape}')
if show_plots: imshow(torchvision.utils.make_grid(x[:,:3,:,:]))
x = pool(x)
print(f'After pool2, size = {x.shape}')
if show_plots: imshow(torchvision.utils.make_grid(x[:,:3,:,:]))





