import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import wandb

from pdb import set_trace






# Define CNN - note: see cnnSizeTest to fit layer sizes.
class ConvNet(nn.Module):
	def __init__(self, color_chans, img_size, conv_kernel_size): 
		super(ConvNet,self).__init__()
		self.conv1 = nn.Conv2d( color_chans, 6, conv_kernel_size, stride=1, padding=2 ) # input_size=color chan, output_size, kernel_size
		self.pool = nn.MaxPool2d( 2, 2 ) # kernel_size, stride 
		self.conv2 = nn.Conv2d( 6, 16, conv_kernel_size, stride=1, padding=0 ) #
		self.fc1 = nn.Linear( 16*conv_kernel_size*conv_kernel_size, 120 )
		self.fc2 = nn.Linear( 120, 84)
		self.fc3 = nn.Linear( 84, 10)

	def forward(self,x):
		#print(x.shape)
		x = self.pool(F.relu(self.conv1(x)))
		#print(x.shape)
		x = self.pool(F.relu(self.conv2(x)))	
		#print(x.shape)
		x = x.view(-1, 16*conv_kernel_size*conv_kernel_size) # flatten
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# Plotting
def imshow(img):
    img = img # / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



use_tensorboard = False
use_wandb = True

which_dataset = 'MNIST' # CIFAR10, FashionMNIST, MNIST, KMNIST, QMNIST

save_model = True

# hyperparameters
num_epochs = 8
batch_size = 4
learning_rate = 0.001




# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # Changes 3-chan color to Grayscale image.
	])




# Load in Dataset (for CNNs: CIFAR10, FashionMNIST, ImageNet) 	(also can try: MNIST, KMNIST, QMNIST)
if which_dataset=='CIFAR10':
	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform = transform, download=True)
	test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform = transform, download=True)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
elif which_dataset=='FashionMNIST':
	train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform = transforms.ToTensor(), download=True)
	test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform = transforms.ToTensor(), download=True)
elif which_dataset=='ImageNet':
	print('NOTE: ImageNet dataset is too large (1TB) to mess with. Also, its no longer publically available. Look on https://image-net.org')
	#train_dataset = torchvision.datasets.ImageNet(root='./data', train=True, transform = transform, download=True)
	#test_dataset = torchvision.datasets.ImageNet(root='./data', train=False, transform = transform, download=True)
elif which_dataset=='MNIST':
	train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform = transforms.ToTensor(), download=True)
	test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform = transforms.ToTensor(), download=True)
	classes = ('0','1','2','3','4','5','6','7','8','9')
elif which_dataset=='KMNIST':
	train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, transform = transforms.ToTensor(), download=True)
	test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, transform = transforms.ToTensor(), download=True)
elif which_dataset=='QMNIST':
	train_dataset = torchvision.datasets.QMNIST(root='./data', train=True, transform = transforms.ToTensor(), download=True)
	test_dataset = torchvision.datasets.QMNIST(root='./data', train=False, transform = transforms.ToTensor(), download=True)	
	classes = ('0','1','2','3','4','5','6','7','8','9')
else:
	print('Dont understand Dataset.')	








train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False)



examples = iter(train_loader)
images, labels = examples.next()

color_chans = images.shape[1]
img_size = images.shape[-2:]

print(labels)

# show images
if False:
	imshow(torchvision.utils.make_grid(images))


# tensorboard
if use_tensorboard: 
	writer = SummaryWriter(f'runs/CNN_{which_dataset}')
	print('Using Tensorboard. Run this in command line to display results in web browser. > tensorboard --logdir=runs')


# wandb
if use_wandb:
	#
	config = dict(
		num_epochs=num_epochs,
		batch_size=batch_size,
		learning_rate=learning_rate)
	#
	wandb.init(project=f'CNN_{which_dataset}', config=config)


# Send images to tensorboard.
if use_tensorboard: 
	img_grid = torchvision.utils.make_grid(samples)
	writer.add_image(f'{which_dataset}_images',img_grid)


#set_trace()

conv_kernel_size=5
model = ConvNet(color_chans, img_size, conv_kernel_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0

for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)

		#forward
		outputs = model(images)
		loss = criterion(outputs,labels)

		#backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# calculate running loss and correct predictions
		_,predictions = torch.max(outputs,1) # returns value, index.
		running_correct += (predictions == labels).sum().item()
		running_loss += loss.item()

		if (i+1)%2000 == 0:
			print(f'epoch{epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

			# write training loss and accuracy to tensorboard.	
			if use_tensorboard:
				writer.add_scalar('training_loss', running_loss/100, epoch*n_total_steps + i)
				writer.add_scalar('accuracy', running_correct/100, epoch*n_total_steps + i)

			if use_wandb:
				wandb.log({'epoch':epoch*n_total_steps + i, 'training_loss':running_loss/100, 
					'accuracy':running_correct/100})
				wandb.watch(model, criterion, log='all')
					
			running_correct = 0
			running_loss = 0.0

print('Finished Training')
if save_model:
	PATH = './cnn.pth'
	torch.save(model.state_dict(), PATH)


# test
with torch.no_grad():
	n_correct = 0
	n_samples = 0 
	n_class_correct = [0 for i in range(10)]
	n_class_samples = [0 for i in range(10)]

	for images,labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)	
		outputs = model(images)

		_,predictions = torch.max(outputs,1) # returns value, index.
		n_samples += labels.shape[0]
		n_correct += (predictions == labels).sum().item()

		for i in range(batch_size):
			label = labels[i]
			pred = predictions[i]
			if (label==pred):
				n_class_correct[label] +=1
			n_class_samples[label] +=1	



acc = 100.0 * n_correct/n_samples
print(f'Accuracy of network = {acc} %')	

for i in range(10):
	acc = 100.0 * n_class_correct[i]/n_class_samples[i]
	print(f'Accuracy of {classes[i]} = {acc} %')








