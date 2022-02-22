import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from pdb import set_trace


# Define CNN - note: see cnnSizeTest to fit layer sizes.
class ConvNet(nn.Module):
	def __init__(self): 
		super(ConvNet,self).__init__()
		self.conv1 = nn.Conv2d( 3, 6, 5 ) # input_size=color chan, output_size, kernel_size
		self.pool = nn.MaxPool2d( 2, 2 ) # kernel_size, stride 
		self.conv2 = nn.Conv2d( 6, 16, 5 ) #
		self.fc1 = nn.Linear( 16*5*5, 120 )
		self.fc2 = nn.Linear( 120, 84)
		self.fc3 = nn.Linear( 84, 10)

	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))	
		x = x.view(-1, 16*5*5) # flatten
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# Plotting
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 2
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
#imshow(torchvision.utils.make_grid(images))

model = ConvNet().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
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

		if (i+1)%2000 == 0:
			print(f'epoch{epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

print('Finished Training')
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








