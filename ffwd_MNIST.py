import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import wandb

import torch.nn.functional as F

import sys

from pdb import set_trace


# Define 2 layer feed forward model.
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet,self).__init__()
		self.l1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(hidden_size,num_classes)

	def forward(self,x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)	
		return out



use_tensorboard = False
use_wandb = True


# hyperparameters
input_size = 28*28 # images dims flattened.
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 64
learning_rate = 0.001

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Load in torchvision dataset: 	(to try: MNIST, KMNIST, QMNIST) (for CNNs: CIFAR10, FashionMNIST, ImageNet)
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform = transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform = transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()

print(samples.shape, labels.shape)

for i in range(6):
	plt.subplot(2,3,i+1)
	plt.imshow(samples[i][0],cmap='gray')
#plt.show()


# tensorboard
if use_tensorboard: 
	writer = SummaryWriter(f'runs/QMNIST')


# wandb
if use_wandb:
	#wandb.login()	
	#
	config = dict(
		input_size=input_size, 
		hidden_size=hidden_size,
		num_classes=num_classes,
		num_epochs=num_epochs,
		batch_size=batch_size,
		learning_rate=learning_rate)
	#
	wandb.init(project='ffwd_MNIST', config=config)





# Send images to tensorboard.
if use_tensorboard: 
	img_grid = torchvision.utils.make_grid(samples)
	writer.add_image('MNIST_images',img_grid)


#set_trace()


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


# Send model graph to tensorboard
if use_tensorboard: writer.add_graph(model, samples.reshape(-1,28*28))





# training loop
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0

for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.reshape(-1, input_size).to(device)
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


		if (i+1)%100 == 0:
			print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

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


# test
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	labelsB = []
	preds = []

	for images,labels in test_loader:
		images = images.reshape(-1, input_size).to(device)
		labels = labels.to(device)	
		outputs = model(images)

		_,predictions = torch.max(outputs,1) # returns value, index.
		n_samples += labels.shape[0]
		n_correct += (predictions == labels).sum().item()

		class_predictions = [F.softmax(output,dim=0) for output in outputs]

		preds.append(class_predictions)
		labelsB.append(predictions)

	preds = torch.cat([torch.stack(batch) for batch in preds])
	labelsB = torch.cat(labelsB)
	

	acc = 100.0 * n_correct/n_samples
	print(f'Accuracy = {acc}')	

	if use_tensorboard:
		classes = range(10)
		for i in classes:
			labels_i = labelsB==i
			preds_i = preds[:,i]
			writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
			writer.close()









