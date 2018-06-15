# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 22:50:46 2018

@author: Philip
"""

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import os


flag_show_dataset_sample = 0

sep = '\n{}\n'.format('-'*70)

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, ))])

train_set = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
test_set = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=4, shuffle=False, num_workers=0)

(data_channels, data_height, data_width) = list(train_set[0][0].size())
data_shape = (data_height, data_width)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))
print('==>>> Data properties:\n  >>> Number channels: {}, Data (height x width): ({} x {})'
      .format(data_channels, data_height, data_width))

print(sep)


if flag_show_dataset_sample:
    def imshow(img, ax):
        img = img / 2 + 0.5
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
    
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    
    fig, ax = plt.subplots()
    imshow(torchvision.utils.make_grid(images), ax)
    fig.text(.5, .05, labels, ha='center')
    plt.show()


class ConvNet(nn.Module):
    
    def __init__(self, N_conv1_out=6, N_conv1_kernel=4, N_conv2_out=16, N_conv2_kernel=5):
        super(ConvNet, self).__init__()
        self.data_shape = data_shape
        self.conv1 = nn.Conv2d(in_channels=data_channels, out_channels=N_conv1_out, kernel_size=N_conv1_kernel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=N_conv1_out, out_channels=N_conv2_out, kernel_size=N_conv2_kernel)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                
        data_shape_conv = self._get_layer_output_size([self.conv1, self.pool1, self.conv2, self.pool2])
        self.NN_in_features = data_shape_conv[0] * data_shape_conv[1] * N_conv2_out
#        print(data_shape_conv, self.NN_in_features)
        
        self.fc1 = nn.Linear(in_features=self.NN_in_features, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.NN_in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _get_layer_output_size(self, layers):
        data_shape_new = list(data_shape)
        for layer in layers:
            F = layer.kernel_size
            P = layer.padding
            S = layer.stride
            F = [F, F] if (type(F) == int) else F
            P = [P, P] if (type(P) == int) else P
            S = [S, S] if (type(S) == int) else S            
            for i in range(2):
                data_shape_new[i] = int((data_shape_new[i] - F[i] + 2 * P[i]) / S[i] + 1)
        return data_shape_new
            
#Net = ConvNet(N_conv1_out=10, N_conv1_kernel=6, N_conv2_out=34, N_conv2_kernel=6)
#print(Net)



def train_Net(Net, optim_method='SGD', lr=1e-3, N_epoch=5):

    criterion = nn.CrossEntropyLoss()
    
    if optim_method == 'SGD':    
        optimizer = torch.optim.SGD(Net.parameters(), lr=lr, momentum=0.9)
    elif optim_method == 'Adam':
        optimizer = torch.optim.Adam(Net.parameters(), lr=lr)
    
    print(optimizer)
    
    print('Start Training')
    for epoch in range(N_epoch):
        running_loss = 0
        for i, data in enumerate(train_loader, start=0):           
            inputs, labels = data
            optimizer.zero_grad()
            outputs = Net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 2500 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2500))
                running_loss = 0
    print('Finished Training')


def test_Net(Net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = Net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 2500 test images: {} %%'.format(100 * correct / total))
    return correct / total
    

def bitlist_to_int(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out.item()



class Chromosome():
    
    def __init__(self, x):
        self.x = x
        (self.N_conv1_out, self.N_conv1_kernel, self.N_conv2_out, self.N_conv2_kernel,
         self.optim_typ, self.lr) = self._mapping(x)
        
        self.Net = ConvNet(N_conv1_out=self.N_conv1_out,  N_conv1_kernel=self.N_conv1_kernel,
                           N_conv2_out=self.N_conv2_out, N_conv2_kernel=self.N_conv2_kernel)
        print(self.Net)
        
        train_Net(Net=self.Net, optim_method=self.optim_typ, lr=self.lr, N_epoch=5)
        self.fitness = test_Net(Net=self.Net)


    def _mapping(self, x):
        N_conv1_out = bitlist_to_int(x[:3]) + 1             # (1,8)
        N_conv1_kernel = bitlist_to_int(x[3:5]) + 2         # (2,5)
        N_conv2_out = bitlist_to_int(x[5:10]) + 1           # (1,32)
        N_conv2_kernel = bitlist_to_int(x[10:12]) + 2       # (2,5)
        
        optim_typ = 'SGD' if (x[12] == 0) else 'Adam'
        lr = (bitlist_to_int(x[13:]) + 1) * 1e-3            # (1e-3, 8e-3)
        return (N_conv1_out, N_conv1_kernel, N_conv2_out, N_conv2_kernel, optim_typ, lr)


def mutation(x, sigma):
    x_mut = [(bit+1)%2 if (np.random.random() < sigma) else bit for bit in x]
    return x_mut



def GA(N_cycles=10):
    sigma = 4 / 16
    
    x_ancestor = np.random.choice([0,1], size=16)    
    parent = Chromosome(x_ancestor)
    print(sep)
    
    for _ in range(N_cycles):
        x_child = mutation(parent.x, sigma=sigma)
        child = Chromosome(x_child)
        
        if child.fitness > parent.fitness:
            parent = child
        print(sep)
    

GA(N_cycles=30)



























