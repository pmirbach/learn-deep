# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:03:41 2018

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
import time


flag_show_dataset_sample = 1

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


#transforms.Lambda()
#
#
#
#transform_new = transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize(mean=(0.5, ), std=(0.5, ))])
#

#import torch.utils.data as data
#
#
#class MyDataSet(data.Dataset):
#    
#    def __init__(self, train=True):
#        self.MNIST = torchvision.datasets.MNIST(root=root, train=train, download=True)
#        self.train = train
#    
#    def __getitem__(self, index):
#        if self.train:
#            img, target = self.MNIST.train_data[index], self.MNIST.train_labels[index]
#        else:
#            img, target = self.MNIST.test_data[index], self.MNIST.test_labels[index]
#        
#        img_2 = img
#        print(type(img), img.size())
#        
##        img = img.ToTensor()
##        img_2 = transforms.ToTensor(img_2)
#        
#        return [img, img_2], target
#    
#    def __len__(self):
#        if self.train:
#            return len(self.MNIST.train_data)
#        else:
#            return len(self.MNIST.test_data)
#
#
#train_set_2 = MyDataSet(train=True)
#train_loader_2 = torch.utils.data.DataLoader(dataset=train_set_2, batch_size=4, shuffle=True, num_workers=0)
#
#
#
#if flag_show_dataset_sample:
#    def imshow(img, ax):
#        img = img / 2 + 0.5
#        npimg = img.numpy()
#        ax.imshow(np.transpose(npimg, (1, 2, 0)))
#    
#    data_iter = iter(train_loader_2)
#    images, labels = data_iter.next()
#    
#    print(len(images))
#    print(images.shape)
#    
#    fig, ax = plt.subplots()
#    imshow(torchvision.utils.make_grid(images), ax)
#    fig.text(.5, .05, labels, ha='center')
#    plt.show()
#



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
        x = np.fft.fft2(x)
        x = torch.from_numpy(x)
        
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
            


def train_Net(Net, optim_method='SGD', lr=1e-3, N_epoch=5):

    criterion = nn.CrossEntropyLoss()
    
    if optim_method == 'SGD':    
        optimizer = torch.optim.SGD(Net.parameters(), lr=lr, momentum=0.9)
    elif optim_method == 'Adam':
        optimizer = torch.optim.Adam(Net.parameters(), lr=lr)
    
    print(optimizer)
    
    start_time_0 = time.time()
    time_epoch = np.zeros((N_epoch, ))
    
    print('Start Training')
    for epoch in range(N_epoch):
        start_time_epoch = time.time()
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
        time_epoch[epoch] = time.time() - start_time_epoch
        time_estimate = (N_epoch - (epoch + 1)) * np.mean(time_epoch[time_epoch.nonzero()])
        print('Estimated time remaining: {0:5.1f} seconds'.format(time_estimate))
    
    print('Finished Training - Duration: {0:5.1f} seconds'.format(time.time() - start_time_0))


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
    
    print('Accuracy of the network on the 2500 test images: {} %'.format(100 * correct / total))
    return correct / total








Net = ConvNet(N_conv1_out=8, N_conv1_kernel=4, N_conv2_out=32, N_conv2_kernel=4)
print(Net)
train_Net(Net, optim_method='SGD', lr=4e-3, N_epoch=5)

test_Net(Net)


