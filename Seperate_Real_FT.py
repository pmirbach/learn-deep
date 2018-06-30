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

#transform = torchvision.transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize(mean=(0.5, ), std=(0.5, ))])
#
#train_set = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
#test_set = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)
#
#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4, shuffle=True, num_workers=0)
#test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=4, shuffle=False, num_workers=0)
#
#(data_channels, data_height, data_width) = list(train_set[0][0].size())
#data_shape = (data_height, data_width)
#
#print('==>>> total trainning batch number: {}'.format(len(train_loader)))
#print('==>>> total testing batch number: {}'.format(len(test_loader)))
#print('==>>> Data properties:\n  >>> Number channels: {}, Data (height x width): ({} x {})'
#      .format(data_channels, data_height, data_width))
#
#print(sep)


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








#Net = ConvNet(N_conv1_out=8, N_conv1_kernel=4, N_conv2_out=32, N_conv2_kernel=4)
#print(Net)
#train_Net(Net, optim_method='SGD', lr=4e-3, N_epoch=5)
#
#test_Net(Net)


from torch.utils.data import Dataset
from keras.datasets import mnist

class MNIST_FT(Dataset):
    
    def __init__(self, root_dir=None, train=True, transform=None):
        
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        
        x_train_ft = np.fft.fft2(self.x_train)
        x_test_ft = np.fft.fft2(self.x_test)
        
        self.x_train_ft_real = np.real(x_train_ft)
        self.x_train_ft_imag = np.imag(x_train_ft)

        self.x_test_ft_real = np.real(x_test_ft)
        self.x_test_ft_imag = np.imag(x_test_ft)
        
    def __len__(self):
        return self.x_train.shape[0]
    
    def __getitem__(self, idx):
        if self.train:
            img = self.x_train[idx,:,:]
            img_ft_real = self.x_train_ft_real[idx,:,:]
            img_ft_imag = self.x_train_ft_imag[idx,:,:]
            label = self.y_train[idx]
        else:
            img = self.x_test[idx,:,:]
            img_ft_real = self.x_test_ft_real[idx,:,:]
            img_ft_imag = self.x_test_ft_imag[idx,:,:]
            label = self.y_test[idx]
        
        sample = {'real': img, 'ft_real': img_ft_real, 
                  'ft_imag': img_ft_imag, 'label': label}
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        (img, ft_real, ft_imag) = (sample['real'], sample['ft_real'], sample['ft_imag'])
        
        sample['real'] = torch.from_numpy(img)
        sample['ft_real'] = torch.from_numpy(ft_real)
        sample['ft_imag'] = torch.from_numpy(ft_imag)
        
        return sample

     
transform = ToTensor()



train_set = MNIST_FT(train=True, transform=transform)
test_set = MNIST_FT(train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0)


print(train_set.__len__())

flag_show_dataset_sample = 1
if flag_show_dataset_sample:
    N_samples = 4
    train_iter = iter(train_loader)
    fig, axes = plt.subplots(nrows=N_samples, ncols=3)
    for i in range(N_samples):
        sample = train_iter.next()
        
        img = sample['real'].numpy()
        print(type(img), img.shape)
        
        axes[i,0].imshow(sample['real'].numpy())
        axes[i,1].imshow(sample['ft_real'].numpy())
        axes[i,2].imshow(sample['ft_imag'].numpy())    
    plt.show()



class CNN(nn.Module):
    
    def __init__(self, sample_channel):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=180, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        
        self.sample_channel = sample_channel
        
    def forward(self, x):
        x = x[self.sample_channel]
        x = self.pool1(F.relu(self.conv1(x)))
        x = x.view(-1, self.NN_in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward_part(self, x):
        x = x[self.sample_channel]
        x = self.pool1(F.relu(self.conv1(x)))
        x = x.view(-1, self.NN_in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x        







