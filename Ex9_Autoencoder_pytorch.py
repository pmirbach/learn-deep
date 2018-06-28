# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:27:43 2018

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

print('==>>> total training batch number: {}'.format(len(train_loader)))
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


class Autoencoder(nn.Module):
    
    def __init__(self, N_enc_1=256, N_enc_2=64, N_dec_1=256):
        super(Autoencoder, self).__init__()
        self.data_shape = data_shape
        self.N_input= data_shape[0] * data_shape[1]
        
        # Encoder
        self.enc_fc1 = nn.Linear(in_features=self.N_input, out_features=N_enc_1)
        self.enc_fc2 = nn.Linear(in_features=N_enc_1, out_features=N_enc_2)
        
        # Decoder
        self.dec_fc1 = nn.Linear(in_features=N_enc_2, out_features=N_dec_1)
        self.dec_fc2 = nn.Linear(in_features=N_dec_1, out_features=self.N_input)
        
    def forward(self, x):
        x = x.view(-1, self.N_input)
        x = self.encoder(x)
        y = self.decoder(x)
#        print(y.size())
        y = y.view(-1, self.data_shape[0], self.data_shape[1])
        return y
    
    def encoder(self, x):
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        return x
    
    def decoder(self, y):
        y = F.relu(self.dec_fc1(y))
        y = F.relu(self.dec_fc2(y))
        return y
    



        
def train_Net(Net, optim_method='SGD', lr=1e-3, N_epoch=5):

    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(Net.parameters(), lr=lr, weight_decay=1e-5)
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
            loss = criterion(outputs, data)
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
    
    print('Accuracy of the network on the 2500 test images: {} %%'.format(100 * correct / total))
    return correct / total


Net = Autoencoder()
train_Net(Net)
test_Net(Net)
















