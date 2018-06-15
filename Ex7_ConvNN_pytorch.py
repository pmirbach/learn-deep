# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 23:35:14 2018

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
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.data_shape = data_shape
        self.conv1 = nn.Conv2d(in_channels=data_channels, out_channels=6, kernel_size=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                
        data_shape_conv = self._get_layer_output_size([self.conv1, self.pool1, self.conv2, self.pool2])
        self.NN_in_features = data_shape_conv[0] * data_shape_conv[1] * 16
#        print(data_shape_conv, self.NN_in_features)
        
        self.fc1 = nn.Linear(in_features=self.NN_in_features, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
#        print(x.size())   
#        x = self.conv1(x)
#        print(x.size())
#        x = F.relu(x)
#        print(x.size())
#        x = self.pool1(x)
#        print(x.size())
#        
#        x = self.conv2(x)
#        print(x.size())
#        x = F.relu(x)
#        print(x.size())
#        x = self.pool2(x)
#        print(x.size())
        x = self.pool1(F.relu(self.conv1(x)))
#        print(x.size())
        x = self.pool2(F.relu(self.conv2(x)))
#        print(x.size())
        x = x.view(-1, self.NN_in_features)
#        print(x.size())
        x = F.relu(self.fc1(x))
#        print(x.size())
        x = F.relu(self.fc2(x))
#        print(x.size())
        x = self.fc3(x)
#        print(x.size())
#        raise('STOP')
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
            
Net = ConvNet()
print(Net)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)



for epoch in range(5):
    
    running_loss = 0
    for i, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = Net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2500 == 2499:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2500))
            running_loss = 0.0

print('Finished Training')        

















































