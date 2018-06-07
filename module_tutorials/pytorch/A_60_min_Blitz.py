#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:52:37 2018

@author: pmirbach
"""

#from __future__ import print_function
import torch

#x = torch.rand(5,3)
#x = torch.zeros((5, 3), dtype=torch.long)
#x = torch.tensor([5.5, 3])
#
#print(x)
#
#x = x.new_ones(5, 3, dtype=torch.double)
#print(x)
#x = torch.randn_like(x, dtype=torch.float)
#print(x)
#
#print(x.size())


#x = torch.randn(5,3)
#y = torch.randn_like(x)
#
#z = x + y
#z2 = torch.add(x, y)
#
#result = torch.empty(5, 3)
#torch.add(x, y, out=result)
#
#y.add_(x)
#
#print(y)


#x = torch.randn(5,3)
#print(x[0, :])


#x = torch.randn((4, 4))
#y = x.view((16))
#z = x.view((-1, 8))
#
#print(x, x.size())
#print(y, y.size())
#print(z, z.size())


x = torch.randn((1))
print(x)
print(x.item())















