#!/usr/bin/env python
# coding: utf-8

# # CIFAR10 Split Mobilenet Client Side
# This code is the server part of CIFAR10 split 2D-CNN model for **multi** client and a server.

# In[4]:


users = 1 # number of clients


# ## Import required packages

# In[5]:


import os
import struct
import socket
import pickle
import time

import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


# In[ ]:


def getFreeDescription():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 1:
            return (line.split()[0:7])


def getFree():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 2:
            return (line.split()[0:7])

from gpiozero import CPUTemperature


def printPerformance():
    cpu = CPUTemperature()

    print("temperature: " + str(cpu.temperature))

    description = getFreeDescription()
    mem = getFree()

    print(description[0] + " : " + mem[1])
    print(description[1] + " : " + mem[2])
    print(description[2] + " : " + mem[3])
    print(description[3] + " : " + mem[4])
    print(description[4] + " : " + mem[5])
    print(description[5] + " : " + mem[6])


# In[ ]:


printPerformance()


# In[6]:


root_path = '../../models/cifar10_data'


# ## SET CUDA

# In[7]:


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.manual_seed(777)
if device =="cuda:0":
    torch.cuda.manual_seed_all(777)


# In[8]:


client_order = int(input("client_order(start from 0): "))


# In[9]:


num_traindata = 50000 // users


# ## Data load

# In[10]:


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

from torch.utils.data import Subset


indices = list(range(50000))

part_tr = indices[num_traindata * client_order : num_traindata * (client_order + 1)]


# In[11]:


trainset = torchvision.datasets.CIFAR10 (root=root_path, train=True, download=True, transform=transform)

trainset_sub = Subset(trainset, part_tr)

train_loader = torch.utils.data.DataLoader(trainset_sub, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10 (root=root_path, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


# ### Size check

# In[13]:


x_train, y_train = next(iter(train_loader))
print(x_train.size())
print(y_train.size())


# ### Total number of batches

# In[14]:


total_batch = len(train_loader)
print(total_batch)


# In[15]:


# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:23:31 2018
@author: tshzzz
"""

import torch
import torch.nn as nn


def conv_dw_client(inplane,stride=1):
    return nn.Sequential(
        nn.Conv2d(inplane,inplane,kernel_size = 3,groups = inplane,stride=stride,padding=1),
        nn.BatchNorm2d(inplane),
        nn.ReLU()  
        )

def conv_bw(inplane,outplane,kernel_size = 3,stride=1):
    return nn.Sequential(
        nn.Conv2d(inplane,outplane,kernel_size = kernel_size,groups = 1,stride=stride,padding=1),
        nn.BatchNorm2d(outplane),
        nn.ReLU() 
        )


class MobileNet(nn.Module):
    
    def __init__(self,num_class=10):
        super(MobileNet,self).__init__()
        
        layers = []
        layers.append(conv_bw(3,32,3,1))
        layers.append(conv_dw_client(32,1))
#         layers.append(conv_dw(64,128,2))
#         layers.append(conv_dw(128,128,1))
#         layers.append(conv_dw(128,256,2))
#         layers.append(conv_dw(256,256,1))
#         layers.append(conv_dw(256,512,2))

#         for i in range(5):
#             layers.append(conv_dw(512,512,1))
#         layers.append(conv_dw(512,1024,2))
#         layers.append(conv_dw(1024,1024,1))

#         self.classifer = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(1024,num_class)
#                 )
        self.feature = nn.Sequential(*layers)
        
        

    def forward(self,x):
        out = self.feature(x)
#         out = out.mean(3).mean(2)
#         out = out.view(-1,1024)
#         out = self.classifer(out)
        return out


# In[16]:


mobilenet_client = MobileNet().to(device)
print(mobilenet_client)


# In[17]:


# from torchsummary import summary

# summary(mobilenet_client, (3, 32, 32))


# ### Set other hyperparameters in the model
# Hyperparameters here should be same with the server side.

# In[18]:


epoch = 20  # default
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mobilenet_client.parameters(), lr=lr, momentum=0.9)


# ## Socket initialization

# ### Required socket functions

# In[19]:


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


# In[ ]:


printPerformance()


# ### Set host address and port number

# In[20]:


host = input("IP address: ")
port = 10080


# ## SET TIMER

# In[21]:


start_time = time.time()    # store start time
print("timmer start!")


# ### Open the client socket

# In[22]:


s = socket.socket()
s.connect((host, port))


# In[23]:


epoch = recv_msg(s)   # get epoch
msg = total_batch
send_msg(s, msg)   # send total_batch of train dataset


# ## Real training process

# In[24]:


for e in range(epoch):
    client_weights = recv_msg(s)
    mobilenet_client.load_state_dict(client_weights)
    mobilenet_client.eval()
    for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Epoch '+str(e+1))):
        x, label = data
        x = x.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = mobilenet_client(x)
        client_output = output.clone().detach().requires_grad_(True)
        msg = {
            'client_output': client_output,
            'label': label
        }
        send_msg(s, msg)
        client_grad = recv_msg(s)
        output.backward(client_grad)
        optimizer.step()
    send_msg(s, mobilenet_client.state_dict())


# In[ ]:


printPerformance()


# In[25]:


end_time = time.time()  #store end time
print("WorkingTime of ",device ,": {} sec".format(end_time - start_time))


# In[ ]:




