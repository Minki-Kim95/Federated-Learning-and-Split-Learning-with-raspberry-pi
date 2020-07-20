#!/usr/bin/env python
# coding: utf-8

# # CIFAR10 Split Resnet20 Client Side
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


# In[3]:


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


# ## Define ECG client model
# Client side has only **2 convolutional layers**.

# In[15]:


from torch.autograd import Variable
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             if option == 'A':
#                 """
#                 For CIFAR10 ResNet paper uses option A.
#                 """
#                 self.shortcut = LambdaLayer(lambda x:
#                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
#             elif option == 'B':
#                 self.shortcut = nn.Sequential(
#                      nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                      nn.BatchNorm2d(self.expansion * planes)
#                 )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [1, 3, 3])


# In[16]:


resnet20_client = resnet20().to(device)
print(resnet20_client)


# In[17]:


# from torchsummary import summary

# summary(resnet20_client, (3, 32, 32))


# ### Set other hyperparameters in the model
# Hyperparameters here should be same with the server side.

# In[18]:


epoch = 20  # default
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet20_client.parameters(), lr=lr, momentum=0.9)


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
    resnet20_client.load_state_dict(client_weights)
    resnet20_client.eval()
    for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Epoch '+str(e+1))):
        x, label = data
        x = x.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = resnet20_client(x)
        client_output = output.clone().detach().requires_grad_(True)
        msg = {
            'client_output': client_output,
            'label': label
        }
        send_msg(s, msg)
        client_grad = recv_msg(s)
        output.backward(client_grad)
        optimizer.step()
    send_msg(s, resnet20_client.state_dict())


# In[ ]:


printPerformance()


# In[25]:


end_time = time.time()  #store end time
print("WorkingTime of ",device ,": {} sec".format(end_time - start_time))


# In[ ]:




