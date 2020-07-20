#!/usr/bin/env python
# coding: utf-8

# # CIFAR10 Federated resnet20 Client Side
# This code is the server part of CIFAR10 federated resnet20 for **multi** client and a server.

# In[3]:


users = 1 # number of clients


# In[4]:


import os
import h5py

import socket
import struct
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import time

from tqdm import tqdm


# In[2]:


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


# In[3]:


printPerformance()


# In[5]:


root_path = '../../models/cifar10_data'


# ## Cuda

# In[6]:


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)


# In[7]:


client_order = int(input("client_order(start from 0): "))


# In[8]:


num_traindata = 50000 // users


# ## Data load

# In[9]:


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

from torch.utils.data import Subset


indices = list(range(50000))

part_tr = indices[num_traindata * client_order : num_traindata * (client_order + 1)]


# In[10]:


trainset = torchvision.datasets.CIFAR10 (root=root_path, train=True, download=True, transform=transform)

trainset_sub = Subset(trainset, part_tr)

train_loader = torch.utils.data.DataLoader(trainset_sub, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10 (root=root_path, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


# In[11]:


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ### Number of total batches

# In[13]:


train_total_batch = len(train_loader)
print(train_total_batch)
test_batch = len(test_loader)
print(test_batch)


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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

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
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

# In[15]:


res_net = resnet20()
res_net.to(device)


# In[16]:


lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res_net.parameters(), lr=lr, momentum=0.9)

rounds = 400 # default
local_epochs = 1 # default


# ## Socket initialization
# ### Required socket functions

# In[17]:


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


# In[15]:


printPerformance()


# ### Set host address and port number

# In[18]:


host = input("IP address: ")
port = 10080
max_recv = 100000


# ### Open the client socket

# In[19]:


s = socket.socket()
s.connect((host, port))


# ## SET TIMER

# In[20]:


start_time = time.time()    # store start time
print("timmer start!")


# In[21]:


msg = recv_msg(s)
rounds = msg['rounds'] 
client_id = msg['client_id']
local_epochs = msg['local_epoch']
send_msg(s, len(trainset_sub))


# In[22]:


# update weights from server
# train
for r in range(rounds):  # loop over the dataset multiple times

 
    
    weights = recv_msg(s)
    res_net.load_state_dict(weights)
    res_net.eval()
    for local_epoch in range(local_epochs):
        
        for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Round '+str(r+1)+'_'+str(local_epoch+1))):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.clone().detach().long().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = res_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    msg = res_net.state_dict()
    send_msg(s, msg)

print('Finished Training')


# In[ ]:


printPerformance()


# In[23]:


end_time = time.time()  #store end time
print("Training Time: {} sec".format(end_time - start_time))


# In[ ]:




