import torch
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

'''This file contains the different models for the experiments of Feature Sieve Regression
 The models are pretrained and as follows:
 1) Resnet34 (CNN) 
 2) Margin Forgetting Loss Model
 3) Cross Entropy Forgetting Loss Model
 4) Ordinal Forgetting Loss Model'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


## Resnet Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

######################################################################
class Resnet34(nn.Module):
    def __init__(self, num_classes):
        super(Resnet34, self).__init__()
        resnet = models.resnet34(weights = 'IMAGENET1K_V1')
        for param in resnet.parameters():
            param.requires_grad = True
            
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
#######################################################################

class SiFer_margin(nn.Module):
    def __init__(self,  input_shape, num_classes, aux_pos = 1, aux_kernels = [128, 128], layers = [3, 4], block = BasicBlock):
        super(SiFer_margin, self).__init__()
        self.num_classes = num_classes
        self.aux_pos = aux_pos + 4
        # main network
        network = models.resnet34(weights = 'IMAGENET1K_V1')
        for param in network.parameters():
            param.requires_grad = True
        
        self.layers = nn.ModuleList(list(network.children())[:-1])
        self.layers.append(nn.Linear(512, num_classes))

        # Auxiliary Network
        self.aux_layers = nn.ModuleList([])
        
        self.inplanes = 128
        for kernel_id in range(len(aux_kernels)):
            self.aux_layers.append(self.__make_layer(block, aux_kernels[kernel_id], layers[kernel_id], stride = 2))
            
        self.aux_layers.append(nn.AvgPool2d(7, stride=1, padding=2))

        # output size calculation
        with torch.no_grad():
            outputs_main = []
            x = torch.zeros(input_shape)
            for layer in self.layers[:-1]:
                x = layer(x)
                outputs_main.append(x)
    
            x = x.view(x.shape[0], -1)
            x = self.layers[-1](x)
            outputs_main.append(x)
    
            # forward for the aux network
            aux = outputs_main[self.aux_pos]
            for aux_layer in self.aux_layers:
                aux = aux_layer(aux)
            aux = aux.reshape((aux.shape[0], -1))
        
        self.aux_layers.append(nn.Linear(aux.shape[-1], num_classes))

        # parameters dict
        self.params = nn.ModuleDict({
            "main" : self.layers,
            "aux"  : self.aux_layers,
            'forget': self.layers[:self.aux_pos]
        })
        
    def forward(self, x):
        # forward for the main network
        outputs_main = []
        
        for layer in self.layers[:-1]:
            x = layer(x)
            outputs_main.append(x)

        x = x.view(x.shape[0], -1)
        x = self.layers[-1](x)
        outputs_main.append(x)

        # forward for the aux network
        aux = outputs_main[self.aux_pos]
        for aux_layer in self.aux_layers[:-1]:
            aux = aux_layer(aux)

        aux = aux.view(aux.shape[0], -1)
        aux = self.aux_layers[-1](aux)
        return x, aux

    def __make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
#################################################################################

class SiFer_ce(nn.Module):
    def __init__(self, input_shape, num_classes, aux_pos = 1, aux_kernels = [128, 128], layers = [3, 4], num_bins = 20, block = BasicBlock):
        super(SiFer_ce, self).__init__()
        self.num_classes = num_classes
        self.aux_pos = aux_pos + 4
        # main network
        network = models.resnet34(weights = 'IMAGENET1K_V1')
        for param in network.parameters():
            param.requires_grad = True
            
        self.layers = nn.ModuleList(list(network.children())[:-1])
        self.layers.append(nn.Linear(512, num_classes))

        # Auxiliary Network
        self.aux_layers = nn.ModuleList([])
        
        self.inplanes = 128
        for kernel_id in range(len(aux_kernels)):
            self.aux_layers.append(self.__make_layer(block, aux_kernels[kernel_id], layers[kernel_id], stride = 2))
            
        self.aux_layers.append(nn.AvgPool2d(7, stride=1, padding=2))

        # output size calculation
        with torch.no_grad():
            outputs_main = []
            x = torch.zeros(input_shape)
            for layer in self.layers[:-1]:
                x = layer(x)
                outputs_main.append(x)
    
            x = x.view(x.shape[0], -1)
            x = self.layers[-1](x)
            outputs_main.append(x)
    
            # forward for the aux network
            aux = outputs_main[self.aux_pos]
            for aux_layer in self.aux_layers:
                aux = aux_layer(aux)
            aux = aux.reshape((aux.shape[0], -1))
        
        self.aux_layers.append(nn.Linear(aux.shape[-1], num_bins))

        # parameters dict
        self.params = nn.ModuleDict({
            "main" : self.layers,
            "aux"  : self.aux_layers,
            'forget': self.layers[:self.aux_pos]
        })
        
    def forward(self, x):
        # forward for the main network
        outputs_main = []
        
        for layer in self.layers[:-1]:
            x = layer(x)
            outputs_main.append(x)

        x = x.view(x.shape[0], -1)
        x = self.layers[-1](x)
        outputs_main.append(x)

        # forward for the aux network
        aux = outputs_main[self.aux_pos]
        for aux_layer in self.aux_layers[:-1]:
            aux = aux_layer(aux)

        aux = aux.view(aux.shape[0], -1)
        aux = self.aux_layers[-1](aux)
        return x, aux

    def __make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
###########################################################################

class SiFer_ord(nn.Module):
    def __init__(self, input_shape, num_classes, aux_pos = 1, aux_kernels = [128, 128], layers = [3, 4], num_bins = 20, block = BasicBlock):
        super(SiFer_ord, self).__init__()
        self.num_classes = num_classes
        self.aux_pos = aux_pos + 4
        # main network
        network = models.resnet34(weights = 'IMAGENET1K_V1')
        for param in network.parameters():
            param.requires_grad = True
            
        self.layers = nn.ModuleList(list(network.children())[:-1])
        self.layers.append(nn.Linear(512, num_classes))

        # Auxiliary Network
        self.aux_layers = nn.ModuleList([])
        
        self.inplanes = 128
        for kernel_id in range(len(aux_kernels)):
            self.aux_layers.append(self.__make_layer(block, aux_kernels[kernel_id], layers[kernel_id], stride = 2))
            
        self.aux_layers.append(nn.AvgPool2d(7, stride=1, padding=2))

        # output size calculation
        with torch.no_grad():
            outputs_main = []
            x = torch.zeros(input_shape)
            for layer in self.layers[:-1]:
                x = layer(x)
                outputs_main.append(x)
    
            x = x.view(x.shape[0], -1)
            x = self.layers[-1](x)
            outputs_main.append(x)
    
            # forward for the aux network
            aux = outputs_main[self.aux_pos]
            for aux_layer in self.aux_layers:
                aux = aux_layer(aux)
            aux = aux.reshape((aux.shape[0], -1))

        
        self.aux_layers.append(nn.Linear(aux.shape[-1], num_bins))

        # parameters dict
        self.params = nn.ModuleDict({
            "main" : self.layers,
            "aux"  : self.aux_layers,
            'forget': self.layers[:self.aux_pos]
        })
        
    def forward(self, x):
        # forward for the main network
        outputs_main = []
        
        for layer in self.layers[:-1]:
            x = layer(x)
            outputs_main.append(x)

        x = x.view(x.shape[0], -1)
        x = self.layers[-1](x)
        outputs_main.append(x)

        # forward for the aux network
        aux = outputs_main[self.aux_pos]
        for aux_layer in self.aux_layers[:-1]:
            aux = aux_layer(aux)

        aux = aux.view(aux.shape[0], -1)
        aux = self.aux_layers[-1](aux)
        return x, aux

    def __make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)