from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from skimage import measure
import os
from model import *
from torchsummary import summary

from models.attention import *
from models.backbone import *
from models.transformer import Transformer

#  Model #
#Region Proposal Networks
region_module = torch.load('model/IAANet/pretrained/rpn.pt', map_location=torch.device('cpu'))

region_module.trainable = False

l = [module for module in region_module.modules() if not isinstance(module, nn.Sequential)]

print(l)


#Attention Encoder
attention_module = Transformer(num_encoder_layers=4, d_model=512)
#IAANet
Model = attention(attention_module, region_module, pos='cosin', d_model=512)
