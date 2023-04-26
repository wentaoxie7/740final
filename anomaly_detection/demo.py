import torch
import numpy as np
from torchsummary import summary

from .RD4AD.resnet  import wide_resnet50_2, resnet34, resnet18, resnet152, resnet50
print('resnet34')
encoder, bn = resnet34(pretrained = True)
input = torch.randn(1, 3, 64, 64)
output = encoder(input)
for o in output:
    print(o.shape)

bn_output = bn(output)
print(bn_output.shape)

print('resnet152')
encoder, bn = resnet152(pretrained = True)
input = torch.randn(1, 3, 96, 96)
output = encoder(input)
for o in output:
    print(o.shape)
bn_output = bn(output)
print(bn_output.shape)