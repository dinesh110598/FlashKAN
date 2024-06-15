import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from layers import FlashKAN, Regular_KAN, KANLinear

from torch.profiler import profile, ProfilerActivity

device = torch.device("cuda:0")

