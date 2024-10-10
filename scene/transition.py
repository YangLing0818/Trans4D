import functools
import math
import os
import time
# from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch.nn.init as init
from collections import OrderedDict
# from scene.hexplane import HexPlaneField


class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)
    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 4 # 32 dim in total, same as AYG
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = p / 3.0 # change to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=-1)
        return p


def custom_activation(x, sharpness=10.0):
    """
    Custom activation function that behaves like a smooth step function.
    - For large negative values, it approaches 0.
    - For large positive values, it approaches 1.
    - Around 0, it transitions smoothly.
    """
    return torch.sigmoid(sharpness * x)


class RankPoints(nn.Module):
    def __init__(self, input_dim=4, output_dim=1):
        super(RankPoints, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)
    
    def forward(self, attr, sharpness=10.0, training=True, iter_rate=1):
        x = F.relu(self.fc1(attr))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.fc4(x)
        output = torch.sigmoid(sharpness * output)  # Apply sigmoid to get values in [0, 1]
        output = output*iter_rate

        if not training:
            # During testing, sample from a Bernoulli distribution based on the output probability
            output_bi = torch.bernoulli(output)
            output = output * output_bi * iter_rate

        return output


class Transition(nn.Module):
    def __init__(self, training=True):
        super(Transition, self).__init__()
        self.mean_3D_ranker = RankPoints(input_dim=32)
        self.threshold_predictor = RankPoints(input_dim=32)
        self.pos_emb = positional_encoding()
        self.training = training

    def forward(self, mean_3D, mean_2D, shs, opacity, scales, rotation, time, iter_rate=1):
        time_expanded = time.expand(mean_3D.size(0), -1)
        mean_3D_emb = self.pos_emb(torch.cat((mean_3D, time_expanded), dim=-1))
        rank = self.mean_3D_ranker(mean_3D_emb, sharpness=100.0, training=self.training, iter_rate=iter_rate)
        # threshold = self.threshold_predictor(mean_3D_emb)
        # opacity_rate = custom_activation(rank - threshold, 200.0)

        # index = ((rank - threshold) > 0.0).squeeze()
        # _, index_1 = torch.topk(rank.squeeze(), int(mean_3D.size(0)/2))
        # if index_1.size(0) > index.sum(0):
        #     index = index_1

        mean_3D_final = mean_3D#[index]
        mean_2D_final = mean_2D#[index]
        shs_final = shs#[index]
        opacity_final = (opacity * rank)#[index]
        scales_final = scales#[index]
        rotation_final = rotation#[index]

        return mean_3D_final, mean_2D_final, shs_final, opacity_final, scales_final, rotation_final

