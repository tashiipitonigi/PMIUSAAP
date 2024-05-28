from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

use_dynamic_edge_conv = False

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       torch.nn.BatchNorm1d(out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels),
                       torch.nn.BatchNorm1d(out_channels),
                       ReLU())

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
    
from torch_geometric.nn import knn_graph

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super().__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x, edge_index)

class im_encoder(nn.Module):
    def __init__(self, input_dim, patch_nums, k, layer_sizes):
        super(im_encoder, self).__init__()
        self.patch_nums = patch_nums
        self.input_dim = input_dim

        self.k = k

        if use_dynamic_edge_conv:
            conv_list = [DynamicEdgeConv(self.input_dim, layer_sizes[0], k=self.k)]
            for i in range(1, len(layer_sizes)):
                conv_list.append(DynamicEdgeConv(layer_sizes[i-1], layer_sizes[i], k=self.k))

            conv_list.append(DynamicEdgeConv(layer_sizes[-1], 1024, k=self.k))
            self.convs = nn.ModuleList(conv_list)
        else:
            conv_list = [EdgeConv(self.input_dim, layer_sizes[0])]
            for i in range(1, len(layer_sizes)):
                conv_list.append(EdgeConv(layer_sizes[i-1], layer_sizes[i]))

            conv_list.append(EdgeConv(layer_sizes[-1], 1024))
            self.convs = nn.ModuleList(conv_list)

    def forward(self, input):
        bs = input.size(0)
        ps = input.size(1)

        res = []    

        if use_dynamic_edge_conv:
            for x in input:
                x_c = x
                for i in range(len(self.convs)):
                    x_c = self.convs[i](x_c)
                res.append(x_c)
        else:
            for x in input:
                x_c = x
                edge_index = knn_graph(x, self.k, loop=False)
                for i in range(len(self.convs)):
                    x_c = self.convs[i](x_c, edge_index)
                res.append(x_c)

        out = torch.stack(res)
        
        return out


class im_decoder(nn.Module):
    def __init__(self):
        super(im_decoder, self).__init__()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

class im_filter_module(nn.Module):
    def __init__(self, input_dim, patch_nums, k, layer_sizes):
        super(im_filter_module, self).__init__()

        self.patch_nums = patch_nums
        self.input_dim = input_dim

        self.encoder = im_encoder(self.input_dim, self.patch_nums, k=k, layer_sizes=layer_sizes)
        self.decoder = im_decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
