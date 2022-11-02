import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from gcn import GCNConv
from torch_scatter import scatter_add
import torch_sparse
from torch_geometric.utils import remove_self_loops, add_self_loops

np.set_printoptions(threshold=np.inf)

class GTN(nn.Module):
    def __init__(self, 
                num_edge_u, 
                num_channels_u,
                u_in,
                u_out,
                num_edge_v, 
                num_channels_v,
                v_in,
                v_out,
                num_nodes_u,
                num_nodes_v):
        super(GTN, self).__init__()

        self.num_edge_u = num_edge_u
        self.num_channels_u = num_channels_u
        self.u_in = u_in
        self.u_out = u_out

        self.num_nodes_u = num_nodes_u
        self.num_nodes_v = num_nodes_v
        self.gcn_u = GCNConv(in_channels=self.u_in, out_channels=u_out)
        layers = []
        layers.append(GTLayer(num_edge_u, num_channels_u, num_nodes_u, first=True))
        self.layers_u = nn.ModuleList(layers)


        self.num_edge_v = num_edge_v
        self.num_channels_v = num_channels_v

        self.v_in = v_in
        self.v_out = v_out
        self.gcn_v = GCNConv(in_channels=self.v_in, out_channels=self.v_out)
        layers = []
        layers.append(GTLayer(num_edge_v, num_channels_v, num_nodes_v, first=True))
        self.layers_v = nn.ModuleList(layers)

        self.loss = nn.CrossEntropyLoss()
        self.mlp = MLP((u_out+v_out)*num_channels_u , 2)


    def normalization_u(self, H):
        norm_H = []
        for i in range(self.num_channels_u):
            edge, value = H[i]
            edge, value = remove_self_loops(edge, value)
            deg_row, deg_col = self.norm_u(edge.detach(), self.num_nodes_u, value)
            value = deg_col * value
            norm_H.append((edge, value))

        return norm_H
    
    def normalization_v(self, H):
        norm_H = []

        for i in range(self.num_channels_v):
            edge, value=H[i]
            edge, value = remove_self_loops(edge, value)
            deg_row, deg_col = self.norm_v(edge.detach(), self.num_nodes_v, value)
            value = deg_col * value
            norm_H.append((edge, value))
        return norm_H



    def norm_u(self, edge_index, num_nodes_u, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                    dtype=dtype,
                                    device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        row, col = edge_index
        deg = scatter_add(edge_weight.clone(), col, dim=0, dim_size=num_nodes_u)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row], deg_inv_sqrt[col]
    
    def norm_v(self, edge_index, num_nodes_v, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                    dtype=dtype,
                                    device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        row, col = edge_index
        deg = scatter_add(edge_weight.clone(), col, dim=0, dim_size=num_nodes_v)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    

    def forward(self, A_u, X_u, A_v, X_v, index_list):
        Ws_u = []

        H_u, W = self.layers_u[0](A_u)



        Ws_u.append(W)

        for i in range(self.num_channels_u):
            if i==0:
                edge_index, edge_weight = H_u[i][0], H_u[i][1]
                Xu_ = self.gcn_u(X_u, edge_index=edge_index.detach(), edge_weight=edge_weight)
                Xu_ = F.relu(Xu_)
            else:
                edge_index, edge_weight = H_u[i][0], H_u[i][1]
                Xu_ = torch.cat((Xu_,F.relu(self.gcn_u(X_u, edge_index=edge_index.detach(), edge_weight=edge_weight))), dim=1)




        Ws_v = []

        H_v, W = self.layers_v[0](A_v)


        Ws_v.append(W)

        for i in range(self.num_channels_v):
            if i==0:
                edge_index, edge_weight = H_v[i][0], H_v[i][1]
                Xv_ = self.gcn_v(X_v,edge_index=edge_index.detach(), edge_weight=edge_weight)
                Xv_ = F.relu(Xv_)
            else:
                edge_index, edge_weight = H_v[i][0], H_v[i][1]
                Xv_ = torch.cat((Xv_, F.relu(self.gcn_v(X_v,edge_index=edge_index.detach(), edge_weight=edge_weight))), dim=1)



        target = []
        for i in index_list:
            target.append(i[2])
        target = torch.Tensor(target)
        index_list = np.array(index_list)

        B = torch.cat((Xu_[index_list[:, 0]], Xv_[index_list[:, 1]]), 1)
        B = self.mlp(B)
        B = torch.squeeze(B)
   
        loss = self.loss(B, torch.as_tensor(target, dtype = int))

        return Xu_, Xv_,  Ws_u ,Ws_v ,loss , B ,target



class MLP(nn.Module):
    def __init__(self, input_size,output_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, output_size),
            nn.Softmax(dim= -1)
        )

    def forward(self, x):
        out = self.linear(x)
        del x
        return out


class GTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes

        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels , num_nodes)
            self.conv2 = GTConv(in_channels, out_channels , num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels , num_nodes)

    def forward(self, A, H_=None):
        if self.first == True:
            result_A = self.conv1(A)
            result_B = self.conv2(A)

            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        H = []

        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]
            edges, values = torch_sparse.spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes , self.num_nodes, self.num_nodes)
            H.append((edges, values))
        del A
        return H, W

class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = None
        self.num_nodes = num_nodes
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, 0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        filter = F.softmax(self.weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index,edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value*filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=self.num_nodes, n=self.num_nodes)
            results.append((index, value))

        del A
        return results
