import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from lib.utils import load_spatialmatrix

def NormalMatrixBasisLs(L_s, last_s, second_last_s):

    rst = torch.matmul(L_s, last_s)

    _t = (rst * last_s).sum(dim=[2, 3])
    rst = rst - _t.unsqueeze(-1).unsqueeze(-1) * last_s

    _t = (rst * second_last_s).sum(dim=[2, 3])
    rst = rst - _t.unsqueeze(-1).unsqueeze(-1) * second_last_s

    rst = rst / torch.norm(rst, p='fro', dim=[2, 3], keepdim=True).clamp(min=1e-8)
    return rst

def NormalMatrixBasisLt(L_t, last_t, second_last_t):

    rst = torch.matmul(last_t, L_t)

    _t = (rst * last_t).sum(dim=[2, 3])  # [B, F]
    rst = rst - _t.unsqueeze(-1).unsqueeze(-1) * last_t

    _t = (rst * second_last_t).sum(dim=[2, 3])
    rst = rst - _t.unsqueeze(-1).unsqueeze(-1) * second_last_t

    rst = rst / torch.norm(rst, p='fro', dim=[2,3], keepdim=True).clamp(min=1e-8)
    return rst



def get_matrix_list(P, Q, M_00, L_s, L_t, last_s, second_last_s, last_t, second_last_t):
    '''
    P:  approximation oder of the spatial dimension
    Q:  approximation oder of the temporal dimension
    L_s:  spatial Laplacian matrix
    L_t:  temporal Laplacian matrix
    retrun: matrices with different orders
    '''

    M_list = []
    M_list.append(M_00)

    for i in range(P+1):
        if i > 0:
            M_i0 = NormalMatrixBasisLs(L_s, last_s, second_last_s)
            second_last_s = last_s
            last_s = M_i0

            M_list.append(M_i0)

            last_t = M_i0
            second_last_t = torch.zeros_like(last_t)

        for _ in range(1, Q + 1):
            M_ij = NormalMatrixBasisLt(L_t, last_t, second_last_t)
            second_last_t = last_t
            last_t = M_ij

            M_list.append(M_ij)

    return torch.stack(M_list, dim=1)   # batch, (P+1)*(Q+1)


class STEmbedding(nn.Module):
    '''
    num_nodes: number of nodes
    s_order: approximation oder of the spatial dimension
    t_order: approximation oder of the temporal dimension
    TE: [batch_size, num_his, 2] (dayofweek, timeofday)
    T: num of time steps in one day
    retrun: set of coefficients containing spatial-temporal information [batch_size, t_order, s_order]
    '''
    def __init__(self, s_order, t_order, num_nodes):
        super(STEmbedding, self).__init__()

        self.SE = nn.Parameter(torch.FloatTensor(num_nodes, 10))  # randomly initialize spatial embedding [N, 10]

        self.tmlp1 = torch.nn.Conv1d(295, s_order, kernel_size=1, padding=0, bias=True)  # change 295 to 6
        self.tmlp2 = torch.nn.Conv1d(12, t_order, kernel_size=1, padding=0, bias=True)

        self.smlp1 = torch.nn.Conv1d(num_nodes, t_order, kernel_size=1, padding=0, bias=True)
        self.smlp2 = torch.nn.Conv1d(10, s_order, kernel_size=1, padding=0, bias=True)

        self.num_nodes = num_nodes

    def forward(self, TE, T=288):

        # set T = 8 for KnowAir dataset and discard the dayofweek

        dayofweek = F.one_hot((TE[..., 0] % 7).long(), num_classes=7)  # shape: [B, T, 7]
        timeofday = F.one_hot((TE[..., 1] % T).long(), num_classes=T)

        TE = torch.cat((dayofweek, timeofday), dim=-1).float()  # B T 295
        TE = F.relu(self.tmlp1(TE.permute(0, 2, 1)))  # B s_order T
        TE = F.relu(self.tmlp2(TE.permute(0, 2, 1)))  # B t_order s_order

        SE = F.relu(self.smlp1(self.SE))
        SE = F.relu(self.smlp2(SE.permute(1, 0)).T)

        STE = F.relu(SE+TE)

        del dayofweek, timeofday
        return STE


class ST_Block(nn.Module):
    '''
    c_in: input dimension
    c_out: output dimension
    s_order: approximation oder of the spatial dimension
    t_order: approximation oder of the temporal dimension
    sp_matrix: [num_nodes, num_nodes]
    tp_matrix: [time_steps, time_steps]
    STE:     [batch_size, t_order, s_order]
    retrun: [batch_size, num_his, C]
    '''
    def __init__(self, c_in, c_out, dropout, s_order=10, t_order=5):
        super(ST_Block, self).__init__()
        self.c_in = c_in
        self.s_order = s_order
        self.t_order = t_order
        self.mlp = torch.nn.Conv2d(self.c_in * 2, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.bn = nn.BatchNorm2d(c_out)
        self.theta_mlp1 = torch.nn.Conv1d(self.t_order, self.t_order+1, kernel_size=1, bias=True)
        self.theta_mlp2 = torch.nn.Conv1d(self.s_order, self.s_order+1, kernel_size=1, bias=True)

        self.dropout = dropout

    def forward(self, x, sp_matrix, tp_matrix, STE):

        out = [x]

        theta_matrix = F.relu(self.theta_mlp2(self.theta_mlp1(STE).permute(0, 2, 1)).permute(0, 2, 1)).reshape(STE.shape[0], -1)  # [batch_size, t_prder, s_order]

        m_00 = x / torch.norm(x, p='fro', dim=[2, 3], keepdim=True).clamp(min=1e-8)
        last_s = m_00  # 标准化后
        last_t = m_00
        second_last_s = torch.zeros_like(x)
        second_last_t = torch.zeros_like(x)

        M_list = get_matrix_list(self.s_order, self.t_order, m_00, sp_matrix, tp_matrix, last_s,
                                    second_last_s, last_t, second_last_t)

        x_st = torch.einsum('br,brfnt->bfnt', theta_matrix, M_list)
        out.append(x_st)
        hidden = self.mlp(torch.cat(out, dim=1))
        hidden = self.bn(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        return hidden


class PyrTempConv(nn.Module):
    def __init__(self, c_in, stride, kernel):
        super(PyrTempConv, self).__init__()
        self.c_in = c_in
        self.pyr1 = nn.Conv2d(self.c_in, self.c_in * 2, kernel_size=(1, 1), stride=(1, 1))
        self.pyr2 = nn.Conv2d(self.c_in, self.c_in * 2, kernel_size=(1, 3), stride=(1, 3))
        self.pyr3 = nn.Conv2d(self.c_in, self.c_in * 2, kernel_size=(1, 6), stride=(1, 6))

        self.conv = nn.Sequential(
            nn.Conv2d(self.c_in*3, self.c_in, kernel_size=1))
        self.bn = nn.BatchNorm2d(self.c_in)

    def forward(self, x):
        #x1 = x
        x1_gate, x1_filter = torch.split(self.pyr1(x), self.c_in, dim=1)
        x1 = torch.sigmoid(x1_gate) * torch.tanh(x1_filter)
        x2_gate, x2_filter = torch.split(self.pyr2(x), self.c_in, dim=1)
        x2 = torch.sigmoid(x2_gate) * torch.tanh(x2_filter)
        x3_gate, x3_filter = torch.split(self.pyr3(x), self.c_in, dim=1)
        x3 = torch.sigmoid(x3_gate) * torch.tanh(x3_filter)
        x2 = F.interpolate(x2, x.shape[2:], mode='bilinear')
        x3 = F.interpolate(x3, x.shape[2:], mode='bilinear')
        concat = torch.cat([x1, x2, x3], 1)
        fusion = self.bn(self.conv(concat))
        return fusion

class MaPFilter(nn.Module):
    def __init__(self, args):
        super(MaPFilter, self).__init__()
        self.dataset = args.dataset
        self.num_nodes = args.num_nodes
        self.feature_dim = args.input_dim
        self.dropout = args.dropout
        self.layers = args.num_layers

        self.kernel = args.kernel_size
        self.stride = args.kernel_size
        self.nhid = args.rnn_units
        self.residual_channels = args.rnn_units
        self.skip_channels = args.skip_channels  # nhid * 8
        self.end_channels = args.end_channels  # nhid * 16
        self.input_window = args.horizon
        self.output_window = args.output_window
        self.output_dim = args.output_dim
        self.device = torch.device('cuda:0')
        self.s_order = args.s_order
        self.t_order = args.t_order

        self.stembedding = STEmbedding(s_order=self.s_order, t_order=self.t_order, num_nodes=self.num_nodes)
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        self.st_blocks = nn.ModuleList()
        self.pry_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        for i in range(self.layers):
            self.st_blocks.append(ST_Block(c_in=self.residual_channels, c_out=self.residual_channels,
                                           dropout=self.dropout, s_order=self.s_order, t_order=self.t_order))
            self.pry_blocks.append(PyrTempConv(c_in=self.residual_channels, stride=self.stride,
                                               kernel=self.kernel))
            self.skip_convs.append(nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(self.residual_channels))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)

        sp_matrix = load_spatialmatrix(self.dataset, self.num_nodes)

        #sp_matrix = np.load('../data/KnowAir/knowair_adj_mat.npy') # spatial adj_mat for KnowAir dataset

        self.sp_matrix = sp_matrix.to(self.device)

        tp_matrix = (F.pad(torch.eye(11), (0, 1, 1, 0), 'constant', 0) + F.pad(torch.eye(11), (1, 0, 0, 1), 'constant',
                                                                               0)).to(self.device)
        dt = torch.sum(tp_matrix, dim=0)
        Dt = torch.diag(torch.rsqrt(dt))
        self.tp_matrix = torch.matmul(Dt, torch.matmul(tp_matrix, Dt))

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        inputs = source[:, :, :, 0:1]
        temp = source[:, :, 1, 1:]
        STE = self.stembedding(temp)  # t_order, s_order

        inputs = inputs.permute(0, 3, 2, 1)  # (batch_size, feature_dim, num_nodes, input_window)
        x = inputs
        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, receptive_field)
        skip = 0

        for i in range(self.layers):
            residual = x
            x_st = self.st_blocks[i](residual, self.sp_matrix, self.tp_matrix, STE)
            x_st_pry = self.pry_blocks[i](x_st)
            s = x_st_pry
            s = self.skip_convs[i](s)

            skip = s + skip
            x = x_st_pry + residual
            x = self.bn[i](x)

        x = F.relu(skip[:, :, :, -1:])
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x
