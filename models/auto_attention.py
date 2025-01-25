# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""

    def __init__(self, in_dim, num_points):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(num_points // 8)
        self.bn2 = nn.BatchNorm1d(num_points // 8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=num_points, out_channels=num_points // 8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=num_points, out_channels=num_points // 8, kernel_size=1, bias=False),
                                      self.bn2,
                                      nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N X 1 )
            returns :
                out : output feature maps( B X C X N X 1 )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat)
        proj_key = self.key_conv(x_hat).permute(0, 2, 1)
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat) - similarity_mat
        affinity_mat = self.softmax(affinity_mat)

        proj_value = self.value_conv(x)
        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight
        out = self.alpha * out + x
        # out = torch.unsqueeze(out, dim=3)
        return out


class EfficientChannelAttention(nn.Module):  # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        w = self.avg_pool(x)
        # a = w.squeeze(-1).transpose(-1, -2)
        w = self.conv1(w.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        w = self.sigmoid(w)
        out = w * x
        out = self.alpha * out + x
        return out


class Audo_attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Audo_attn, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.conv1 = nn.Conv1d(in_channels=in_dim*2, out_channels=in_dim*2, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=in_dim * 2, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
        self.sigmoid = nn.Sigmoid()
        # self.pooling = nn.AvgPool1d(num_points, 1)


    def self_att(self, input):
        att = torch.matmul(input, input.transpose(2, 1))
        att = self.softmax(att)
        energy = torch.bmm(att, input)
        input = input + energy

        return input

    def cross_att(self, input1, input2):
        xy = torch.cat((input1, input2), dim=1)
        att = torch.matmul(xy, xy.transpose(2, 1))
        att_pool = torch.unsqueeze(torch.mean(att, dim=1), dim=2)
        energy = torch.bmm(att, att_pool)
        energy = self.sigmoid(self.conv1(energy))
        output = energy*xy
        output = self.conv2(output)

        return output

    def forward(self, x, y):
        x = torch.unsqueeze(x, dim=2)
        y = torch.unsqueeze(y, dim=2)

        ### self attention
        x = self.self_att(x)
        y = self.self_att(y)

        xy = torch.squeeze(self.cross_att(x,y), dim=-1)

        return xy


class ChannelAttention(nn.Module):  # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(ChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.conv1(w.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        w = self.sigmoid(w)
        out = w * x
        out = self.alpha * out + x
        return out

class RCS_attn(nn.Module):
    """ RCS attention Layer"""
    def __init__(self, dim1, dim2, embedding_dim, num_points):
        super(RCS_attn, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=dim1, out_channels=embedding_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(embedding_dim))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=dim2, out_channels=embedding_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(embedding_dim))
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=num_points, out_channels=num_points, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(num_points))

        # self.softmax = nn.Softmax(dim=-2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, fea):
        x = (self.conv1(x))
        x = x.permute(0, 2, 1)
        y = self.conv2(y)
        fea = torch.squeeze(fea, dim=3)
        similarity_mat = torch.bmm(x, y)

        affinity_mat_mean = torch.mean(similarity_mat, -1, keepdim=True) + torch.max(similarity_mat, -1, keepdim=True)[0]
        affinity_mat = self.conv3(affinity_mat_mean)
        affinity_mat = self.sigmoid(affinity_mat).permute(0, 2, 1)

        output = fea * affinity_mat

        return output

class ZR_attn(nn.Module):
    """ RCS attention Layer"""
    def __init__(self, dim1, dim2, embedding_dim, num_points):
        super(ZR_attn, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=embedding_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(embedding_dim))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=embedding_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(embedding_dim))
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=num_points, out_channels=num_points, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(num_points))

        # self.softmax = nn.Softmax(dim=-2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, fea):
        x = self.conv1(x)
        x = x.mean(dim=-1, keepdim=False).permute(0, 2, 1)
        y = self.conv2(y)
        y = y.mean(dim=-1, keepdim=False)
        fea = torch.squeeze(fea, dim=3)
        similarity_mat = torch.bmm(x, y)

        affinity_mat_mean = torch.mean(similarity_mat, -1, keepdim=True) + torch.max(similarity_mat, -1, keepdim=True)[0]
        affinity_mat = self.conv3(affinity_mat_mean)
        affinity_mat = self.sigmoid(affinity_mat).permute(0, 2, 1)

        output = fea * affinity_mat

        return output
