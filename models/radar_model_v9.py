# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.auto_attention import RCS_attn
from linear_attention_transformer.linear_attention_transformer import SelfAttention


def knn(x, k):
    device = x.device
    mask = torch.eye(x.size(2), device=device).bool().unsqueeze(0).expand(x.size(0), -1, -1)
    distences = torch.cdist(x.transpose(2, 1), x.transpose(2, 1))
    distences.masked_fill(mask, float('inf'))

    dist, indices = torch.topk(distences, k, largest=False)

    return dist, indices


def get_graph_feature(dist, idx, fea, k=20, RCS=False):
    device = fea.device
    batch_size, _, num_points = fea.size()

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    fea = fea.transpose(2, 1).contiguous()
    feature = fea.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, -1)

    feature_rcs = torch.zeros_like(feature)
    if RCS:
        #### raw
        feature_rcs = feature[:, :, :, 3].reshape(batch_size, num_points, -1)
        fea_rcs = fea[:, :, 3].view(batch_size, num_points, -1)
        feature_rcs = torch.cat((fea_rcs, feature_rcs), dim=2).permute(0, 2, 1)

        feature = feature[:, :, :, :3].max(dim=2)[0] # + feature[:, :, :, :3].mean(dim=2)
        fea = fea.view(batch_size, num_points, -1)[:, :, :3]

    else:
        feature = feature.max(dim=2)[0] # + feature.mean(dim=2)
        fea = fea.view(batch_size, num_points, -1)

    dist = torch.unsqueeze(dist, dim=3)
    dist = dist.max(dim=2)[0] # + dist.mean(dim=2)
    feature = torch.cat((fea, feature - fea, dist), dim=2).permute(0, 2, 1)

    return feature, feature_rcs


class radar_model(nn.Module):
    def __init__(self, min_k=5, max_k=20, input_channel=3, embedding=128, num_points=1024):
        super(radar_model, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 64, (1, input_channel)),
            nn.Conv2d(64, 64, (1, 1)),
            nn.Conv2d(64, 64, (1, 1)),
            nn.Conv2d(64, 128, (1, 1)),
            nn.Conv2d(128, embedding, (1, 1))
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(embedding)])


        self.min_k = min_k
        self.max_k = max_k

        self.dgcnn_f_conv = nn.ModuleList([
            self._conv_block(3 * 2 + 1, 64),
            self._conv_block(3 * 2 + 1, 64),
            self._conv_block((64+3) * 2 + 1, 64),
            self._conv_block((64+3) * 2 + 1, 64),
            self._conv_block((64+3+64) * 2 + 1, 128),
            self._conv_block((64+3+64) * 2 + 1, 128)
        ])

        self.conv_enhanced = nn.Sequential(nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128))
        self.multihead_attn = SelfAttention(dim=128, heads=4, causal=False)
        self.sigma = nn.Parameter(torch.rand(1))

        self.RCS_attn = RCS_attn(dim1=min_k + 1, dim2=max_k + 1, embedding_dim=32, num_points=num_points)
        self.gamma = nn.Parameter(torch.rand(1))

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        xyz = torch.squeeze(x[:, :, :, :3], dim=1).permute(0, 2, 1)
        xyzi = torch.squeeze(x, dim=1).permute(0, 2, 1)

        dist_mink, idx_mink = knn(xyz, k=self.min_k)
        dist_maxk, idx_maxk = knn(xyz, k=self.max_k)

        for conv, bn in zip(self.conv_layers[:-1], self.bn_layers[:-1]):
            x = F.relu(bn(conv(x)))

        ## GCN
        x_g1, x_rcs1 = get_graph_feature(dist_mink, idx_mink, xyzi, k=self.min_k, RCS=True)
        x_g2, x_rcs2 = get_graph_feature(dist_maxk, idx_maxk, xyzi, k=self.max_k, RCS=True)
        x_g1 = self.dgcnn_f_conv[0](x_g1)
        x_g2 = self.dgcnn_f_conv[1](x_g2)

        x_g1 = torch.cat((x_g1, xyzi[:,:3,:]), dim=1)
        x_g2 = torch.cat((x_g2, xyzi[:,:3,:]), dim=1)

        x_g3, _ = get_graph_feature(dist_mink, idx_mink, x_g1, k=self.min_k)
        x_g4, _ = get_graph_feature(dist_maxk, idx_maxk, x_g2, k=self.max_k)
        x_g3 = self.dgcnn_f_conv[2](x_g3)
        x_g4 = self.dgcnn_f_conv[3](x_g4)

        x_g3 = torch.cat((x_g3, x_g1), dim=1)
        x_g4 = torch.cat((x_g4, x_g2), dim=1)

        x_g5, _ = get_graph_feature(dist_mink, idx_mink, x_g3, k=self.min_k)
        x_g6, _ = get_graph_feature(dist_maxk, idx_maxk, x_g4, k=self.max_k)
        x_g5 = self.dgcnn_f_conv[4](x_g5)
        x_g6 = self.dgcnn_f_conv[5](x_g6)
        x3 = self.sigma * x_g5 + (1 - self.sigma) * x_g6

        x3 = self.multihead_attn(x=torch.squeeze(x, dim=3).permute(2, 0, 1), context=x3.permute(2, 0, 1))
        x3 = x3.permute(1, 2, 0)
        x3 = F.relu(self.conv_enhanced(torch.cat((torch.squeeze(x, dim=3), x3), dim=1)))

        x3 = torch.unsqueeze(x3, dim=3)

        x = self.bn_layers[-1](self.conv_layers[-1](x3))

        # ### rcs ###
        x_enhanced = self.RCS_attn(x_rcs1, x_rcs2, x)
        x = x + self.gamma * torch.unsqueeze(x_enhanced, dim=3)
        return x
