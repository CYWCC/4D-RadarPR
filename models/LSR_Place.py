from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import math
from models.radar_model_v9 import *

class MAX_FC(nn.Module):
    def __init__(self, num_points=512, feature_size=1024, output_dim=256):
        super(MAX_FC, self).__init__()
        self.maxpooling = nn.MaxPool1d(num_points)
        self.FC = nn.Conv1d(feature_size, output_dim, 1)
        self.bn = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        x = torch.squeeze(x,dim=3)
        x = self.maxpooling(x)
        x = torch.squeeze(self.bn(self.FC(x)), dim=-1)

        return x

class PNT_GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, kernel=(1024,1), feature_size=1024, output_dim=256):
        super(PNT_GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.kernel = kernel
        self.FC = nn.Conv1d(feature_size, output_dim, 1)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = torch.squeeze(x, dim=3).permute(0, 2, 1)
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, kernel_size=self.kernel)
        x = x.pow(1. / self.p).permute(0, 2, 1)
        x = torch.squeeze(self.bn(self.FC(x)), dim=-1)

        return x

class LSR_Place(nn.Module):
    def __init__(self, input_dim=3, output_dim=256, num_points=2500):
        super(LSR_Place, self).__init__()

        self.radar_model = radar_model(min_k=5, max_k=20,input_channel=input_dim, embedding=512, num_points=num_points)
        self.GeM = PNT_GeM(kernel=(num_points, 1), feature_size=512, output_dim=output_dim)
        # self.MAX_FC = MAX_FC(num_points=num_points, feature_size=512, output_dim=256)

    def forward(self, x):
        # ---------------------radar model----------------------#
        x = self.radar_model(x)  # （b，dim，N，1）
        x = F.normalize(x)

        x = self.GeM(x)
        # x = self.MAX_FC(x)
        x = F.normalize(x)

        return x


if __name__ == '__main__':
    num_points = 1024
    sim_data = Variable(torch.rand(44, 1, num_points, 4))
    sim_data = sim_data.cuda()

    pnv = LSR_Place.LSR_Place(global_feat=True, feature_transform=True, max_pool=False,
                                    output_dim=256, num_points=num_points).cuda()
    pnv.train()
    out3 = pnv(sim_data)
    print('pnv', out3.size())
