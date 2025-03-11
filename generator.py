import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        """
        nz: 噪声向量维度
        ngf: 生成器中 feature map 的基数
        nc: 输出图像通道数 (例如 3 表示 RGB)
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入: (nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),  # (ngf*8, 4, 4)
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 状态: (ngf*8, 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # (ngf*4, 8, 8)
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态: (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # (ngf*2, 16, 16)
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态: (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),  # (ngf, 32, 32)
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 状态: (ngf, 32, 32)
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),  # (nc, 64, 64)
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, input):
        return self.main(input)
