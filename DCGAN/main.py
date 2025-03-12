import torch
import torch.nn as nn

from model import DCGANGenerator, DCGANDiscriminator
from dataloader import get_dataloader
from training import train

def weights_init_he(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def main():
    # ----------------------------------
    # 超参数设置
    # ----------------------------------
    nz = 100      # 噪声维度
    ngf = 64      # Generator 特征图数量
    ndf = 64      # Discriminator 特征图数量
    lr = 2e-4
    epochs = 100
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------------------------
    # 数据加载
    # ----------------------------------
    dataloader = get_dataloader(batch_size)

    # ----------------------------------
    # 初始化模型
    # ----------------------------------
    netG = DCGANGenerator(nz=nz, ngf=ngf).to(device)
    netD = DCGANDiscriminator(ndf=ndf).to(device)

    '''
    weights_init_he(netG)
    weights_init_he(netD)
    
    checkpoint = torch.load("../checkpoints2/epoch_100.pth", map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    '''

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # 开始训练
    train(netG, netD, dataloader, criterion, device,
          nz=nz, epochs=25, fixed_noise=fixed_noise)

if __name__ == "__main__":
    main()
