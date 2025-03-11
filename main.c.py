import torch
from data_loader import get_celeba_dataloader
from generator import Generator
from discriminator import Discriminator
from training import train_wgan_gp


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataloader = get_celeba_dataloader(root="./data", batch_size=64, image_size=64, shuffle=True)

    nz = 100
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)


    '''
        g_state = torch.load("checkpoints1/netG_epoch_26.pth", map_location=device)
        netG.load_state_dict(g_state)

        d_state = torch.load("checkpoints1/netD_epoch_26.pth", map_location=device)
        netD.load_state_dict(d_state)
    '''
    # t
    train_wgan_gp(netG, netD, dataloader, nz=nz, epochs=1000, n_critic=7, lambda_gp=17, device=device)


if __name__ == "__main__":
    main()
