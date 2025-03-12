import os
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt


def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, loss_g, loss_d, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
    torch.save({
        'epoch': epoch + 101,  # 下次恢复时从 epoch+1 开始
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'loss_g': loss_g,
        'loss_d': loss_d,
    }, checkpoint_path)


def train(netG, netD, dataloader, criterion, device, nz=100, epochs=25, fixed_noise=None,
          results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    lr_D = 4e-4  # 判别器学习率
    lr_G = 1e-4  # 生成器学习率
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(0.0, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(0.0, 0.9))
    #optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    #optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    # 如果没有传入固定噪声，则生成一个用于可视化的固定噪声
    if fixed_noise is None:
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    loss_G_history = []
    loss_D_history = []

    loss_G_epoch = []
    loss_D_epoch = []
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False)
        for i, (real_imgs, _) in enumerate(pbar):

            netD.train()
            netG.train()

            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # 真实和虚假图片的标签
            label_real = torch.ones(b_size, 1, device=device)
            label_fake = torch.zeros(b_size, 1, device=device)

            # -----------------------------
            # (1) 训练判别器
            # -----------------------------
            optimizerD.zero_grad()
            output_real = netD(real_imgs)
            loss_d_real = criterion(output_real, label_real)

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_imgs = netG(noise)
            output_fake = netD(fake_imgs.detach())
            loss_d_fake = criterion(output_fake, label_fake)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()

            # -----------------------------
            # (2) 训练生成器
            # -----------------------------
            optimizerG.zero_grad()
            output_fake_g = netD(fake_imgs)  # 不 detach 以便生成器获得梯度
            loss_g = criterion(output_fake_g, label_real)
            loss_g.backward()
            optimizerG.step()

            pbar.set_postfix({
                "loss_d": f"{loss_d.item():.4f}",
                "loss_g": f"{loss_g.item():.4f}"
            })

            loss_G_history.append(loss_g.item())
            loss_D_history.append(loss_d.item())

        loss_G_epoch.append(loss_g.item())
        loss_D_epoch.append(loss_d.item())

        save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, loss_g.item(), loss_d.item())
        print(f"Epoch {epoch + 1} checkpoints saved.")


        netG.eval()
        with torch.no_grad():
            samples = netG(fixed_noise).cpu()  # 生成图片
        torchvision.utils.save_image(samples, os.path.join(results_dir, f"epoch_{epoch + 1}.png"),
                                     nrow=8,
                                     normalize=True,
                                     value_range=(-1, 1))
        print(f"Epoch {epoch + 1} finished and image saved.")
        netG.train()

    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(loss_G_history) + 1), loss_G_history, label="Generator Loss")
    plt.plot(range(1, len(loss_D_history) + 1), loss_D_history, label="Discriminator Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig("plots/loss_curve.png")
    plt.show()
    print("Loss curves saved as plots/loss_curve.png")

    plt.figure()
    plt.plot(range(1, epochs + 1), loss_G_epoch, label="Generator Loss")
    plt.plot(range(1, epochs + 1), loss_D_epoch, label="Discriminator Loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig("plots/loss_curve.png")
    plt.show()
    print("Loss curves saved as plots/loss_curve_epoch.png")
