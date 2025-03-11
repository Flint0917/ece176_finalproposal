import os
import torch
import torchvision
from tqdm import tqdm


def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, loss_g, loss_d, checkpoint_dir="checkpoints2"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 101}.pth")
    torch.save({
        'epoch': epoch + 101,  # 下次恢复时从 epoch+1 开始
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'loss_g': loss_g,
        'loss_d': loss_d,
    }, checkpoint_path)


def train(netG, netD, dataloader, criterion, device, checkpoint, nz=100, epochs=100, fixed_noise=None,
          results_dir="results2"):
    os.makedirs(results_dir, exist_ok=True)
    lr_D = 4e-4  # 判别器学习率
    lr_G = 1e-4  # 生成器学习率
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(0.0, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(0.0, 0.9))
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    # 如果没有传入固定噪声，则生成一个用于可视化的固定噪声
    if fixed_noise is None:
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

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

        save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, loss_g.item(), loss_d.item())
        print(f"Epoch {epoch + 101} checkpoints saved.")


        netG.eval()
        with torch.no_grad():
            samples = netG(fixed_noise).cpu()  # 生成图片
        torchvision.utils.save_image(samples, os.path.join(results_dir, f"epoch_{epoch + 101}.png"),
                                     nrow=8,
                                     normalize=True,
                                     value_range=(-1, 1))
        print(f"Epoch {epoch + 101} finished and image saved.")
        netG.train()
