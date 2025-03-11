import os
import torch
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
import torch.nn as nn


def truncated_normal(batch_size, nz, threshold=2.0, device='cuda'):
    """
    从截断正态分布采样噪声，保证每个维度的值在 [-threshold, threshold] 内。
    返回 shape: (batch_size, nz)
    """
    z = torch.randn(batch_size, nz, device=device)
    while True:
        mask = z.abs() > threshold
        if not mask.any():
            break
        z[mask] = torch.randn_like(z[mask])
    return z


def compute_gradient_penalty(D, real_samples, fake_samples, device, lambda_gp=5):
    """
    计算梯度惩罚项:
      GP = λ * E[ (||∇_x D(α*real + (1-α)*fake)||_2 - 1)^2 ]
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return penalty


def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, loss_g, loss_d):
    checkpoint_path = f"checkpoints3/epoch_{epoch + 1}.pth"
    torch.save({
        'epoch': epoch + 1,  # 下次恢复时从 epoch+1 开始
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'loss_g': loss_g,
        'loss_d': loss_d,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def train_wgan_gp(netG, netD, dataloader, nz=100, epochs=25, n_critic=5, lambda_gp=10,
                  device=torch.device("cpu")):
    optimizerD = optim.Adam(netD.parameters(), lr=1.5e-5, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0005, betas=(0.0, 0.9))

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        for i, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # ---------------------
            # 训练判别器 D
            # ---------------------
            for _ in range(n_critic):
                optimizerD.zero_grad()
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_imgs = netG(noise)

                d_real = netD(real_imgs)
                d_fake = netD(fake_imgs.detach())
                gp = compute_gradient_penalty(netD, real_imgs, fake_imgs, device, lambda_gp=lambda_gp)
                loss_D = torch.mean(d_fake) - torch.mean(d_real) + gp
                loss_D.backward()
                optimizerD.step()

            # ---------------------
            # 训练生成器 G
            # ---------------------
            optimizerG.zero_grad()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_imgs = netG(noise)
            loss_G = -torch.mean(netD(fake_imgs))
            loss_G.backward()
            optimizerG.step()

            pbar.set_postfix({
                "loss_d": f"{loss_D.item():.4f}",
                "loss_g": f"{loss_G.item():.4f}"
            })

        # 保存 checkpoint
        save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, loss_G.item(), loss_D.item())
        print(f"Epoch {epoch + 1} checkpoints saved.")

        # 每个 epoch 结束后保存生成样本
        netG.eval()
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        os.makedirs("results3", exist_ok=True)
        vutils.save_image(fake, f"results3/epoch_{epoch + 1}.png", nrow=8, normalize=True, value_range=(-1, 1))
        print(f"Epoch {epoch + 1} finished and image saved.")
        netG.train()