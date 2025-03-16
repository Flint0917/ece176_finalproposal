import optuna
import torch
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
from training import compute_gradient_penalty
from training import train_wgan_gp, validate_model
from data_loader import get_celeba_dataloader

# 全局设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
latent_dim = 100
max_iterations = 1000  # 每个 trial 运行 1000 次迭代

def objective(trial):
    # 通过 Optuna 采样超参数
    lr_G = trial.suggest_float("lr_G", 1e-5, 1e-3, log=True)
    lr_D = trial.suggest_float("lr_D", 1e-5, 1e-3, log=True)
    n_critic = trial.suggest_int("n_critic", 1, 10)
    lambda_gp = trial.suggest_float("lambda_gp", 1.0, 20.0)

    # 获取训练和验证 dataloader
    train_loader, valid_loader = get_celeba_dataloader()

    # 初始化生成器和判别器
    generator = Generator(nz=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.0, 0.9))

    iteration = 0
    total_g_loss = 0.0
    num_g_updates = 0

    # 简单的训练循环，只跑 max_iterations 次迭代
    for epoch in range(1000):  # epoch 数可以设大点，但内部会根据 iteration 数提前结束
        for i, (imgs, _) in enumerate(train_loader):
            real_imgs = imgs.to(device)
            current_batch_size = real_imgs.size(0)
            iteration += 1

            # ---------------------
            # 更新判别器
            # ---------------------
            optimizer_D.zero_grad()
            z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs.detach())
            gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), device, lambda_gp)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gp
            d_loss.backward()
            optimizer_D.step()

            # ---------------------
            # 每 n_critic 次更新一次生成器
            # ---------------------
            if iteration % n_critic == 0:
                optimizer_G.zero_grad()
                z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
                fake_imgs = generator(z)
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()

                total_g_loss += g_loss.item()
                num_g_updates += 1

            # 当迭代数达到 max_iterations 时退出
            if iteration >= max_iterations:
                break
        if iteration >= max_iterations:
            break

    # 计算平均生成器 loss（作为调优目标，Optuna 会尝试使该目标最小化）
    if num_g_updates > 0:
        avg_g_loss = total_g_loss / num_g_updates
    else:
        avg_g_loss = float('inf')

    # 返回目标值，注意在 GAN 中 loss 不一定直接代表生成质量，仅作为调参参考
    return avg_g_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # 可根据需要调整试验次数

    print("Best trial:")
    best_trial = study.best_trial
    print("  Avg G Loss:", best_trial.value("Avg G Loss", "N/A"))
    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
