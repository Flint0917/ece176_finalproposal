import optuna
import torch
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
from training import compute_gradient_penalty
from training import train_wgan_gp, validate_model
from data_loader import get_celeba_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
latent_dim = 100
max_iterations = 1000

def objective(trial):

    lr_G = trial.suggest_float("lr_G", 1e-5, 1e-3, log=True)
    lr_D = trial.suggest_float("lr_D", 1e-5, 1e-3, log=True)
    n_critic = trial.suggest_int("n_critic", 1, 10)
    lambda_gp = trial.suggest_float("lambda_gp", 1.0, 20.0)

    train_loader, valid_loader = get_celeba_dataloader()

    generator = Generator(nz=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.0, 0.9))

    iteration = 0
    total_g_loss = 0.0
    num_g_updates = 0

    for epoch in range(1000):
        for i, (imgs, _) in enumerate(train_loader):
            real_imgs = imgs.to(device)
            current_batch_size = real_imgs.size(0)
            iteration += 1

            optimizer_D.zero_grad()
            z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs.detach())
            gp = compute_gradient_penalty(discriminator, real_imgs, fake_imgs.detach(), device, lambda_gp)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gp
            d_loss.backward()
            optimizer_D.step()

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

            if iteration >= max_iterations:
                break
        if iteration >= max_iterations:
            break

    if num_g_updates > 0:
        avg_g_loss = total_g_loss / num_g_updates
    else:
        avg_g_loss = float('inf')

    return avg_g_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    best_trial = study.best_trial
    print("  Avg G Loss:", best_trial.value("Avg G Loss", "N/A"))
    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
