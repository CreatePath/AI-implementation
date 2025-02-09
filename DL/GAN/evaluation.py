import config
import torch

def evaluate(discriminator, generator, eval_loader, criterion, device):
    discriminator.eval()
    generator.eval()
    total_loss_d, total_loss_g = 0, 0

    with torch.no_grad():
        for real, _ in eval_loader:
            real = real.to(device)
            ones = torch.ones((real.shape[0], 1), device=device, dtype=torch.float32)

            z = torch.randn((real.shape[0], config.DIM_Z), device=device)
            fake = generator(z)
            zeros = torch.zeros((real.shape[0], 1), device=device, dtype=torch.float32)

            pred_real = discriminator(real)
            loss_d_real = criterion(pred_real, ones)

            pred_fake = discriminator(fake)
            loss_g = criterion(pred_fake, ones)
            loss_d_fake = criterion(pred_fake, zeros)

            total_loss_d += loss_d_real.item() + loss_d_fake.item()
            total_loss_g += loss_g.item()

    total_loss_d /= len(eval_loader)
    total_loss_g /= len(eval_loader)

    return total_loss_d, total_loss_g