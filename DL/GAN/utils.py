import math
import torch
import matplotlib.pyplot as plt

def generate_samples(generator, n, device):
    generator.eval()
    z = torch.randn((n, generator.dim_z), device=device)
    with torch.no_grad():
        samples = generator(z)
    return samples

def plot_samples(samples, title, file_path):
    n = samples.shape[0]
    samples_per_rows = math.ceil(math.sqrt(n))

    plt.title(title)

    for i in range(n):
        plt.subplot(samples_per_rows, samples_per_rows, i+1)
        plt.imshow(samples[i].permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
    
    plt.savefig(file_path)
    plt.close()