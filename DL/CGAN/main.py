import argparse
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from networks import Generator, Discriminator
import config

def main(args):
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)

    pretrained = args.pretrained
    epochs = args.epochs
    lr_g = args.lr_g
    lr_d = args.lr_d
    split_ratio = args.split_ratio
    batch_size = args.batch_size
    weight_decay_g = args.weight_decay_g
    weight_decay_d = args.weight_decay_d
    dropout = args.dropout

    device = args.device
    if 0 <= device.find("cuda") and not torch.cuda.is_available():
        device = "cpu"

    train_dataset = MNIST(root=config.ROOT, train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root=config.ROOT, train=False, download=True, transform=ToTensor())

    tr_dset, val_dset = random_split(train_dataset, [1.0-split_ratio, split_ratio])

    print(f"train_data: {len(tr_dset)}")
    print(f"val_data: {len(val_dset)}")
    print(f"test_data: {len(test_dataset)}")

    tr_loader = DataLoader(tr_dset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"batch size: {batch_size}")
    print(f"tr_batches: {len(tr_loader)}")
    print(f"val_batches: {len(val_loader)}")
    print(f"test_batches: {len(test_loader)}")

    generator = Generator(config.DIM_Z, config.DIM_Y, config.OUT_G, config.HIDDEN_G, activation=nn.LeakyReLU, dropout=dropout).to(device)
    discriminator = Discriminator(config.IN_D, config.OUT_D, config.HIDDEN_D, activation=nn.LeakyReLU, dropout=dropout).to(device)

    print(generator)
    print(discriminator)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, weight_decay=weight_decay_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, weight_decay=weight_decay_d)

    if pretrained:
        checkpoint = torch.load(pretrained, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])

    nets = {"generator": generator,
            "discriminator": discriminator,}
    optims = {"opt_g": optimizer_g,
              "opt_d": optimizer_d,}
    loaders = {"train": tr_loader,
               "val": val_loader,
               "test": test_loader}
    criterion = nn.BCELoss()

    if args.train:
        train_loss_history, val_loss_history = train(epochs, nets, optims, loaders, criterion, device)
        visualize_history(train_loss_history, val_loss_history)

    loss_d_test, loss_g_test = evaluate(nets, test_loader, criterion, device)
    print(f"Test loss_D: {loss_d_test}, loss_G: {loss_g_test}")

    generator.eval()
    with torch.no_grad():
        test_generation(config.IMAGEPATH+"final.png", generator, config.NUM_CLASSES, config.N_SAMPLES, device)

def train(epochs,
          nets: dict[str, nn.Module],
          optims: dict[str, nn.Module],
          loaders: dict[str, DataLoader],
          criterion: nn.Module,
          device: str):

    train_loss_history = {"generator": [],
                        "discriminator": [],}
    val_loss_history = {"generator": [],
                        "discriminator": [],}

    tr_loader = loaders["train"]
    val_loader = loaders["val"]

    generator = nets["generator"]
    discriminator = nets["discriminator"]

    opt_g = optims["opt_g"]
    opt_d = optims["opt_d"]

    num_batches = len(tr_loader)

    for epoch in range(epochs):
        loss_d = 0
        loss_g = 0
        for i, (imgs, labels) in enumerate(tr_loader):
            real_imgs = imgs.flatten(start_dim=1).to(device)
            real_labels = F.one_hot(labels, config.NUM_CLASSES).to(device)

            loss_d += train_discriminator_onestep(nets, opt_d, real_imgs, real_labels, criterion, device)
            loss_g += train_generator_onestep(nets, opt_g, real_imgs.shape[0], criterion, device)

        loss_d /= num_batches
        loss_g /= num_batches

        train_loss_history["generator"].append(loss_g)
        train_loss_history["discriminator"].append(loss_d)
        print(f"Epoch {epoch+1} / {epochs} Train loss_D: {loss_d}, loss_G: {loss_g}")

        loss_d_val, loss_g_val = evaluate(nets, val_loader, criterion, device)
        val_loss_history["discriminator"].append(loss_d_val)
        val_loss_history["generator"].append(loss_g_val)

        print(f"Epoch {epoch+1} / {epochs} Val loss_D: {loss_d_val}, loss_G: {loss_g_val}")

        generator.eval()
        with torch.no_grad():
            fname = config.IMAGEPATH+f"epoch_{epoch+1}.png"
            test_generation(fname, generator, config.NUM_CLASSES, config.N_SAMPLES, device)
            print(f"{fname} is saved.")

        checkpoint_name = config.MODELPATH + f"epoch_{epoch+1}.pkl"
        torch.save({"generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),}, checkpoint_name)
        print(f"{checkpoint_name} is saved.")

    return train_loss_history, val_loss_history

def train_discriminator_onestep(nets: dict[str, nn.Module],
                                opt_d: optim.Optimizer,
                                imgs: torch.Tensor,
                                labels: torch.Tensor,
                                criterion: nn.Module,
                                device: str) -> float:
    discriminator = nets["discriminator"]
    generator = nets["generator"]

    discriminator.train()
    batch_size = imgs.shape[0]

    preds_real = discriminator(imgs, labels)
    loss_real = criterion(preds_real, torch.ones((batch_size, 1), device=device))

    fake_imgs, fake_labels = generate_random_imgs(generator, batch_size, device)

    preds_fake = discriminator(fake_imgs, fake_labels)
    loss_fake = criterion(preds_fake, torch.zeros((batch_size, 1), device=device))

    loss = loss_real + loss_fake

    opt_d.zero_grad()
    loss.backward()
    opt_d.step()

    return loss.item()

def train_generator_onestep(nets: dict[str, nn.Module],
                            opt_g: optim.Optimizer,
                            batch_size: int,
                            criterion: nn.Module,
                            device: str) -> float:
    discriminator = nets["discriminator"]
    generator = nets["generator"] 

    generator.train()
    fake_imgs, fake_labels = generate_random_imgs(generator, batch_size, device)
    preds_fake = discriminator(fake_imgs, fake_labels)
    loss = criterion(preds_fake, torch.ones((batch_size, 1), device=device))

    opt_g.zero_grad()
    loss.backward()
    opt_g.step()

    return loss.item()

def test_generation(fname: str, generator: Generator, num_classes: int, n_samples: int, device: str):
    labels = torch.arange(num_classes, device=device)
    imgs, _ = generate_imgs(generator, labels, device)

    for _ in range(n_samples-1):
        tmp, _ = generate_imgs(generator, labels, device)
        imgs = torch.cat([imgs, tmp], dim=0)

    for i in range(n_samples):
        for j in range(num_classes):
            plt.subplot(num_classes, num_classes, i*num_classes+j+1)
            plt.imshow(imgs[i*num_classes+j].cpu().numpy().reshape(config.HEIGHT, config.WIDTH))
            plt.axis("off")
    plt.savefig(fname)
    plt.close()

def generate_random_imgs(generator: Generator, batch_size: int, device: str) -> tuple[torch.Tensor]:
    fake_labels = torch.randint(0, config.NUM_CLASSES, (batch_size,)).to(device)
    fake_imgs, encoded_labels = generate_imgs(generator, fake_labels, device)
    return fake_imgs, encoded_labels

def generate_imgs(generator: Generator, labels: torch.Tensor, device: str):
    z = torch.randn(len(labels), config.DIM_Z, device=device)
    encoded_labels = F.one_hot(labels, config.NUM_CLASSES).to(device).type(dtype=torch.float32)
    imgs = generator(z, encoded_labels)
    return imgs, encoded_labels

def evaluate(nets, loader: DataLoader, criterion, device) -> float:
    discriminator = nets["discriminator"]
    generator = nets["generator"]

    discriminator.eval()
    generator.eval()

    total_loss_d = 0
    total_loss_g = 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            real_imgs = imgs.flatten(start_dim=1).to(device)
            real_labels = F.one_hot(labels, config.NUM_CLASSES).to(device)

            preds_real = discriminator(real_imgs, real_labels)
            loss_d_real = criterion(preds_real, torch.ones((real_labels.shape[0], 1), device=device))

            fake_imgs, fake_labels = generate_random_imgs(generator, real_labels.shape[0], device)

            preds_fake = discriminator(fake_imgs, fake_labels)
            loss_d_fake = criterion(preds_fake, torch.zeros((real_labels.shape[0], 1), device=device))
            loss_g = criterion(preds_fake, torch.ones((real_labels.shape[0], 1), device=device))

            loss_d = loss_d_real + loss_d_fake

            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()
    
    return total_loss_d / len(loader), total_loss_g / len(loader)

def visualize_generated_sample(fname: str,
                               imgs: torch.Tensor,
                               labels: torch.Tensor,
                               show: bool = False):
    n = len(labels)
    plt.figure(figsize=(15, 12))
    for i in range(n):
        plt.subplot(n, 1, i+1)
        plt.title(f"Label: {labels[i].argmax()}")
        plt.imshow(imgs[i].cpu().numpy())
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname)
    if show:
        plt.show()
    plt.close()
    print(f"{fname} is saved.")

def visualize_history(train_loss_history, val_loss_history):
    train_x = range(len(train_loss_history["generator"]))
    val_x = range(len(val_loss_history["generator"]))
    
    plt.title("Train & Validation Loss")
    plt.plot(train_x, train_loss_history["generator"], label="Train_Loss_G")
    plt.plot(train_x, train_loss_history["discriminator"], label="Train_Loss_D")
    plt.plot(val_x, val_loss_history["generator"], label="Validation_Loss_G")
    plt.plot(val_x, val_loss_history["discriminator"], label="Validation_Loss_D")
    plt.legend()
    plt.savefig("./results/loss_history.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", action="store", type=int, default=200)
    parser.add_argument("--lr_g", dest="lr_g", action="store", type=float, default=1e-4)
    parser.add_argument("--lr_d", dest="lr_d", action="store", type=float, default=1e-4)
    parser.add_argument("--weight_decay_g", dest="weight_decay_g", action="store", type=float, default=1e-3)
    parser.add_argument("--weight_decay_d", dest="weight_decay_d", action="store", type=float, default=1e-3)
    parser.add_argument("--device", dest="device", action="store", type=str, default="cpu")
    parser.add_argument("--split_ratio", dest="split_ratio", action="store", type=float, default=0.2)
    parser.add_argument("--batch_size", dest="batch_size", action="store", type=int, default=32)
    parser.add_argument("--pretrained", dest="pretrained", action="store", type=str, default=None)
    parser.add_argument("--dropout", dest="dropout", action="store", type=float, default=0.3)
    parser.add_argument("--train", dest="train", action="store_true")
    args = parser.parse_args()

    main(args)