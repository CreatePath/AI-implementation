import config
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.datasets.mnist import FashionMNIST, MNIST
from torchvision.transforms import transforms

def get_dataloader():
    dir = config.DATADIR
    train_ratio = config.TRAIN_RATIO
    val_ratio = config.VAL_RATIO

    transform = transforms.Compose([transforms.ToTensor(),])
    train_dset = FashionMNIST(dir, train=True, transform=transform, download=True)
    test_dset = FashionMNIST(dir, train=False, transform=transform, download=True)
    # train_dset = MNIST(dir, train=True, transform=transform, download=True)
    # test_dset = MNIST(dir, train=False, transform=transform, download=True)

    tr_samples = int(len(train_dset) * train_ratio)
    val_samples = len(train_dset) - tr_samples

    tr_dset, val_dset = random_split(train_dset, [tr_samples, val_samples])

    tr_loader = DataLoader(tr_dset, batch_size=config.BATCH_SIZE)
    val_loader = DataLoader(val_dset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dset, batch_size=config.BATCH_SIZE)

    return tr_loader, val_loader, test_loader