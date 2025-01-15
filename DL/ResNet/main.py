import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

import random
import numpy as np
import argparse
import matplotlib.pyplot as plt

from config.resnet_config import get_resnet_config
from network import ResNet
from train import train
from eval import evaluate

SEED = 42
NUM_CLASS = 10

def main(args):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print(args)
    device = args.device
    if 0 <= device.find("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print("device:", device)

    resnet_cfg = get_resnet_config(args.version)
    resnet = ResNet(resnet_cfg, NUM_CLASS).to(device)
    print(resnet)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=device)
        resnet.load_state_dict(checkpoint["resnet"])

    optimizer = SGD(resnet.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
    scheduler = args.scheduler
    if scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, args.step_size, args.gamma, verbose=True)

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),])
    target_transform = transforms.Compose([lambda x: torch.tensor(x),
                                           lambda x: F.one_hot(x, NUM_CLASS).type(torch.FloatTensor)])

    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform, target_transform=target_transform)
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform, target_transform=target_transform)

    val_samples = int(len(train_dataset) * args.split_ratio)
    train_samples = len(train_dataset) - val_samples
    test_samples = len(test_dataset)

    print("Train:", train_samples)
    print("Val:", val_samples)
    print("Test:", test_samples)

    tr_set, val_set = random_split(train_dataset, [train_samples, val_samples])

    tr_loader = DataLoader(tr_set, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, batch_size=100)
    test_loader = DataLoader(test_dataset, batch_size=100)

    loss_fn = nn.BCELoss()

    if args.istrain:
        history = train(resnet, optimizer, scheduler, tr_loader, val_loader, loss_fn, args.epochs, device, args.version)

        plt.title("Train & Validation Loss")
        plt.plot(range(len(history["loss_train"])), history["loss_train"])
        plt.plot(range(len(history["loss_val"])), history["loss_val"])
        plt.savefig("./result/loss.png")
        plt.close()
        print("Loss history is saved.")

        plt.title("Train & Validation Accuracy")
        plt.plot(range(len(history["acc_train"])), history["acc_train"])
        plt.plot(range(len(history["acc_val"])), history["acc_val"])
        plt.savefig("./result/acc.png")
        plt.close()
        print("Accuracy history is saved.")
        
    loss, acc = evaluate(resnet, test_loader, loss_fn, device)
    print(f"Test Loss {loss} Test Acc {acc}") # resnet18_30.pkl Loss 0.06679854666814208 Acc 87.31%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ResNet version
    parser.add_argument("-v", "--version", dest="version", type=str, action="store", default="resnet50")

    # device info
    parser.add_argument("-dev", "--device", dest="device", action="store", type=str, default="cpu")

    # do train or not
    parser.add_argument("-t", "--train", dest="istrain", action="store_true")

    # load pretrained model
    parser.add_argument("-p", "--pretrained", dest="pretrained", action="store", type=str, default=None)

    # Hyperparameters for train
    parser.add_argument("-b", "--batch_size", dest="batch_size", action="store", type=int, default=64)
    parser.add_argument("-e", "--epochs", dest="epochs", action="store", type=int, default=100)
    parser.add_argument("-d", "--dataset", dest="dataset", action="store", type=str, default="CIFAR10")
    parser.add_argument("-r", "--split_ratio", dest="split_ratio", action="store", type=float, default=0.2)

    # Optimizer options
    parser.add_argument("-o", "--optimizer", dest="optimizer", action="store", type=str, default="SGD")
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, action="store", default=1e-4)
    parser.add_argument("-m", "--momentum", dest="momentum", action="store", type=float, default=0.9)
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", action="store", type=float, default=1e-4)

    # Scheduler options
    parser.add_argument("-s", "--scheduler", dest="scheduler", action="store_true", default=None)
    parser.add_argument("-st", "--step_size", dest="step_size", action="store", type=int, default=50)
    parser.add_argument("-g", "--gamma", dest="gamma", action="store", type=float, default=0.1)

    args = parser.parse_args()

    main(args)