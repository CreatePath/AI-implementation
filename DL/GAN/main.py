import torch
import argparse
from torch.optim import Adam
from torch.nn import BCELoss

import config
from data import get_dataloader
from discriminator import Discriminator
from generator import Generator
from trainer import GANTrainer
from evaluation import evaluate

def main(args):
    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    tr_loader, val_loader, test_loader = get_dataloader()

    discriminator = Discriminator(config.DIM_X, config.HIDDEN_D, config.DIM_Y, config.DROPOUT_D).to(device)
    generator = Generator(config.DIM_Z, config.HIDDEN_G, config.HEIGHT, config.WIDTH, config.DROPOUT_G).to(device)

    opt_d = Adam(discriminator.parameters(), lr=args.lr_d, weight_decay=args.weight_decay_d)
    opt_g = Adam(generator.parameters(), lr=args.lr_g, weight_decay=args.weight_decay_g)
    criterion = BCELoss()

    gan_trainer = GANTrainer(discriminator,
                             generator,
                             opt_d,
                             opt_g,
                             criterion,
                             args.model_dir,
                             args.sample_dir,
                             args.history_dir,
                             device)
    
    history = gan_trainer.train(tr_loader, val_loader, args.epochs)
    loss_d, loss_g = evaluate(discriminator, generator, test_loader, criterion, device)
    print("Final Test Loss D: {}, G: {}".format(loss_d, loss_g))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--device", action="store", dest="device", type=str, default="cpu")

    # Optimizer Hyper-parameters
    parser.add_argument("-lr_d", "--learning_rate_d", action="store", dest="lr_d", type=float, default=1e-3)
    parser.add_argument("-lr_g", "--learning_rate_g", action="store", dest="lr_g", type=float, default=1e-5)
    parser.add_argument("-wd_d", "--weight_decay_d", action="store", dest="weight_decay_d", type=float, default=1e-8)
    parser.add_argument("-wd_g", "--weight_decay_g", action="store", dest="weight_decay_g", type=float, default=1e-9)
    parser.add_argument("-e", "--epochs", action="store", dest="epochs", type=int, default=20)

    # Output Directories
    parser.add_argument("-md", "--model_dir", action="store", dest="model_dir", type=str, default="./model")
    parser.add_argument("-sd", "--sample_dir", action="store", dest="sample_dir", type=str, default="./samples")
    parser.add_argument("-hd", "--history_dir", action="store", dest="history_dir", type=str, default="./history")

    args = parser.parse_args()

    main(args)