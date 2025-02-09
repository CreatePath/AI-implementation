import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from evaluation import evaluate
from utils import generate_samples, plot_samples


class GANTrainer:
    def __init__(self, discriminator,
                 generator,
                 opt_d,
                 opt_g,
                 criterion,
                 model_dir,
                 sample_dir,
                 history_dir,
                 device):
        self.discriminator = discriminator
        self.generator = generator
        self.criterion = criterion
        self.opt_d = opt_d
        self.opt_g = opt_g

        self.model_dir = model_dir
        self.sample_dir = sample_dir
        self.history_dir = history_dir

        self.device = device
        self.history = {"loss_train_d": [],
                        "loss_train_g": [],
                        "loss_val_d": [],
                        "loss_val_g": [],}
    
    def train(self, tr_loader, val_loader, epochs):
        self.discriminator.eval()
        self.generator.eval()

        for i in range(epochs):
            loss_train_d, loss_train_g = self.train_one_epoch(tr_loader, i+1)
            loss_val_d, loss_val_g = evaluate(self.discriminator, self.generator, val_loader, self.criterion, self.device)

            self.history["loss_train_d"].append(loss_train_d)
            self.history["loss_train_g"].append(loss_train_g)
            self.history["loss_val_d"].append(loss_val_d)
            self.history["loss_val_g"].append(loss_val_g)

            print("Epoch {}: Train Loss D: {:.5f}, G: {:.5f}, Val Loss D: {:.5f}, G: {:.5f}".format(i+1,
                                                                                                    loss_train_d,
                                                                                                    loss_train_g,
                                                                                                    loss_val_d,
                                                                                                    loss_val_g))
                            
            samples = generate_samples(self.generator, config.N_SAMPLES, self.device)

            title = "Samples in Training Epoch {}".format(i+1)
            file_path = "{}/sample_epoch_{}.png".format(self.sample_dir, i+1)
            plot_samples(samples, title, file_path)

            if (i+1) % 10 == 0:
                torch.save({"discriminator": self.discriminator.state_dict(),
                            "generator": self.generator.state_dict(),
                            "history": self.history,}, "{}/gan_train_{}.pt".format(self.model_dir, i+1))
        
            self.plot_history()

        return self.history

    def train_one_epoch(self, tr_loader, epoch):
        self.discriminator.train()
        self.generator.train()
        total_loss_d, total_loss_g = 0, 0

        with tqdm(tr_loader) as pbar:
            pbar.set_description("Epoch {}".format(epoch))
            for real, _ in pbar:
                # Train Discriminator
                real = real.unsqueeze(1).to(self.device)
                ones = torch.ones((real.shape[0], 1), device=self.device, dtype=torch.float32)

                z = torch.randn((real.shape[0], config.DIM_Z), device=self.device)
                fake = self.generator(z)
                zeros = torch.zeros((real.shape[0], 1), device=self.device, dtype=torch.float32)

                pred_real = self.discriminator(real)
                loss_d_real = self.criterion(pred_real, ones)

                pred_fake = self.discriminator(fake.detach())
                loss_d_fake = self.criterion(pred_fake, zeros)
                
                loss_d = loss_d_real + loss_d_fake

                self.opt_d.zero_grad()
                loss_d.backward()
                self.opt_d.step()

                # Train Generator
                z = torch.randn((real.shape[0], config.DIM_Z), device=self.device)

                fake = self.generator(z)
                pred_fake = self.discriminator(fake)

                loss_g = self.criterion(pred_fake, ones)

                self.opt_g.zero_grad()
                loss_g.backward()
                self.opt_g.step()

                total_loss_d += loss_d.item()
                total_loss_g += loss_g.item()

                pbar.set_postfix(loss_d=loss_d.item(),
                                 loss_d_real=loss_d_real.item(),
                                 loss_d_fake=loss_d_fake.item(),
                                 loss_g=loss_g.item())

        total_loss_d /= len(tr_loader)
        total_loss_g /= len(tr_loader)
        
        return total_loss_d, total_loss_g

    def plot_history(self):
        x_train = range(len(self.history["loss_train_d"]))

        plt.title("Loss History in Training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(x_train, self.history["loss_train_d"], label="loss_train_d")
        plt.plot(x_train, self.history["loss_train_g"], label="loss_train_g")
        plt.plot(x_train, self.history["loss_val_d"], label="loss_val_d")
        plt.plot(x_train, self.history["loss_val_g"], label="loss_val_g")
        plt.legend()

        plt.savefig("{}/{}".format(self.history_dir, "history.png"))