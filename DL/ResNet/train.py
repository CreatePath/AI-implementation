import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from eval import evaluate

def train(model: nn.Module, optimizer, scheduler, train_loader, val_loader, loss_fn, epochs, device, version):
    history = {"loss_train": [], "loss_val": [], "acc_train": [], "acc_val": []}
    for epoch in range(epochs):
        loss_train, acc_train = train_one_epoch(model, optimizer, train_loader, loss_fn, device)
        history["loss_train"].append(loss_train)
        history["acc_train"].append(acc_train)

        loss_val, acc_val = evaluate(model, val_loader, loss_fn, device)
        history["loss_val"].append(loss_val)
        history["acc_val"].append(acc_val)

        print(f"{epoch + 1} / {epochs} Train Loss {loss_train} Train acc {acc_train} Val Loss {loss_val} Val acc {acc_val}")

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save({"resnet": model.state_dict()}, f"./pretrained/{version}_{epoch+1}.pkl")
            print("Model saved.")
    
    return history

def train_one_epoch(model, optimizer, train_loader: DataLoader, loss_fn, device):
    model.train()
    batches = len(train_loader)
    acc = 0
    total_loss = 0
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)

        pred = model(img)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_samples = torch.sum(torch.argmax(label, 1) == torch.argmax(pred, 1))
        acc += correct_samples.item() 
        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"{i+1} / {batches}: Loss {loss.item()} Acc {correct_samples / label.shape[0]}")
    
    total_loss /= batches
    acc /= len(train_loader.dataset)

    return total_loss, acc
