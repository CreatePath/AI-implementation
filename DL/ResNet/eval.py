import torch
from torch.utils.data import DataLoader

def evaluate(model, val_loader: DataLoader, loss_fn, device):
    model.eval()
    batches = len(val_loader)
    acc = 0
    total_loss = 0

    with torch.no_grad():
        for i, (img, label) in enumerate(val_loader):
            img = img.to(device)
            label = label.to(device)

            pred = model(img)
            loss = loss_fn(pred, label)

            acc += torch.sum(torch.argmax(label, 1) == torch.argmax(pred, 1)).item()
            total_loss += loss.item()
    
    total_loss /= batches
    acc /= len(val_loader.dataset)

    return total_loss, acc