from tqdm import tqdm, tqdm_notebook
from Sequences import prep_data
import torch
import numpy as np
train_dataset,train_loader,test_dataset,test_loader=prep_data()
def LNtrain(model,optimizer,criterion,):
    model.train()
    train_losses = []
    for epoch in range(10):
        progress_bar = tqdm_notebook(train_loader, leave=False)
        losses = []
        total = 0
        for inputs, target in progress_bar:
            # Inputs are of shape (bs, 1, voc size), we remove the 1 with squeeze() and we convert them to floats
            inputs = inputs.squeeze().float()  # Convertir l'entrée en float
            targets = target.float()
            ### TODO: implement the training loop as usual.
            model.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'Loss: {loss.item():.3f}')
            losses.append(loss.item())
            total += 1
        epoch_loss = sum(losses) / total
        train_losses.append(epoch_loss)
        tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')
    return model,train_losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def LNtest(model,test_loader):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(test_loader):
            features = features.to(device)
            targets = targets.float().to(device)
            logits = model(features.squeeze().float())
            predicted_labels = torch.round(logits)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets.squeeze()).sum()
    accuracy=np.round(float((correct_pred.float()/num_examples)),4) * 100
    return accuracy
