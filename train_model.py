"""Train function"""

import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils import RetDataset
from models import get_mobilenet_v3
# hyperparameters
lr = 0.01
epochs = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, dataloader, criterion, optimizer, save=True, checkpt_path=None) -> None:
    """train model and save it if save is true"""
    model.train()
    model.to(device)
    checkpt = {}
    for epoch in range(epochs):
        acc = 0
        for x, y in dataloader:
            x.to(device)
            y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y.flatten())
            loss.backward()
            acc += (y_pred.argmax(dim=1) == y.flatten()).sum()
            optimizer.zero_grad()
            optimizer.step()
        
        if epoch%100 == 0 and save:
            checkpt['model_st_dict'] = model.state_dict()
            checkpt['acc'] = acc
            checkpt['loss'] = loss
            if checkpt_path is None:
                checkpt_path = ''
            checkpt_file_name =  f'checkpoint_{epoch}.pth'
            checkpt_file_path = os.path.join(checkpt_path, checkpt_file_name)
            torch.save(checkpt, checkpt_file_path)
            
        print(f'{epoch} epochs: {loss.item()} acc = {acc/len(dataloader.dataset)}')

if __name__=="__main__":
    dataset_path = 'datasets/ret_dataset/'
    annot_path = 'annot_balanced_200.csv'

    dataset_pretrained = RetDataset(dataset_path, annot_path)
    dataloader_pretrained = DataLoader(dataset_pretrained, batch_size=16, shuffle=True)
    
    model = get_mobilenet_v3()
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, dataloader_pretrained, criterion, optimizer)