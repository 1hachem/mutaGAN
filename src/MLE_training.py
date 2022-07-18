import torch
from tqdm import tqdm 

from src.encode import to_ix
from utils.utils import read_json


def MLE_train(seq2seq, optimizer, criterion, parent_data_loader, child_data_loader, num_epochs, device, neptune_run=None):

    for epoch in range(num_epochs):
        loop = tqdm(zip(parent_data_loader, child_data_loader), total=len(parent_data_loader))
        for parent, child in loop:
        
            parent = parent.to(device)
            child = child.to(device)

            #feed forward 
            output = seq2seq(parent, child, sos_token=to_ix["<sos>"])
            loss = criterion(output, child[:,:output.shape[-1]])

            #backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            try:
                neptune_run["train_MLE/loss"].log(loss.item())
            except:
                print("error connecting to neptune")
