import torch
from tqdm import tqdm 

from src.encode import to_ix
from utils.utils import read_json


def MLE_train(seq2seq, optimizer, criterion, parent_data_loader, child_data_loader, num_epochs, device, neptune_run=None):

    for epoch in range(num_epochs):
        progress_bar = tqdm(zip(parent_data_loader, child_data_loader), total=len(parent_data_loader))
        for parent, child in progress_bar:
        
            parent = parent.to(device)
            child = child.to(device)

            #feed forward 
            output, _ = seq2seq(parent, child, sos_token=to_ix["<sos>"])
            loss = criterion(output, child[:,:output.shape[-1]])

            #backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            try:
                neptune_run["train_MLE/loss"].log(loss.item())
            except:
                progress_bar.set_description("error connecting to neptune")
