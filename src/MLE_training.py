import torch
from tqdm import tqdm 

from src.encode import to_ix
from utils.utils import read_json

import neptune.new as neptune
auth = read_json("configuration/auth.json")
params = read_json("configuration/original_hyper_params.json")

run = neptune.init(
    project= auth["project"],
    api_token= auth["api_token"]
)

run["parameters"] = params

def MLE_train(seq2seq, optimizer, criterion, parent_data_loader, child_data_loader, num_epochs, device):

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

            run["train/loss"].log(loss.item())


run.stop()