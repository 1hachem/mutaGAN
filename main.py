import torch
import torch.nn as nn
from torch.optim import Adam

import neptune.new as neptune

from src.model import Encoder, Decoder, Seq2Seq, Classifier
from src.load_data import read_data, create_pairs, load_data, fast_load, save_loaders 
from src.load_data import BiologicalSequenceDataset, collate_fn # used in case of is_fast_load=True
from src.MLE_training import MLE_train
from src.GAN_training import GAN_train
from src.encode import to_ix

from utils.utils import read_json

device = "cuda" if torch.cuda.is_available() else "cpu"

is_fast_load = True #use when you already have pickled dataloaders for both parent and child sequences
is_neptune = False #is you enable neptune (set configuration/auth.json) 

paths = read_json("configuration/files.json")
params = read_json(paths["hyper_params"])

#Generator
encoder = Encoder(input_size=params["vocab_size"], embedding_size=params["encoder_emb_size"], 
        hidden_size=params["encoder_hidden_size"], num_layers=params["encoder_num_layers"], device=device)

decoder = Decoder(input_size=params["vocab_size"], embedding_size=params["decoder_emb_size"], 
        hidden_size=params["decoder_hidden_size"], output_size=params["vocab_size"], num_layers=params["decoder_num_layers"], device=device)

seq2seq = Seq2Seq(encoder, decoder, teacher_forcing_ratio=params["teacher_forcing_ratio"], device=device)

#Discriminator
encoder_disc = Encoder(input_size=params["vocab_size"], embedding_size=params["encoder_emb_size"], 
        hidden_size=params["encoder_hidden_size"], num_layers=params["encoder_num_layers"], disc=True, device=device)

classifier = Classifier(params["encoder_hidden_size"]*4, device=device)

#optimizers
optimizer_generator = Adam(seq2seq.parameters(), lr= params["MLE_learning_rate"])
optimizer_discriminator = Adam(classifier.parameters(), lr= params["GAN_learning_rate"])

#criterion
MLE_criterion = nn.CrossEntropyLoss(ignore_index= to_ix["<pad>"])

#dataloaders
if is_fast_load:
    parent_data_loader, child_data_loader, not_child_data_loader = fast_load()

else:
    clades, protein_records = read_data()
    parents, children, not_children = create_pairs(clades, protein_records)

    parent_data_loader, child_data_loader, not_child_data_loader = load_data(parents, children, not_children, params["MLE_batch_size"])
    save_loaders(parent_data_loader, child_data_loader, not_child_data_loader)

#### monitoring with neptune.ai
if is_neptune:
    auth = read_json("configuration/auth.json")

    neptune_run = neptune.init(
        project=auth["project"],
        api_token=auth["api_token"],
    )
    neptune_run["parameters"] = params
else:
    neptune_run = None

#### MLE training
print("MLE training ...")
MLE_train(seq2seq, optimizer_generator, MLE_criterion, parent_data_loader, child_data_loader, num_epochs=params["MLE_num_epochs"], device=device, neptune_run=neptune_run)
torch.save(seq2seq.state_dict(), paths["saved_seq2seq_MLE"])

#### GAN training 
print("GAN training ...")
seq2seq.set_teacher_forcing_ratio(params["GAN_teacher_forcing_ratio"]) #set teacher forcing to 0

for g in optimizer_generator.param_groups: #changing the learning rate of the optimizer
    g['lr'] = params["GAN_learning_rate"]

GAN_train(seq2seq, encoder_disc, classifier, optimizer_generator, optimizer_discriminator, parent_data_loader, child_data_loader, not_child_data_loader,
                params["GAN_num_epochs"], device, neptune_run)
torch.save(seq2seq.state_dict(), paths["saved_seq2seq_GAN"])
torch.save(classifier.state_dict(), paths["saved_classifier_GAN"])

if is_neptune:
    neptune_run.stop()