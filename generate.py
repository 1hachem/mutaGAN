import torch
import torch.nn as nn

from src.model import Encoder, Decoder, Seq2Seq
from src.load_data import fast_load
from src.encode import to_ix, inv_to_ix

from utils.utils import read_json, write_fasta


device = "cuda" if torch.cuda.is_available() else "cpu"

paths = read_json("configuration/files.json")
params = read_json(paths["hyper_params"])

#Generator
encoder = Encoder(input_size=params["vocab_size"], embedding_size=params["encoder_emb_size"], 
        hidden_size=params["encoder_hidden_size"], num_layers=params["encoder_num_layers"], device=device)

decoder = Decoder(input_size=params["vocab_size"], embedding_size=params["decoder_emb_size"], 
        hidden_size=params["decoder_hidden_size"], output_size=params["vocab_size"], num_layers=params["decoder_num_layers"], device=device)

seq2seq = Seq2Seq(encoder, decoder, teacher_forcing_ratio=params["teacher_forcing_ratio"], device=device)

#load trained models
seq2seq.load_state_dict(torch.load(paths["saved_seq2seq_GAN"]))

#parent loader TODO get future parent sequences (date split trick)
parent_data_loader, child_data_loader, _= fast_load()



#generate
with torch.no_grad():
    parent = iter(parent_data_loader).next()
    parent = parent.to(device)

    generated_child, _= seq2seq(parent, None, sos_token=to_ix["<sos>"], inference=True)

#save gnerated sequences
print([[inv_to_ix[aa.item()] for aa in seq ] for seq in generated_child.argmax(-1)], paths["generated_sequences"])


