import torch
import pandas as pd
import pickle

from src.encode import to_ix
from utils.utils import read_json, read_fasta

class BiologicalSequenceDataset:
    def __init__(self, sequences):
        self.records = sequences

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        seq = self.records[i]
        return torch.tensor([to_ix[residue] for residue in seq])

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(
        batch,
        batch_first=True,
        padding_value=to_ix["<pad>"]
    )

files = read_json("configuration/files.json")

clade_assignment_path = files["clade_assignment"]
genomic_sample = files["genomic_sample"]
clades_in_out_path = files["clades_in_out"]
parent_pickeled_loader, child_pickeled_loader, not_child_pickeled_loader = (files["parent_pickeled_loader"], 
                                                                            files["child_pickeled_loader"], 
                                                                            files["not_child_pickeled_loader"])


def read_data():

    print("reading clades ...")
    clades = pd.read_csv(clade_assignment_path,"\t")

    print("reading genomic records...")
    genomic_records = read_fasta(genomic_sample)

    print("translating records ...") 
    protein_records = [seq.translate() for seq in genomic_records]

    
    #drop unassigned sequences
    indexes_to_drop = clades[clades["clade"]=="recombinant"].index

    clades = clades.drop(index=indexes_to_drop)
    clades = clades.reset_index()

    for index in sorted(indexes_to_drop, reverse=True):
        del protein_records[index]

    return clades, protein_records


def create_pairs(clades, protein_records):
    #parent clades with respect to their child clades
    clades_in_out = read_json(clades_in_out_path)

    #compose parent-child pairs
    parents = []
    children = []
    not_children = []
    for parent_clade in clades_in_out.keys():
        for child_clade in clades_in_out[parent_clade]:
            parent_index = clades[clades["clade"]==parent_clade].index
            child_index = clades[clades["clade"]==child_clade].index
            not_child = clades[clades["clade"]!=child_clade].index
            for p in parent_index:
                for c,n in zip(child_index, not_child):
                    parents.append(str(protein_records[p].seq))
                    children.append(str(protein_records[c].seq))
                    not_children.append(str(protein_records[n].seq))
                    

    return parents, children, not_children


def load_data(parents, children, not_children, batch_size):
    parent_data_loader = torch.utils.data.DataLoader(
    BiologicalSequenceDataset(parents),
    batch_size,
    collate_fn=collate_fn
    )

    child_data_loader = torch.utils.data.DataLoader(
        BiologicalSequenceDataset(children),
        batch_size,
        collate_fn=collate_fn
    )

    not_child_data_loader = torch.utils.data.DataLoader(
        BiologicalSequenceDataset(not_children),
        batch_size,
        collate_fn=collate_fn
    )

    return parent_data_loader, child_data_loader, not_child_data_loader


def save_loaders(parent_data_loader, child_data_loader, not_child_data_loader):
    pickle.dump(parent_data_loader, open(parent_pickeled_loader, "wb"))
    pickle.dump(child_data_loader, open(child_pickeled_loader, "wb"))
    pickle.dump(not_child_data_loader, open(not_child_pickeled_loader, "wb"))
    print("saved at: ", parent_pickeled_loader + " and "+ child_pickeled_loader)
    

def fast_load():
    parent_data_loader = pickle.load(open(parent_pickeled_loader, "rb"))  
    child_data_loader = pickle.load(open(child_pickeled_loader, "rb"))
    not_child_data_loader = pickle.load(open(not_child_pickeled_loader, "rb"))
    return parent_data_loader, child_data_loader, not_child_data_loader