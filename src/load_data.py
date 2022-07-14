import torch
import pandas as pd

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


def read_data():

    print("reading clades ...")
    clades = pd.read_csv(clade_assignment_path,"\t")

    print("reading genomic records...")
    genomic_records = read_fasta(genomic_sample)

    print("translating records ...") 
    protein_records = [seq.translate() for seq in genomic_records]

    
    #drop unassigned sequences
    indexes_to_drop = clades[clades["clade"]=="recombinant"].index

    clades = clades.reset_index()
    clades = clades.drop(index=indexes_to_drop )

    for index in sorted(indexes_to_drop, reverse=True):
        del protein_records[index]

    return clades, protein_records


def create_pairs(clades, protein_records):
    #parent clades with respect to their child clades
    clades_in_out = read_json(clades_in_out_path)

    #compose parent-child pairs
    parents = []
    children = []
    for parent_clade in clades_in_out.keys():
        for child_clade in clades_in_out[parent_clade]:
            parent_index = clades[clades["clade"]==parent_clade].index
            child_index = clades[clades["clade"]==child_clade].index
            for p in parent_index:
                for c in child_index:
                    parents.append(str(protein_records[p].seq))
                    children.append(str(protein_records[c].seq))

    return parents, children


def load_data(parents, children, batch_size):
    training_parents = torch.utils.data.DataLoader(
    BiologicalSequenceDataset(parents),
    batch_size,
    collate_fn=collate_fn
    )

    training_children = torch.utils.data.DataLoader(
        BiologicalSequenceDataset(children),
        batch_size,
        collate_fn=collate_fn
    )

    return training_parents, training_children
    

