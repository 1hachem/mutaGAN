import json
from Bio import SeqIO

  

def read_json(file):
# Opening JSON file
    with open(file) as f: 
        # returns JSON object as 
        # a dictionary
        data = json.load(f)

    return data


def read_fasta(file):
    return list(SeqIO.parse("data/ncbidata/genomic.fna", "fasta"))