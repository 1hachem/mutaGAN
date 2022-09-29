import json
from Bio import SeqIO

def read_json(file):
# Opening JSON file
    with open(file) as f: 
        # returns JSON object as 
        # a dictionary
        data = json.load(f)

    return dat

paths = read_json("configuration/files.json")

def read_fasta(file):
    return list(SeqIO.parse(paths["genomic_sample"], "fasta"))

def write_fasta(sequences,file): #TODO write fasta
    pass 