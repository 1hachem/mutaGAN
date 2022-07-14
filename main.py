from src.load_data import read_data, create_pairs, load_data, fast_load 
from src.load_data import BiologicalSequenceDataset, collate_fn # used in case of is_fast_load=True
from utils.utils import read_json

params = read_json("configuration/original_hyper_params.json")
is_fast_load = True #use when you already have pickled data loader 

if is_fast_load:
    parent_data_loader, child_data_loader = fast_load()
    print("success")

else:
    clades, protein_records = read_data()
    parents, children = create_pairs(clades, protein_records)

    parent_data_loader, child_data_loader = load_data(parents, children, params["MLE_batch_size"])

    print("success")