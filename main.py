from src.load_data import read_data, create_pairs, load_data 
from utils.utils import read_json

params = read_json("configuration/original_hyper_params.json")


clades, protein_records = read_data()
parents, children = create_pairs(clades, protein_records)

parent_data_loader, child_data_loader = load_data(parents, children, params["MLE_batch_size"])

print("success")