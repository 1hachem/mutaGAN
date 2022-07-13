import json
  

def read_json(file):
# Opening JSON file
    with open(file) as f: 
        # returns JSON object as 
        # a dictionary
        data = json.load(f)

    return data