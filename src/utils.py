import pickle
import json

def pickle_load(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        obj = pickle.load(pkl_file)
    return obj


def json_load(json_path):
    with open(json_path, 'r') as json_file:
        obj = json.load(json_file)
    return obj

def chunk_list(input_list, batch_size):
    chunklist = [input_list[x:x + batch_size] for
                x in range(0, len(input_list), batch_size)]
    return chunklist

def str2bool(var):
    if isinstance(var, bool):
        return var
    elif var.lower() in ['false', 'f', 0]:
        return False
    elif var.lower() in ['true', t, 1]:
        return True
    else:
        raise ValueError(f'Expecting boolean variable, but got {var}')