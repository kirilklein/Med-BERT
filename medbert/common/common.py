import os
import numpy as np
import pandas as pd


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def construct_encodings_from_npz(file):
    """Returns patient keys and encodings from a npz file"""
    npzfile = np.load(file, allow_pickle=True)
    return npzfile['arr_0'], npzfile['arr_1']