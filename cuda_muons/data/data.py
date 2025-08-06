import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
import pickle

with gzip.open('/home/hep/lprate/projects/cuda_muons/cuda_muons/data/multi_2/joined.pkl', 'rb') as f:
    data = pickle.load(f)  

def random_choice(steps):
    step = steps[-1]-steps[-2]
    step = (0.05-steps[-2])/step
    assert 0 <= step <= 1
    if np.random.uniform(0, 1) < step:
        return -1
    else:
        return -2

print("Data Loaded", data.shape)