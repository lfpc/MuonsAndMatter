import gzip
import pickle
import time
import numpy as np
import multiprocessing as mp
from run_simulation import run
import subprocess


DEF_PARAMS = [208.0, 207.0, 281.0, 248.0, 305.0, 242.0, 72.0, 51.0, 29.0, 46.0, 10.0, 7.0, 54.0, 38.0, 46.0, 192.0, 14.0, 9.0, 10.0, 31.0, 35.0, 31.0, 51.0, 11.0, 3.0, 32.0, 54.0, 24.0, 8.0, 8.0, 22.0, 32.0, 209.0, 35.0, 8.0, 13.0, 33.0, 77.0, 85.0, 241.0, 9.0, 26.0]
DEF_INPUT_FILE = 'data/inputs.pkl'#'data/oliver_data_enriched.pkl'
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("-c", type=int, default=45)
    parser.add_argument("-f", type=str, default=DEF_INPUT_FILE)
    parser.add_argument("-tag", type=str, default='geant4')
    parser.add_argument("-params", nargs='+', default=DEF_PARAMS)
    args = parser.parse_args()
    tag = args.tag
    cores = args.c
    params=list(args.params)
    n_muons = args.n
    input_file = args.f

    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)
    np.random.shuffle(data)
    if 0<n_muons<=data.shape[0]:
        data = data[:n_muons]
        cores = min(cores,n_muons)
    data = data[0:cores*int(len(data) / cores)]
    data[:,5] = 17*np.ones_like(data[:,5]) #how to pass correctly the z data?
    division = int(len(data) / cores)

    workloads = []
    for i in range(cores):
        workloads.append(data[i * division:(i + 1) * division, :])

    t1 = time.time()
    with mp.Pool(cores) as pool:
        result = pool.starmap(run, [(workload,params,50,True) for workload in workloads])
    t2 = time.time()

    all_results = []
    for i, rr in enumerate(result):
        resulting_data,weight = rr
        all_results += [resulting_data]

    print(f"Workload of {division} samples spread over {cores} cores took {t2 - t1:.2f} seconds.")
    all_results = np.concatenate(all_results, axis=0)
    with gzip.open(f'data/results_{tag}.pkl', 'wb') as f:
        pickle.dump(all_results, f)


    
