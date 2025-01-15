
import json
import numpy as np
from lib.ship_muon_shield_customfield import get_design_from_params, get_field
from muon_slabs import simulate_muon, initialize, collect, kill_secondary_tracks, collect_from_sensitive
from plot_magnet import plot_magnet, construct_and_plot
from time import time
from run_simulation import run


DEF_INPUT_FILE = 'data/inputs.pkl'#'data/oliver_data_enriched.pkl'
if __name__ == '__main__':
    import argparse
    import gzip
    import pickle
    import multiprocessing as mp
    from time import time
    from lib.reference_designs.params import sc_v6
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--c", type=int, default=45)
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("--f", type=str, default=DEF_INPUT_FILE)
    parser.add_argument("-tag", type=str, default='geant4')
    parser.add_argument("-params", nargs='+', default=sc_v6)
    parser.add_argument("--z", type=float, default=0.9)
    parser.add_argument("-sens_plane", type=float, default=83.2)
    parser.add_argument("-real_fields", action = 'store_true')
    parser.add_argument("-field_file", type=str, default='data/outputs/fields.pkl')
    parser.add_argument("-shuffle_input", action = 'store_true')
    parser.add_argument("-remove_cavern", dest = "add_cavern", action = 'store_false')
    parser.add_argument("-plot_magnet", action = 'store_true')
    parser.add_argument("-warm",dest="SC_mag", action = 'store_false')
    

    args = parser.parse_args()
    tag = args.tag
    cores = args.c
    params = list(args.params)#np.array(data)
    def split_array(arr, K):
        N = len(arr)
        base_size = N // K
        remainder = N % K
        sizes = [base_size + 1 if i < remainder else base_size for i in range(K)]
        splits = np.split(arr, np.cumsum(sizes)[:-1])
        return splits

    n_muons = args.n
    input_file = args.f
    input_dist = args.z
    if args.sens_plane is not None: 
        sensitive_film_params = {'dz': 0.01, 'dx': 20, 'dy': 20, 'position':args.sens_plane}
    else: sensitive_film_params = None
    t1_fem = time()
    if args.real_fields: 
        from_file = False
        field = get_field(from_file,np.asarray(params),Z_init = 0., fSC_mag=args.SC_mag,
                                file_name=args.field_file)
    t2_fem = time()

    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)
    if args.shuffle_input: np.random.shuffle(data)
    if 0<n_muons<=data.shape[0]:
        data_n = data[:n_muons]
        cores = min(cores,n_muons)
    else: data_n = data
    workloads = split_array(data_n,cores)
    seed = 0
    for i in range(100):
        seed += 3
        t1 = time()
        with mp.Pool(cores) as pool:
            result = pool.starmap(run, [(workload,params,input_dist,True,args.SC_mag,sensitive_film_params,args.add_cavern, 
                                        args.real_fields,args.field_file,True,seed, False) for workload in workloads])
        t2 = time()
        print(f"Time to FEM: {t2_fem - t1_fem:.2f} seconds.")
        print(f"Workload of {np.shape(workloads[0])[0]} samples spread over {cores} cores took {t2 - t1:.2f} seconds.")
        if args.real_fields:  print('Field SHAPE', np.shape(field['points']), np.shape(field['B']))
        
        all_results = []
        for rr in result:
            resulting_data,weight = rr
            if len(resulting_data)==0: continue
            all_results += [resulting_data]
        print(f"Weight = {weight} kg")
        all_results = np.concatenate(all_results, axis=0)
        with gzip.open(f'/home/hep/lprate/projects/MuonsAndMatter/data/outputs/results_var/outputs_{i}.pkl', "wb") as f:
            pickle.dump(all_results, f)
        print('Data Shape', all_results.shape)
                                         
