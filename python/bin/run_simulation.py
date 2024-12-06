
import json
import numpy as np
from lib.ship_muon_shield_customfield import get_design_from_params
from muon_slabs import simulate_muon, initialize, collect, kill_secondary_tracks, collect_from_sensitive
from copy import deepcopy
from plot_magnet import plot_magnet, construct_and_plot


def run(muons, 
        phi, 
        z_bias:float=50, 
        input_dist:float = None,
        return_weight = False,
        fSC_mag:bool = True,
        sensitive_film_params:dict = {'dz': 0.01, 'dx': 4, 'dy': 6,'position':67},
        use_simulated_fields = False,
        return_nan:bool = False,
        seed:int = None,
        draw_magnet = False,
        kwargs_plot = {},):

    if type(muons) is tuple:
        muons = muons[0]

    detector = get_design_from_params(params = phi,
                                      z_bias=z_bias,
                                      force_remove_magnetic_field=False,
                                      fSC_mag = fSC_mag, 
                                      use_simulated_fields=use_simulated_fields,
                                      sensitive_film_params=sensitive_film_params,)

    detector["store_primary"] = False
    detector["store_all"] = False

    
    if seed is None: output_data = initialize(np.random.randint(256), np.random.randint(256), np.random.randint(256), np.random.randint(256), json.dumps(detector))
    else: output_data = initialize(seed,seed, seed, seed, json.dumps(detector))
    output_data = json.loads(output_data)

    # set_field_value(1,0,0)
    # set_kill_momenta(65)
    kill_secondary_tracks(True)
    if muons.shape[-1] == 8: px,py,pz,x,y,z,charge,W = muons.T
    else: px,py,pz,x,y,z,charge = muons.T
    

    if (np.abs(charge)==13).all(): charge = charge/(-13)
    assert((np.abs(charge)==1).all())

    if input_dist is not None:
        z_pos = detector['magnets'][0]['z_center'] - detector['magnets'][0]['dz']-input_dist
        z = z_pos*np.ones_like(z)
    else:
        z = z/100 + 70.845 - 68.685 + 66.34
        z = detector['magnets'][0]['z_center'] - detector['magnets'][0]['dz'] + z
    muon_data_s = []
    #muon_data = []
    for i in range(len(px)):
        simulate_muon(px[i], py[i], pz[i], int(charge[i]), x[i],y[i], z[i])
        #data = collect()
        data_s = collect_from_sensitive()
        if len(data_s['px'])>0 and 13 in np.abs(data_s['pdg_id']): 
            #muon_data += [[data['px'][-1], data['py'][-1], data['pz'][-1],data['x'][-1], data['y'][-1], data['z'][-1]]]
            j = 0
            while int(abs(data_s['pdg_id'][j])) != 13:
                j += 1
            output_s = [data_s['px'][j], data_s['py'][j], data_s['pz'][j],data_s['x'][j], data_s['y'][j], data_s['z'][j],data_s['pdg_id'][j]]
            if muons.shape[-1] == 8: output_s.append(W[i])
            muon_data_s += [output_s]
        elif return_nan:
            muon_data_s += [[0]*muons.shape[-1]]
    #muon_data = np.asarray(muon_data)
    muon_data_s = np.asarray(muon_data_s)
    if draw_magnet: 
        plot_magnet(detector,
                muon_data = muon_data_s, 
                z_bias = z_bias,
                sensitive_film_position = 5,#sensitive_film_params['position'], 
                **kwargs_plot)
    if return_weight: return muon_data_s, output_data['weight_total']
    else: return muon_data_s



DEF_INPUT_FILE = 'data/inputs.pkl'#'data/oliver_data_enriched.pkl'
if __name__ == '__main__':
    import argparse
    import gzip
    import pickle
    import time
    import multiprocessing as mp
    from lib.reference_designs.params import sc_v6
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--c", type=int, default=45)
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("--f", type=str, default=DEF_INPUT_FILE)
    parser.add_argument("-tag", type=str, default='geant4')
    parser.add_argument("-params", nargs='+', default=sc_v6)
    parser.add_argument("--z", type=float, default=0.9)
    parser.add_argument("-sens_plane", type=float, default=67)
    parser.add_argument("-real_fields", action = 'store_true')
    parser.add_argument("-shuffle_input", action = 'store_true')
    parser.add_argument("-plot_magnet", action = 'store_true')
    parser.add_argument("-return_nan", action = 'store_true')
    

    args = parser.parse_args()
    tag = args.tag
    cores = args.c
    params = params=list(args.params)#np.array(data)
    def split_array(arr, K):
        N = len(arr)
        base_size = N // K
        remainder = N % K
        sizes = [base_size + 1 if i < remainder else base_size for i in range(K)]
        splits = np.split(arr, np.cumsum(sizes)[:-1])
        return splits

    n_muons = args.n
    input_file = args.f
    z_bias = 50
    input_dist = args.z
    sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6, 'position':args.sens_plane}

    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)
    if args.shuffle_input: np.random.shuffle(data)
    if 0<n_muons<=data.shape[0]:
        data_n = data[:n_muons]
        cores = min(cores,n_muons)
    else: data_n = data

    workloads = split_array(data_n,cores)
    t1 = time.time()
    with mp.Pool(cores) as pool:
        result = pool.starmap(run, [(workload,params,z_bias,input_dist,True,True,sensitive_film_params, #args.real_fields,
                                    args.return_nan,args.seed, args.plot_magnet) for workload in workloads])
    t2 = time.time()
    print(f"Workload of {np.shape(workloads[0])[0]} samples spread over {cores} cores took {t2 - t1:.2f} seconds.")
    if args.plot_magnet:
        all_results = []
        for rr in result:
            resulting_data,weight = rr
            if len(resulting_data)==0: continue
            all_results += [resulting_data]
        print(f"Weight = {weight} kg")
        all_results = np.concatenate(all_results, axis=0)
        #with gzip.open(f'data/outputs/outputs_{tag}.pkl', "wb") as f:
        #    pickle.dump(all_results, f)
        print('Data Shape', all_results.shape)
        sensitive_film_params['position'] = 5
        with mp.Pool(1) as pool:
            result = pool.starmap(construct_and_plot, [(all_results,params,z_bias,True,sensitive_film_params)])

