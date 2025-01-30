
import json
import numpy as np
from lib.ship_muon_shield_customfield import get_design_from_params, get_field, initialize_geant4
from muon_slabs import simulate_muon, collect, kill_secondary_tracks, collect_from_sensitive
from plot_magnet import plot_magnet, construct_and_plot
from time import time


def run(muons, 
        phi, 
        input_dist:float = None,
        return_weight = False,
        fSC_mag:bool = True,
        sensitive_film_params:dict = {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82},
        add_cavern = True,
        simulate_fields = False,
        field_map_file = None,
        return_nan:bool = False,
        seed:int = None,
        draw_magnet = False,
        SmearBeamRadius:float = 5., #cm
        add_target:bool = True,
        kwargs_plot = {}):
    """
        Simulates the passage of muons through the muon shield and collects the resulting data.
        Parameters:
        muons (ndarray): Array of muon parameters. Each muon is represented by its momentum (px, py, pz), 
                         position (x, y, z), charge, and optionally weight.
        phi: List of 72 parameters for the detector design. 
        input_dist (float, optional): If different than None, define the distance to set the initial z position of all muons. If None (default), the z position is taken from the file.
        return_weight (bool, optional): If True, returns the total weight of the muon shield. Defaults to False.
        fSC_mag (bool, optional): Flag to use the hybrid configuration (i.e., with the superconducting magnet) of the muon shield in the simulation. Defaults to True.
        sensitive_film_params (dict, optional): Parameters for the sensitive film. Defaults to {'dz': 0.01, 'dx': 10, 'dy': 10, 'position': 67}. If None, the simulation collects all the data from the muons (not only hits).
        simulate_fields (bool, optional): Flag to use simulate (FEM) and use field maps in the simulation. Defaults to False.
        field_map_file (str, optional): Path to the field map file. Defaults to None.
        return_nan (bool, optional): If True, returns a list of zeros as the output for muons that do not hit the sensitive film. Defaults to False.
        seed (int, optional): Seed for random number generation. Defaults to None.
        draw_magnet (bool, optional): If True, plot the muon shield. Defaults to False.
        kwargs_plot (dict, optional): Additional keyword arguments for plotting.
        Returns:
        ndarray: Array of simulated muon data (momentum, position, particle ID and possibly weight (if presented in the input)). 
        float (optional): Total weight of the muon shield if return_weight is True.
    """
    

    if type(muons) is tuple:
        muons = muons[0]
        
    detector = get_design_from_params(params = phi,
                                      force_remove_magnetic_field= False,
                                      fSC_mag = fSC_mag,
                                      simulate_fields=simulate_fields,
                                      sensitive_film_params=sensitive_film_params,
                                      field_map_file = field_map_file,
                                      add_cavern = add_cavern,
                                      add_target = add_target)

    detector["store_primary"] = sensitive_film_params is None
    detector["store_all"] = False
    t1 = time()
    output_data = initialize_geant4(detector, seed)
    if not draw_magnet: del detector #save memory?
    print('Time to initialize', time()-t1)
    output_data = json.loads(output_data)    

    # set_field_value(1,0,0)
    # set_kill_momenta(65)
    
    kill_secondary_tracks(True)
    if muons.shape[-1] == 8: px,py,pz,x,y,z,charge,W = muons.T
    else: px,py,pz,x,y,z,charge = muons.T
    
    if (np.abs(charge)==13).all(): charge = charge/(-13)
    assert((np.abs(charge)==1).all())

    if input_dist is not None:
        z = (-input_dist)*np.ones_like(z)
    z = np.minimum(z, -0.9)

    if SmearBeamRadius > 0: #ring transformation
        gauss = np.random.normal(0, 1, size=x.shape) 
        uniform = np.random.uniform(0, 2, size=x.shape)
        r = SmearBeamRadius + 0.8 * gauss
        _phi = uniform * np.pi
        dx = r * np.cos(_phi)
        dy = r * np.sin(_phi)
        x += dx / 100
        y += dy / 100

    muon_data = []
    for i in range(len(px)):
        simulate_muon(px[i], py[i], pz[i], int(charge[i]), x[i],y[i], z[i])
        if sensitive_film_params is None: 
            #If sensitive film is not present, we collect all the track
            data = collect()
            muon_data += [data]
        else: 
            #If sensitive film is defined, we collect only the muons that hit the sensitive film
            data_s = collect_from_sensitive()
            if len(data_s['px'])>0 and 13 in np.abs(data_s['pdg_id']): 
                j = 0
                while int(abs(data_s['pdg_id'][j])) != 13:
                    j += 1
                output_s = [data_s['px'][j], data_s['py'][j], data_s['pz'][j],data_s['x'][j], data_s['y'][j], data_s['z'][j],data_s['pdg_id'][j]]
                if muons.shape[-1] == 8: output_s.append(W[i])
                muon_data += [output_s]
            elif return_nan:
                muon_data += [[0]*muons.shape[-1]]

    muon_data = np.asarray(muon_data)
    if draw_magnet: 
        plot_magnet(detector,
                muon_data = muon_data, 
                sensitive_film_position = sensitive_film_params['position'], 
                **kwargs_plot)
    if return_weight: return muon_data, output_data['weight_total']
    else: return muon_data



DEF_INPUT_FILE = 'data/enriched_input.pkl'
if __name__ == '__main__':
    import argparse
    import gzip
    import pickle
    import multiprocessing as mp
    from time import time
    from lib.reference_designs.params import sc_v6, optimal_oliver, new_parametrization, oliver_scaled
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--c", type=int, default=45)
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("--f", type=str, default=DEF_INPUT_FILE)
    parser.add_argument("-params", type=str, default='sc_v6')
    parser.add_argument("--z", type=float, default=None)
    parser.add_argument("-sens_plane", type=float, default=82)
    parser.add_argument("-real_fields", action = 'store_true')
    parser.add_argument("-field_file", type=str, default='data/outputs/fields.pkl') 
    parser.add_argument("-shuffle_input", action = 'store_true')
    parser.add_argument("-remove_cavern", dest = "add_cavern", action = 'store_false')
    parser.add_argument("-plot_magnet", action = 'store_true')
    parser.add_argument("-warm",dest="SC_mag", action = 'store_false')
    parser.add_argument("-save_data", action = 'store_true')
    parser.add_argument("-return_nan", action = 'store_true')

    args = parser.parse_args()
    cores = args.c
    if args.params == 'sc_v6': params = sc_v6
    elif args.params == 'oliver': params = optimal_oliver
    elif args.params == 'oliver_scaled': params = oliver_scaled
    else:
        with open(args.params, "r") as txt_file:
            params = np.array([float(line.strip()) for line in txt_file])
        params_idx = new_parametrization['M1'] + new_parametrization['M2'] + new_parametrization['M3'] + new_parametrization['M4'] + new_parametrization['M5'] + new_parametrization['M6']
    params = np.array(params)
    if params.size != 63:
        new_phi = np.array(sc_v6, dtype=params.dtype)
        new_phi[np.array(params_idx)] = params
        if args.SC_mag:
            new_phi[new_parametrization['M2'][2]] = new_phi[new_parametrization['M2'][1]]
            new_phi[new_parametrization['M2'][4]] = new_phi[new_parametrization['M2'][3]]
        params = new_phi
    
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
    if args.sens_plane is None or args.sens_plane == 0:
        sensitive_film_params = None
    else: 
        sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6, 'position':args.sens_plane}
    t1_fem = time()
    detector = None
    if not args.real_fields:
        args.field_file = None
    else: 
        detector = get_design_from_params(np.asarray(params), args.SC_mag, False,True, args.field_file, sensitive_film_params, False, True, cores_field=cores)
    t2_fem = time()

    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)
    if args.shuffle_input: np.random.shuffle(data)
    if 0<n_muons<=data.shape[0]:
        data_n = data[:n_muons]
        cores = min(cores,n_muons)
    else: data_n = data

    workloads = split_array(data_n,cores)
    t1 = time()
    with mp.Pool(cores) as pool:
        result = pool.starmap(run, [(workload,params,input_dist,True,args.SC_mag,sensitive_film_params,args.add_cavern, 
                                    False,args.field_file,args.return_nan,args.seed, False) for workload in workloads])
    t2 = time()
    print(f"Time to FEM: {t2_fem - t1_fem:.2f} seconds.")
    print(f"Workload of {np.shape(workloads[0])[0]} samples spread over {cores} cores took {t2 - t1:.2f} seconds.")
    
    all_results = []
    for rr in result:
        resulting_data,weight = rr
        if len(resulting_data)==0: continue
        all_results += [resulting_data]
    print(f"Weight = {weight} kg")
    try: all_results = np.concatenate(all_results, axis=0)
    except: pass
    print(params)
    try: print('Data Shape', all_results.shape)
    except: print('Data Shape', len(all_results))
    all_results = all_results[:10000]
    print('n_input', data_n[:,7].sum())
    if args.save_data:
        with gzip.open(f'data/outputs/outputs_optimal.pkl', "wb") as f:
            pickle.dump(all_results, f)
    if not (args.sens_plane is None or args.sens_plane == 0):
        print('n_hits', all_results[:,7].sum())
    if args.plot_magnet:
        sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6, 'position':38}
        angle = 90
        elev = 0
        if False:#detector is not None:
            plot_magnet(detector, muon_data = all_results, sensitive_film_position = sensitive_film_params['position'], azim = angle, elev = elev)
        else:
            result = construct_and_plot(muons = all_results,phi = params,fSC_mag = args.SC_mag,sensitive_film_params = sensitive_film_params, simulate_fields=args.real_fields, field_map_file = args.field_file, cavern = False, azim = angle, elev = elev)#args.add_cavern)
                                         
