
import json
import numpy as np
from lib.ship_muon_shield_customfield import get_design_from_params, get_field, initialize_geant4
from muon_slabs import simulate_muon, initialize, collect, kill_secondary_tracks, collect_from_sensitive
from plot_magnet import plot_magnet, construct_and_plot
from time import time



def run(muons, 
        phi, 
        input_dist:float = None,
        return_weight = False,
        fSC_mag:bool = True,
        sensitive_film_params:dict = {'dz': 0.01, 'dx': 10, 'dy': 10,'position':67},
        add_cavern = True,
        use_field_maps = False,
        field_map_file = None,
        return_nan:bool = False,
        seed:int = None,
        draw_magnet = False,
        kwargs_plot = {},):
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
        use_field_maps (bool, optional): Flag to use simulate (FEM) and use field maps in the simulation. Defaults to False.
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
                                      use_field_maps=use_field_maps,
                                      sensitive_film_params=sensitive_film_params,
                                      field_map_file = field_map_file,
                                      add_cavern = add_cavern)

    detector["store_primary"] = sensitive_film_params is None
    detector["store_all"] = False
    t1 = time()
    output_data = initialize_geant4(detector, seed)
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

    muon_data = []
    if sensitive_film_params is None:
        #If sensitive film is not present, we collect all the muons
        for i in range(len(px)):
            simulate_muon(px[i], py[i], pz[i], int(charge[i]), x[i],y[i], z[i])
            data = collect()
            muon_data += [data]
    else:
        #If sensitive film is defined, we collect only the muons that hit the sensitive film
        for i in range(len(px)):
            simulate_muon(px[i], py[i], pz[i], int(charge[i]), x[i],y[i], z[i])
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
                sensitive_film_position = 5,#sensitive_film_params['position'], 
                **kwargs_plot)
    if return_weight: return muon_data, output_data['weight_total']
    else: return muon_data



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
    parser.add_argument("-sens_plane", type=float, default=None)
    parser.add_argument("-real_fields", action = 'store_true')
    parser.add_argument("-field_file", type=str, default='data/outputs/fields.pkl') 
    parser.add_argument("-shuffle_input", action = 'store_true')
    parser.add_argument("-remove_cavern", dest = "add_cavern", action = 'store_false')
    parser.add_argument("-plot_magnet", action = 'store_true')
    parser.add_argument("-warm",dest="SC_mag", action = 'store_false')
    parser.add_argument("-return_nan", action = 'store_true')
    

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
        sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6, 'position':args.sens_plane}
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
    t1 = time()
    with mp.Pool(cores) as pool:
        result = pool.starmap(run, [(workload,params,input_dist,True,args.SC_mag,sensitive_film_params,args.add_cavern, 
                                    args.real_fields,args.field_file,args.return_nan,args.seed, False) for workload in workloads])
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
    #with gzip.open(f'data/outputs/outputs_{tag}.pkl', "wb") as f:
    #    pickle.dump(all_results, f)
    print('Data Shape', all_results.shape)
    if args.plot_magnet:
        sensitive_film_params['position'] = 5
        with mp.Pool(1) as pool:
            result = pool.starmap(construct_and_plot, [(all_results,params,True,sensitive_film_params, False, None)])
                                         
