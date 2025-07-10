import json
import numpy as np
from lib.ship_muon_shield_customfield import get_design_from_params, initialize_geant4, get_field
from muon_slabs import simulate_muon, collect, kill_secondary_tracks, collect_from_sensitive
from python.bin.plot_magnet import construct_and_plot, plot_fields
from time import time
import h5py

def run(muons, 
    params, 
    input_dist:float = None,
    return_cost = False,
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
    keep_tracks_of_hits = False,
    extra_magnet = False,
    NI_from_B = True,
    use_diluted = False,
    SND = False,
    kwargs_plot = {}):
    """
    Simulates the passage of muons through the muon shield and collects the resulting data.
    
    Parameters:
    muons (ndarray): Array of muon parameters. Each muon is represented by its momentum (px, py, pz), 
             position (x, y, z), charge, and optionally weight.
    phi (ndarray): Array of parameters for the detector design (magnet configuration).
    input_dist (float, optional): If different than None, define the distance to set the initial z position 
                     of all muons. If None (default), the z position is taken from the file.
    return_cost (bool, optional): If True, returns the total weight of the muon shield. Defaults to False.
    fSC_mag (bool, optional): Flag to use the hybrid configuration (i.e., with the superconducting magnet) 
                 of the muon shield in the simulation. Defaults to True.
    sensitive_film_params (dict, optional): Parameters for the sensitive film. Defaults to 
                           {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82}. 
                           If None, the simulation collects all the data from the muons.
    add_cavern (bool, optional): Include cavern geometry in the simulation. Defaults to True.
    simulate_fields (bool, optional): Flag to simulate (FEM) and use field maps in the simulation. Defaults to False.
    field_map_file (str, optional): Path to the field map file. Defaults to None.
    return_nan (bool, optional): If True, returns a list of zeros as the output for muons that do not hit 
                    the sensitive film. Defaults to False.
    seed (int, optional): Seed for random number generation. Defaults to None.
    draw_magnet (bool, optional): If True, plot the muon shield. Defaults to False.
    SmearBeamRadius (float, optional): Radius in cm for beam smearing. Defaults to 5cm.
    add_target (bool, optional): Include target geometry in simulation. Defaults to True.
    keep_tracks_of_hits (bool, optional): Store full tracks of muons that hit the sensitive film. Defaults to False.
    extra_magnet (bool, optional): Add an additional small magnet to the configuration. Defaults to False.
    kwargs_plot (dict, optional): Additional keyword arguments for plotting.
    
    Returns:
    ndarray: Array of simulated muon data (momentum, position, particle ID and possibly weight (if presented in the input)). 
    float (optional): Total weight of the muon shield if return_cost is True.
    """

    if type(muons) is tuple:
        muons = muons[0]
    
    detector = get_design_from_params(params = params,
                      force_remove_magnetic_field= False,
                      fSC_mag = fSC_mag,
                      simulate_fields=simulate_fields,
                      sensitive_film_params=sensitive_film_params,
                      field_map_file = field_map_file,
                      add_cavern = add_cavern,
                      add_target = add_target,
                      extra_magnet=extra_magnet,
                      NI_from_B = NI_from_B,
                      use_diluted = use_diluted,
                      SND = SND)
    cost = detector['cost']
    length = detector['dz']

    detector["store_primary"] = sensitive_film_params is None or keep_tracks_of_hits
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
        sigma = 1.6
        gauss_x = np.random.normal(0, sigma, size=x.shape)
        gauss_y = np.random.normal(0, sigma, size=y.shape)
        uniform = np.random.uniform(0, 1, size=x.shape)
        _phi = uniform * 2 * np.pi
        dx = SmearBeamRadius * np.cos(_phi) + gauss_x
        dy = SmearBeamRadius * np.sin(_phi) + gauss_y
        x += dx / 100
        y += dy / 100

    muon_data = []
    for i in range(len(px)):
        simulate_muon(px[i], py[i], pz[i], int(charge[i]), x[i],y[i], z[i])
        if sensitive_film_params is None: 
            #If sensitive film is not present, we collect all the track
            data = collect()
            data['pdg_id'] = charge[i]*-13
            data['W'] = W[i] if muons.shape[-1] == 8 else 1
            muon_data += [data]
        elif keep_tracks_of_hits:
            data = collect()
            data_s = collect_from_sensitive()
            if not (len(data_s['px'])>0 and 13 in np.abs(data_s['pdg_id'])): continue 
            data['pdg_id'] = charge[i]*-13
            data['W'] = W[i] if muons.shape[-1] == 8 else 1
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
    
    print('TOTAL COST:', cost)
    print('MASS:', output_data['weight_total'])
    print('LENGTH:', length)
    if return_cost: return muon_data, cost
    else: return muon_data


DEF_INPUT_FILE = 'data/muons/subsample_biased_v4.pkl'
if __name__ == '__main__':
    import argparse
    import gzip
    import pickle
    import multiprocessing as mp
    import lib.reference_designs.params as params_lib
    from lib.magnet_simulations import construct_grid, RESOL_DEF
    from functools import partial
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="Number of muons to process, 0 means all")
    parser.add_argument("--c", type=int, default=45, help="Number of CPU cores to use for parallel processing")
    parser.add_argument("-seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--f", type=str, default=DEF_INPUT_FILE, help="Input file (gzip .pkl) path containing muon data")
    parser.add_argument("-params", type=str, default='sc_v6', help="Magnet parameters configuration - name or file path")
    parser.add_argument("--z", type=float, default=None, help="Initial z-position distance for all muons if specified (default is to use from input file)")
    parser.add_argument("-sens_plane", type=float, default=82, help="Position of the sensitive plane in z (m), 0 means no sensitive plane")
    parser.add_argument("-real_fields", action='store_true', help="Use realistic field maps (FEM) instead of uniform fields")
    parser.add_argument("-field_file", type=str, default='data/outputs/fields_mm.h5', help="Path to save field map file") 
    parser.add_argument("-shuffle_input", action='store_true', help="Randomly shuffle the input data")
    parser.add_argument("-remove_cavern", dest="add_cavern", action='store_false', help="Remove the cavern from simulation")
    parser.add_argument("-plot_magnet", action='store_true', help="Generate visualization of the magnet and muon tracks")
    parser.add_argument("-warm", dest="SC_mag", action='store_false', help="Use warm magnets instead of hybrid")
    parser.add_argument("-save_data", action='store_true', help="Save simulation results to output file")
    parser.add_argument("-return_nan", action='store_true', help="Return zeros for muons that don't hit the sensitive film")
    parser.add_argument("-use_diluted", action = 'store_true', help="Use diluted field map")
    parser.add_argument("-keep_tracks_of_hits", action='store_true', help="Store full tracks of muons that hit the sensitive film")
    parser.add_argument("-use_B_goal", action='store_true', help="Use B goal for the field map")
    parser.add_argument("-expanded_sens_plane", action='store_true', help="Use big sensitive plane")
    parser.add_argument("-extra_magnet", action='store_true', help="Add an additional small magnet to the configuration (old designs)")
    parser.add_argument("-angle", type=float, default=90, help="Azimuthal viewing angle for 3D plot")
    parser.add_argument("-elev", type=float, default=90, help="Elevation viewing angle for 3D plot")
    parser.add_argument("-SND", action='store_true', help="Use SND configuration")


    args = parser.parse_args()
    cores = args.c
    if args.params == 'test':
        params_input = input("Enter the params as a Python list (e.g., [1.0, 2.0, 3.0]): ")
        params = eval(params_input)
    elif args.params in params_lib.params.keys():
        params = params_lib.params[args.params]
    elif os.path.isfile(args.params):
        with open(args.params, "r") as txt_file:
            params = [float(line.strip()) for line in txt_file]
    else: 
        raise ValueError(f"Invalid params: {args.params}. Must be a valid parameter name or a file path.")
    params = np.asarray(params)
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
    elif args.expanded_sens_plane:
        sensitive_film_params = {'dz': 0.01, 'dx': 8, 'dy': 8, 'position':args.sens_plane}
    else: 
        sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6, 'position':args.sens_plane}
    t1_fem = time()
    detector = None
    if not args.real_fields:
        args.field_file = None
    else:
         
        core_fields = 8
        detector = get_design_from_params(np.asarray(params), args.SC_mag, False,True, args.field_file, sensitive_film_params, False, True, cores_field=core_fields, extra_magnet=args.extra_magnet, NI_from_B=args.use_B_goal, use_diluted = args.use_diluted)
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
        run_partial = partial(run, 
                              params=params, 
                              input_dist=input_dist, 
                              return_cost=True, 
                              fSC_mag=args.SC_mag, 
                              sensitive_film_params=sensitive_film_params, 
                              add_cavern=args.add_cavern, 
                              simulate_fields=False, 
                              field_map_file=args.field_file, 
                              return_nan=args.return_nan, 
                              seed=args.seed, 
                              draw_magnet=False, 
                              SmearBeamRadius=5, 
                              add_target=True, 
                              keep_tracks_of_hits=args.keep_tracks_of_hits, 
                              extra_magnet=args.extra_magnet,
                              use_diluted = args.use_diluted,
                              SND  = args.SND)

        result = pool.map(run_partial, workloads)
        cost = 0
        t2 = time()
    print(f"Time to FEM: {t2_fem - t1_fem:.2f} seconds.")
    print(f"Workload of {np.shape(workloads[0])[0]} samples spread over {cores} cores took {t2 - t1:.2f} seconds.")
    print(params.tolist())
    all_results = []
    for rr in result:
        resulting_data,cost = rr
        if len(resulting_data)==0: continue
        all_results += [resulting_data]
    try: all_results = np.concatenate(all_results, axis=0)
    except: all_results = []
    
    try: 
        print('Data Shape', all_results.shape)
        print('n_hits', all_results[:,7].sum())
        print('n_input', data_n[:,7].sum())
    except: 
        print('Data Shape', len(all_results))
        print('Input Shape', len(data_n))
    print(f"Cost = {cost} CHF")
    if args.save_data:
        try: tag = f"{args.params.split('/')[-2]}_{args.f.split('/')[-1].split('.')[0]}"
        except: tag = f"{args.params}_{args.f.split('/')[-1].split('.')[0]}"
        data_file = f"data/outputs/output_{tag}.pkl"
        with open(data_file, "wb") as f:
            pickle.dump(all_results, f)
        print("Data saved to ", data_file)
    if args.plot_magnet:
        if args.real_fields: 
            with h5py.File(detector['global_field_map']['B'], 'r') as f:
                fields = f["B"][:]
                d_space = f["d_space"][:].tolist()
            d_space = [[round(val, 2) for val in axis] for axis in d_space]
            limits = ((d_space[0][0], d_space[1][0], d_space[2][0]), (d_space[0][1], d_space[1][1], d_space[2][1]))
            resol = (d_space[0][2], d_space[1][2], d_space[2][2])
            assert resol == RESOL_DEF, resol
            X, Y, Z = construct_grid(limits=limits, resol=resol)
            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            plot_fields(points,fields)
        all_results = all_results[:3000]
        if sensitive_film_params is None: sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82}
        if False:#detector is not None:
            plot_magnet(detector, muon_data = all_results, sensitive_film_position = sensitive_film_params['position'], azim = args.angle, elev = args.elev)
        else:
            result = construct_and_plot(muons = all_results,phi = params,fSC_mag = args.SC_mag,sensitive_film_params = sensitive_film_params, simulate_fields=False, field_map_file = None, cavern = False, azim = args.angle, elev = args.elev)#args.add_cavern)
                                         