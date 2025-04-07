
import json
import numpy as np
from lib.ship_muon_shield_customfield import create_magnet, initialize_geant4
from muon_slabs import simulate_muon, collect, kill_secondary_tracks, collect_from_sensitive
from plot_magnet import plot_magnet, construct_and_plot
from time import time
from lib.reference_designs.params import new_parametrization

def design_muon_shield(params,simulate_fields = False, field_map_file = None, cores_field:int = 1):
    cm = 1
    tesla = 1


    dZ, dXIn, dXOut, dYIn, dYOut, gapIn, gapOut, ratio_yokesIn, ratio_yokesOut, dY_yokeIn, dY_yokeOut, midGapIn, midGapOut, NI = params
    Z = dZ

    tShield = {
        'magnets':[],
        'global_field_map': {'B': np.array([])},
    }
    if field_map_file is not None or simulate_fields: 
        simulate_fields = (not exists(field_map_file)) or simulate_fields
        max_x = max(np.max(dXIn + dXIn * ratio_yokesIn + gapIn+midGapIn), np.max(dXOut + dXOut * ratio_yokesOut+gapOut+midGapOut))/100
        max_y = max(np.max(dYIn + dY_yokeIn), np.max(dYOut + dY_yokeOut))/100
        max_x = np.round(max_x,decimals=1).item()
        max_y = np.round(max_y,decimals=1).item()
        d_space = (max_x+0.3, max_y+0.3, (-0.5, np.ceil((Z[-1]+dZf[-1]+50+10)/100).item()))
        resol = RESOL_DEF
        field_map = get_field(simulate_fields,np.asarray(params),Z_init = (Z[0] - dZf[0]), fSC_mag=fSC_mag, 
                              resol = resol, d_space = d_space,
                              file_name=field_map_file, only_grid_params=True, NI_from_B_goal = NI_from_B,
                              cores = min(cores_field,n_magnets))
        tShield['global_field_map'] = field_map
        #tShield['cost'] = cost
    Ymgap = 0
    ironField_s = 1.9 * tesla

    if simulate_fields or field_map_file is not  None:
        field_profile = 'global'
        fields_s = [[],[],[]]
    else:
        field_profile = 'uniform'
        magFieldIron_s = [0., ironField_s, 0.]
        RetField_s = [0., -ironField_s/ratio_yokesIn, 0.]
        ConRField_s = [-ironField_s/ratio_yokesIn, 0., 0.]
        ConLField_s = [ironField_s/ratio_yokesIn, 0., 0.]
        fields_s = [magFieldIron_s, RetField_s, ConRField_s, ConLField_s]

        create_magnet('mag0', "G4_Fe", tShield, fields_s, field_profile, dXIn, dYIn, dXOut,
              dYOut, dZ.item(), midGapIn, midGapOut, ratio_yokesIn, ratio_yokesOut,
              dY_yokeIn, dY_yokeOut, gapIn, gapOut, Z, False, Ymgap=Ymgap)
    field_profile = 'global' if simulate_fields else 'uniform'
    return tShield

def get_design_from_params(params, 
                           simulate_fields = False,
                           field_map_file = None,
                           sensitive_film_params:dict = {'dz': 0.01, 'dx': 4, 'dy': 6,'position':82},
                           cores_field:int = 1):
    params = np.round(params, 2)
    shield = design_muon_shield(params, simulate_fields = simulate_fields, field_map_file = field_map_file, cores_field=cores_field)
    World_dZ = 200 #m
    World_dX = World_dY = 15

    shield.update({
     "worldSizeX": World_dX, "worldSizeY": World_dY, "worldSizeZ": World_dZ,
        "type" : 1,
        "limits" : {
            "max_step_length": 0.05,
            "minimum_kinetic_energy": 0.1,
        },
    })
    if sensitive_film_params is not None:
        shield.update({
            "sensitive_film": {
            "z_center" : sensitive_film_params['position'],
            "dz" : sensitive_film_params['dz'],
            "dx": sensitive_film_params['dx'],
            "dy": sensitive_film_params['dy']}})
    return shield



def run(muons, 
    phi, 
    simulate_fields = False,
    field_map_file = None,
    return_nan:bool = False,
    seed:int = None,
    keep_tracks_of_hits = False):
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

    z_gap = 10#cm
    z_sens = (params[0]*2+z_gap/2)/100
    print('Sensitive Film Position', z_sens)
    sensitive_film_params = {'dz': 0.01, 'dx': 14, 'dy': 14, 'position': z_sens}
    if type(muons) is tuple:
        muons = muons[0]
    
    detector = get_design_from_params(params = phi,
                      simulate_fields=simulate_fields,
                      sensitive_film_params=sensitive_film_params,
                      field_map_file = field_map_file)

    detector["store_primary"] = sensitive_film_params is None or keep_tracks_of_hits
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

    z = -(z_gap/200)*np.ones_like(z)

    muon_data = []
    for i in range(len(px)):
        simulate_muon(px[i], py[i], pz[i], int(charge[i]), x[i],y[i], z[i])
        
        #If sensitive film is defined, we collect only the muons that hit the sensitive film
        if keep_tracks_of_hits:
            data = collect()
            data_s = collect_from_sensitive()
            if not (len(data_s['px'])>0 and 13 in np.abs(data_s['pdg_id'])): continue 
            data['pdg_id'] = charge[i]*-13
            data['W'] = W[i] if muons.shape[-1] == 8 else 1
            muon_data += [data]
        else:
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
    
    return muon_data

def Smear(x, y, SmearBeamRadius):
    if SmearBeamRadius > 0: #ring transformation
        gauss = np.random.normal(0, 1, size=x.shape) 
        uniform = np.random.uniform(0, 2, size=x.shape)
        r = SmearBeamRadius + 0.8 * gauss
        _phi = uniform * np.pi
        dx = r * np.cos(_phi)
        dy = r * np.sin(_phi)
        x += dx / 100
        y += dy / 100
    return x, y


DEF_INPUT_FILE = 'data/muons/subsample_4M.pkl'
if __name__ == '__main__':
    import argparse
    import gzip
    import pickle
    import multiprocessing as mp
    from time import time
    from lib.reference_designs.params import *
    import os
    from functools import partial
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="Number of muons to process, 0 means all")
    parser.add_argument("--c", type=int, default=45, help="Number of CPU cores to use for parallel processing")
    parser.add_argument("-seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--f", type=str, default=DEF_INPUT_FILE, help="Input file (gzip .pkl) path containing muon data")
    parser.add_argument("-real_fields", action='store_true', help="Use realistic field maps (FEM) instead of uniform fields")
    parser.add_argument("-field_file", type=str, default='data/outputs/fields_mm.npy', help="Path to save field map file") 
    parser.add_argument("-shuffle_input", action='store_true', help="Randomly shuffle the input data")
    parser.add_argument("-plot_magnet", action='store_true', help="Generate visualization of the magnet and muon tracks")
    parser.add_argument("-save_data", action='store_true', help="Save simulation results to output file")
    parser.add_argument("-return_nan", action='store_true', help="Return zeros for muons that don't hit the sensitive film")
    parser.add_argument("-keep_tracks_of_hits", action='store_true', help="Store full tracks of muons that hit the sensitive film")
    parser.add_argument("-angle", type=float, default=90, help="Azimuthal viewing angle for 3D plot")
    parser.add_argument("-elev", type=float, default=90, help="Elevation viewing angle for 3D plot")

    args = parser.parse_args()
    cores = args.c
    params = np.asarray(melvin)[new_parametrization['HA']]
    print(params)
    def split_array(arr, K):
        N = len(arr)
        base_size = N // K
        remainder = N % K
        sizes = [base_size + 1 if i < remainder else base_size for i in range(K)]
        splits = np.split(arr, np.cumsum(sizes)[:-1])
        return splits
    n_muons = args.n
    input_file = args.f
    t1_fem = time()
    if not args.real_fields:
        args.field_file = None
    else: 
        core_fields = 8
        detector = get_design_from_params(np.asarray(params), args.SC_mag, False,True, args.field_file, None, False, True, cores_field=core_fields, extra_magnet=args.extra_magnet)
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
                              phi=params, 
                              simulate_fields=False, 
                              field_map_file=args.field_file, 
                              return_nan=args.return_nan, 
                              seed=args.seed, 
                              keep_tracks_of_hits=args.keep_tracks_of_hits)

        result = pool.map(run_partial, workloads)
        t2 = time()
    print(f"Time to FEM: {t2_fem - t1_fem:.2f} seconds.")
    print(f"Workload of {np.shape(workloads[0])[0]} samples spread over {cores} cores took {t2 - t1:.2f} seconds.")
    print(params)
    all_results = []
    for rr in result:
        if len(rr)==0: continue
        all_results += [rr]
    try: all_results = np.concatenate(all_results, axis=0)
    except: all_results = []
    
    try: 
        print('Data Shape', all_results.shape)
        print('n_hits', all_results[:,7].sum())
        print('n_input', data_n[:,7].sum())
    except: 
        print('Data Shape', len(all_results))
        print('Input Shape', len(data_n))
    cost = 0
    print(f"Cost = {cost} CHF")
    if args.save_data:
        with open(f'data/outputs/output_data_one_magnet.pkl', "wb") as f:
            pickle.dump(all_results, f)

    z_gap = 10#cm
    z_sens = (params[0]*2+z_gap)/100
    print('Sensitive Film Position', z_sens)
    sensitive_film_params = {'dz': 0.01, 'dx': 14, 'dy': 14, 'position': z_sens}
    detector = get_design_from_params(params = params,
                      simulate_fields=False,
                      sensitive_film_params=sensitive_film_params,
                      field_map_file = None)
    plot_magnet(detector, muon_data = all_results[:1000], sensitive_film_position = sensitive_film_params['position'], azim = args.angle, elev = args.elev)
        
                                         