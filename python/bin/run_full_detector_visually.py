import gzip
import json

import numpy as np
from muon_slabs import simulate_muon, initialize, collect, kill_secondary_tracks, collect_from_sensitive
from lib.reference_designs.params_design_9 import get_design as get_design_9
from lib.reference_designs.params_design_8 import get_design as get_design_8
from lib.reference_designs.params import *
import time
from lib.ship_muon_shield_customfield import get_design_from_params
import pickle
from plot_magnet import plot_magnet

def add_fixed_params(phi: np.ndarray):
    if len(phi) == 21:  # insert sc_v6 SC mag
        phi = np.concatenate([np.array([0., 353.0780, 125.0830]), phi])
        phi = np.concatenate([phi[:6], np.array([72.0000, 51.0000, 29.0000, 46.0000, 10.0000, 7.0000, 
                                                45.6888, 45.6888, 22.1839, 22.1839, 27.0063, 16.2448, 
                                                10.0000, 31.0000, 35.0000, 31.0000, 51.0000, 11.0000]), phi[6:]])
    elif len(phi) == 24:
        phi = np.concatenate([np.array([0., 353.0780, 125.0830]), phi])
        phi = np.concatenate([phi[:6], np.array([72.0000, 51.0000, 29.0000, 46.0000, 10.0000, 7.0000, 1.0, 
                                                45.6888, 45.6888, 22.1839, 22.1839, 27.0063, 16.2448, 3.0, 
                                                10.0000, 31.0000, 35.0000, 31.0000, 51.0000, 11.0000, 1.0]), phi[6:]])
    if len(phi) == 42:  # insert hadron absorber and other (?)
        phi = np.concatenate([np.array([40.0, 231.0]), phi])
        phi = np.concatenate([phi[:8], np.array([40.0, 40.0, 150.0, 150.0, 1.0, 1.0, 50.0, 50.0, 130.0, 
                                                130.0, 2.0, 2.0]), phi[8:]])
    elif len(phi) == 48:
        phi = np.concatenate([np.array([40.0, 231.0]), phi])
        phi = np.concatenate([phi[:8], np.array([40.0, 40.0, 150.0, 150.0, 1.0, 1.0, 1.0, 
                                                50.0, 50.0, 130.0, 130.0, 2.0, 2.0, 1.0]), phi[8:]])
    if len(phi) == 56:
        # Insert specific values at given indices
        insert_indices = [13, 19, 25, 31, 37, 43, 49, 55]
        insert_values = np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0])
        for idx, val in zip(insert_indices, insert_values):
            phi = np.concatenate([phi[:idx], np.array([val]), phi[idx:]])
    
    return phi


def main(n_muons:int,
        design = 100, 
         output_file='plots/detector_visualization.png', 
         params=None,
          sensitive_film_position:float = 57, 
          fSC_mag:bool = True,
          input_file = 'data/full_sample/full_sample_0.pkl'):
    design = int(design)
    assert design in {100, 9, 8}

    z_bias = 50
    if design == 100:
        if params is None:
            if fSC_mag: params = sc_v6
            else: params = optimal_oliver
        elif isinstance(params,str):
            with open(params, 'r') as file:
                params = []
                for line in file:
                    number = float(line.strip())
                    params.append(number)
            
        if len(params)==42: #shield might have 14 fixed parameters
            params = np.insert(params,0,[70.0, 170.0])
            params = np.insert(params,8,[40.0, 40.0, 150.0, 150.0, 2.0, 2.0, 80.0, 80.0, 150.0, 150.0, 2.0, 2.0])
    

        detector = get_design_from_params(params, z_bias=z_bias, force_remove_magnetic_field=False, fSC_mag=fSC_mag, use_simulated_fields=False)
    elif design == 9:
        detector = get_design_9(z_bias=z_bias, force_remove_magnetic_field=False)
    elif design == 8:
        # Design 8 is built directly from the parameters and using similar code, shield optimizations can be performed
        detector = get_design_8(z_bias=z_bias, force_remove_magnetic_field=False)

    with open('data/gdetector.json', 'w') as f:
        json.dump(detector, f)

    detector['limits']["max_step_length"] = 0.05 # meter
    detector['limits']["minimum_kinetic_energy"] = 0.1 # GeV
    detector["store_primary"] = True # If you place a sensitive film, you can also set this to False because you can
                                     # get all the hits at the sensitive film.
    detector["store_all"] = False

    sensitive_film_params:dict = {'dz': 0.01, 'dx': 30, 'dy': 30, 'position':sensitive_film_position}
    for k,v in sensitive_film_params.items():
        if k=='position': 
            if isinstance(v,tuple): #if it is a tuple, the first value indicates the magnet number and the second the position to its end
                detector['sensitive_film']['z_center'] = v[1] + detector['magnets'][v[0]]['z_center'] + detector['magnets'][v[0]]['dz']
            else: detector['sensitive_film']['z_center'] += v
        else: detector['sensitive_film'][k] = v

    # detector["store_all"] = True
    json.dumps(detector)
    output_data = initialize(np.random.randint(256), np.random.randint(256), np.random.randint(256), np.random.randint(256), json.dumps(detector))
    output_data = json.loads(output_data)
    print("Detector weight: %f kilograms or %f tonnes "%(output_data['weight_total'], output_data['weight_total'] / 1E3))


    # set_field_value(1,0,0)
    # set_kill_momenta(65)
    kill_secondary_tracks(True)
    const = False

    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)
    np.random.shuffle(data)
    px,py,pz,x,y,z,particle,factor = data.T

    charge = (particle/(-13)).astype(int)
    assert (np.abs(charge)==1).all()
    z = z/100 + 70.845 - 68.685 + 66.34
    z = detector['magnets'][0]['z_center'] - detector['magnets'][0]['dz'] + z

    muon_data = []
    muon_data_sensitive = []


    def simulate_muon_(muon_data, muon_data_sensitive, *args):
        simulate_muon(*args)
        data = collect()
        if muon_data_sensitive is not None:
            data_sensitive = collect_from_sensitive()
            muon_data_sensitive += [data_sensitive]
        muon_data += [data]
    for i in range(n_muons):
        simulate_muon_(muon_data, muon_data_sensitive, px[i], py[i], pz[i],charge[i], x[i], y[i], z[i])
    dz = 0
    for n,i in enumerate(detector['magnets']):
        #print('components', i['components'])
        print('Magnet ', n)
        print('DZ = ', i['dz']*2)
        print('Z center = ', i['z_center'])
        print('Z in ', [i['z_center']-i['dz'],i['z_center']+i['dz']])
        dz+=i['dz']*2
    print('Total Magnets Length:', dz)
    print('Total Magnets Length real:', detector['magnets'][-1]['z_center']+detector['magnets'][-1]['dz'] - (detector['magnets'][0]['z_center']-detector['magnets'][0]['dz']))
    print('Sensitive Film Position:', detector['sensitive_film']['z_center'])
    plot_magnet(detector, 
                output_file,
                muon_data, 
                z_bias,
                sensitive_film_position, 
                azim = args.angle,
                elev = args.elev,
                #ignore_magnets=[0,2,3,4,5,6,7],
                )
    return output_data['weight_total']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",type=int,default = 1)
    parser.add_argument("-input_file", type=str, default = None)
    parser.add_argument("-sens",type=float,default = None)
    parser.add_argument("-params",type=str,default = 'sc_v6')
    parser.add_argument("-angle",type=float,default = 126)
    parser.add_argument("-elev",type=float,default = 17)
    parser.add_argument("-params_test", nargs='+', default=None)
    parser.add_argument("-warm", dest = 'SC', action='store_false')
    args = parser.parse_args()
    if args.params == 'sc_v6': params = sc_v6
    elif args.params == 'oliver': params = optimal_oliver
    else:
        with open(args.params, "r") as txt_file:
            params = np.array([float(line.strip()) for line in txt_file])
    params = add_fixed_params(params)
    if args.params_test is not None:
        assert len(args.params_test) % 2 == 0
        for i in range(0, len(args.params_test), 2):
            params[int(args.params_test[i])] = float(args.params_test[i + 1])

        print('PARAMS: ',np.array(params)[np.array([1, 12,13,14,15,16,17])])
    sensitive_film = 57
    main(n_muons = args.n,params=params,sensitive_film_position=sensitive_film, fSC_mag=args.SC)

