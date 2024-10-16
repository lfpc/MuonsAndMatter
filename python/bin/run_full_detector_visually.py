import gzip
import json

import numpy as np
from muon_slabs import simulate_muon, initialize, collect, kill_secondary_tracks, collect_from_sensitive
from lib.reference_designs.params_design_9 import get_design as get_design_9
from lib.reference_designs.params_design_8 import get_design as get_design_8
from lib.reference_designs.params import *
import time
from lib.ship_muon_shield import get_design_from_params
import pickle
from plot_magnet import plot_magnet

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
    

        detector = get_design_from_params(params, z_bias=z_bias, force_remove_magnetic_field=False, fSC_mag=fSC_mag)
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
        if k=='position' and v is not None: detector['sensitive_film']['z_center'] += v
        else: detector['sensitive_film'][k] = v

    # detector["store_all"] = True
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
    print('PARTICLES', muon_data[-1])
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
    plot_magnet(detector, 
                output_file,
                muon_data, 
                z_bias,
                sensitive_film_position, 
                azim = args.angle)
    return output_data['weight_total']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",type=int,default = 1)
    parser.add_argument("-sens",type=float,default = None)
    parser.add_argument("-params",type=str,default = 'sc_v6')
    parser.add_argument("-angle",type=float,default = 126)
    parser.add_argument("-warm", dest = 'SC', action='store_false')
    parser.add_argument("-input_file", type=str, default = None)
    args = parser.parse_args()
    if args.params == 'sc_v6': params = sc_v6
    elif args.params == 'oliver': params = optimal_oliver
    else:
        with open(args.params, "r") as txt_file:
            params = np.array([float(line.strip()) for line in txt_file])
        
    main(n_muons = args.n,params=params,sensitive_film_position=args.sens, fSC_mag=args.SC)

