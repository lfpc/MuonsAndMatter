import gzip
import json

import numpy as np
from muon_slabs import simulate_muon, initialize, collect, kill_secondary_tracks, collect_from_sensitive
from lib.reference_designs.params_design_9 import get_design as get_design_9
from lib.reference_designs.params_design_8 import get_design as get_design_8
from lib.reference_designs.params import *
import time
from lib.ship_muon_shield_customfield import get_design_from_params, initialize_geant4
import pickle
from plot_magnet import plot_magnet


def main(n_muons:int,
        design = 100, 
         output_file='plots/detector_visualization.png', 
         params=None,
          sensitive_film_position:float = 57, 
          add_cavern = True,
          fSC_mag:bool = True,
          input_dist:float = 0.9,
          input_file = 'data/inputs.pkl'):
    design = int(design)
    assert design in {100, 9, 8}

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

        detector = get_design_from_params(params, force_remove_magnetic_field=False, fSC_mag=fSC_mag, add_cavern=add_cavern,
                                          use_field_maps=False, sensitive_film_params={'dz': 0.01, 'dx': 10, 'dy': 10, 'position':sensitive_film_position})
    elif design == 9:
        detector = get_design_9(force_remove_magnetic_field=False)
    elif design == 8:
        # Design 8 is built directly from the parameters and using similar code, shield optimizations can be performed
        detector = get_design_8(force_remove_magnetic_field=False)

    detector['limits']["max_step_length"] = 0.05 # meter
    detector['limits']["minimum_kinetic_energy"] = 0.1 # GeV
    detector["store_primary"] = True # If you place a sensitive film, you can also set this to False because you can
                                     # get all the hits at the sensitive film.
    detector["store_all"] = False



    output_data = initialize_geant4(detector)
    output_data = json.loads(output_data) 
    print(f"Detector weight: {output_data['weight_total']} kilograms or {output_data['weight_total'] / 1E3} tonnes ")


    # set_field_value(1,0,0)
    # set_kill_momenta(65)
    kill_secondary_tracks(True)
    const = False

    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)
    np.random.seed(42)
    np.random.shuffle(data)
    px,py,pz,x,y,z,particle = data.T
    charge = particle.astype(int)
    #charge = (particle/(-13)).astype(int)
    assert (np.abs(charge)==1).all()
    if input_dist is not None:
        z_pos = detector['magnets'][0]['z_center'] - detector['magnets'][0]['dz']-input_dist
        z = z_pos*np.ones_like(z)
    else:
        z = z/100 + 70.845 - 68.685 + 66.34
        z = detector['magnets'][0]['z_center'] - detector['magnets'][0]['dz'] + z
    #z = z/100 + 70.845 - 68.685 + 66.34
    #z = detector['magnets'][0]['z_center'] - detector['magnets'][0]['dz'] + z

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
    print('SHAPE of DATA', np.array(muon_data).shape)
    plot_magnet(detector, 
                output_file,
                muon_data_sensitive,
                sensitive_film_position, 
                azim = args.angle,
                elev = args.elev,
                #ignore_magnets=[0,2,3,4,5,6,7],
                )
    return output_data['weight_total']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",type=int,default = 10)
    parser.add_argument("-input_file", type=str, default = 'data/inputs.pkl')
    parser.add_argument("-sens",type=float,default = 83.2)
    parser.add_argument("-params",type=str,default = 'sc_v6')
    parser.add_argument("-angle",type=float,default = 126)
    parser.add_argument("-elev",type=float,default = 17)
    parser.add_argument("-params_test", nargs='+', default=None)
    parser.add_argument("-remove_cavern", dest = "add_cavern", action = 'store_false')
    parser.add_argument("-warm", dest = 'SC', action='store_false')
    args = parser.parse_args()
    if args.params == 'sc_v6': params = sc_v6
    elif args.params == 'oliver': params = optimal_oliver
    else:
        with open(args.params, "r") as txt_file:
            params = np.array([float(line.strip()) for line in txt_file])
        params_idx = (np.array(new_parametrization['M2'])[[0, 1, 3, 5, 6, 7]]).tolist() + [new_parametrization['M3'][0]]+\
           new_parametrization['M4'] + new_parametrization['M5'] + new_parametrization['M6']
        #params_idx =  new_parametrization['M2'][:-2] + [new_parametrization['M3'][0]]+\
        #   new_parametrization['M4'] + new_parametrization['M5'] + new_parametrization['M6']

        import torch
        params = torch.tensor(params)
        if params.size(-1) != 72:
            new_phi = torch.tensor(sc_v6, dtype=params.dtype)
            new_phi[torch.as_tensor(params_idx)] = params
            if args.SC:
                new_phi[new_parametrization['M2'][2]] = new_phi[new_parametrization['M2'][1]]
                new_phi[new_parametrization['M2'][4]] = new_phi[new_parametrization['M2'][3]]
        params = new_phi.numpy()
    if args.params_test is not None:
        assert len(args.params_test) % 2 == 0
        for i in range(0, len(args.params_test), 2):
            params[int(args.params_test[i])] = float(args.params_test[i + 1])
    sensitive_film = args.sens
    main(n_muons = args.n,params=params,sensitive_film_position=sensitive_film, fSC_mag=args.SC, input_file=args.input_file, add_cavern=args.add_cavern)

