from lib import magnet_simulations
import lib.reference_designs.params as params_lib
import os
import numpy as np

def get_NI(params, fSC_mag = False, use_diluted = False):
    params = np.asarray(params, dtype=np.float32)
    new_params = params.copy()
    for i, (mag,idx) in enumerate(params_lib.new_parametrization.items()):
        mag_params = params[idx]
        if mag_params[0]<1: continue
        if mag_params[1]<1: 
            continue
        if fSC_mag and mag  == 'M2':
            Ymgap = magnet_simulations.SC_Ymgap; yoke_type = 'Mag2'
        elif use_diluted: Ymgap = 0.; yoke_type = 'Mag1'
        else: 
            Ymgap = 0.; 
            yoke_type = 'Mag3' if mag_params[13]<0 else 'Mag1'

        d = magnet_simulations.get_magnet_params(mag_params, Ymgap=Ymgap, use_B_goal=True, yoke_type=yoke_type, use_diluted = use_diluted)
        print(f"Magnet: {mag}, NI: {d['NI(A)']:.2f} A")
        new_params[idx[-1]] = -d['NI(A)'] if yoke_type == 'Mag3' else d['NI(A)']
    return new_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get NI for given parameters.")
    parser.add_argument('--params', type=str, required=True, help='Path to the parameters file')
    parser.add_argument('--fSC_mag', action='store_true', help='Use superconducting magnet')
    parser.add_argument('--use_diluted', action='store_true', help='Use diluted parameters')
    args = parser.parse_args()

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
    params = np.array(params, dtype=np.float32)
    ni_values = get_NI(params, fSC_mag=args.fSC_mag, use_diluted=args.use_diluted).tolist()
    ni_values = [round(val, 2) for val in ni_values]
    print("NI values:", ni_values)

