
import json
import numpy as np
from muon_slabs import simulate_muon, initialize, collect, kill_secondary_tracks

def split_array(arr, K):
    N = len(arr)
    base_size = N // K
    remainder = N % K
    sizes = [base_size + 1 if i < remainder else base_size for i in range(K)]
    splits = np.split(arr, np.cumsum(sizes)[:-1])
    return splits

def generate_random_magnetic_field_in_sphere(radius=500, num_points=1000):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)

    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    points = np.vstack((x, y, z)).T

    Bx = np.random.uniform(-1, 1, num_points)*10
    By = np.random.uniform(-1, 1, num_points)*10
    Bz = np.random.uniform(-1, 1, num_points)*10

    B = np.vstack((Bx, By, Bz)).T

    return {'points': points.tolist(), 'B': B.tolist()}


def run(muons, field_interpolation:str):
    print(muons)
    if type(muons) is tuple:
        muons = muons[0]

    d_mag = generate_random_magnetic_field_in_sphere(500,1000)
    d_mag['interpolation'] = field_interpolation
    detector = {
        # "worldPositionX": 0, "worldPositionY": 0, "worldPositionZ": 0, "worldSizeX": 11, "worldSizeY": 11,
        # "worldSizeZ": 100,
        # "magnets": magnets,
        "type": 3,
        "limits": {
            "max_step_length": -1,
            "minimum_kinetic_energy": -1
        },
        "magnetic_field": d_mag,}

    #for k,v in sensitive_film_params.items():
    #    if k=='position': detector['sensitive_film']['z_center'] += v
    #    else: detector['sensitive_film'][k] = v

    detector['limits']["max_step_length"] = 0.05 # meter
    detector['limits']["minimum_kinetic_energy"] = 0.1 # GeV
    detector["store_primary"] = False # If you place a sensitive film, you can also set this to False because you can
                                     # get all the hits at the sensitive film.
    detector["store_all"] = False

    
    output_data = initialize(np.random.randint(256), np.random.randint(256), np.random.randint(256), np.random.randint(256), json.dumps(detector))
    output_data = json.loads(output_data)

    # set_field_value(1,0,0)
    # set_kill_momenta(65)
    kill_secondary_tracks(True)
    if muons.shape[-1] == 8: px,py,pz,x,y,z,charge,W = muons.T
    else: px,py,pz,x,y,z,charge = muons.T
    z = -1000*np.ones_like(z)
    

    if (np.abs(charge)==13).all(): charge = charge/(-13)
    assert((np.abs(charge)==1).all())

    muon_data = []
    for i in range(len(px)):
        simulate_muon(px[i], py[i], pz[i], int(charge[i]), x[i],y[i], z[i])
        data = collect()
        if len(data['px'])==0: continue
        else: muon_data += [[data['px'][-1], data['py'][-1], data['pz'][-1],data['x'][-1], data['y'][-1], data['z'][-1]]]
    muon_data = np.asarray(muon_data)

    return muon_data, output_data['weight_total']



DEF_INPUT_FILE = '/home/hep/lprate/projects/MuonsAndMatter/data/inputs.pkl'#'data/oliver_data_enriched.pkl'
if __name__ == '__main__':
    import argparse
    import gzip
    import pickle
    import time
    import multiprocessing as mp
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--c", type=int, default=45)
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("--f", type=str, default=DEF_INPUT_FILE)
    parser.add_argument("-shuffle_input", action = 'store_true')
    parser.add_argument("-interpolation", type=str, default='linear')
    args = parser.parse_args()

    cores = args.c
    n_muons = args.n
    input_file = args.f

    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)
    if args.shuffle_input: np.random.shuffle(data)
    if 0<n_muons<=data.shape[0]:
        data = data[:n_muons]
        cores = min(cores,n_muons)

    workloads = split_array(data,cores)
    t1 = time.time()
    with mp.Pool(cores) as pool:
        result = pool.starmap(run, [(workload, args.interpolation) for workload in workloads])
    t2 = time.time()
    all_results = []
    for rr in result:
        resulting_data,weight = rr
        if len(resulting_data)==0: continue
        all_results += [resulting_data]

    print(f"Workload of {np.shape(workloads[0])[0]} samples spread over {cores} cores took {t2 - t1:.2f} seconds.")
    print(f"Weight = {weight} kg")
    all_results = np.concatenate(all_results, axis=0)
    print('Data Shape', all_results.shape)
    