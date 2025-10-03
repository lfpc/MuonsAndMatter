import gzip
import numpy as np
import matplotlib.pyplot as plt
from lib.reference_designs.params import *
import time
from lib.ship_muon_shield import get_design_from_params
import pickle
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation
import os
from copy import deepcopy
import argparse
from MuonsAndMatter.muons_and_matter.muons_and_matter import run, split_array
from plot_magnet import construct_and_plot


DEF_INPUT_FILE = '/home/hep/lprate/projects/MuonsAndMatter/data/inputs.pkl'

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=0)
parser.add_argument("--c", type=int, default=45)
parser.add_argument("-seed", type=int, default=None)
parser.add_argument("--f", type=str, default=DEF_INPUT_FILE)
parser.add_argument("-params", nargs='+', default=sc_v6)
parser.add_argument("--z", type=float, default=0.1)
parser.add_argument("--sens_plane", type=float, default=57)
parser.add_argument("-shuffle_input", action = 'store_true')


args = parser.parse_args()
cores = args.c
params=list(args.params)
n_muons = args.n
input_file = args.f
z_bias = 50
input_dist = args.z
sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6,'position': args.sens_plane}

def GetBounds(zGap:float = 1.):
    magnet_lengths = [(170 + zGap, 300 + zGap)] * 6  
    dX_bounds = [(10, 100)] * 2
    dY_bounds = [(20, 200)] * 2 
    gap_bounds = [(2, 70)] * 2 
    bounds = magnet_lengths + 6*(dX_bounds + dY_bounds + gap_bounds)
    return np.array(bounds).T
def muon_loss(x,y,particle):
    left_margin = 2.6
    right_margin = 3
    y_margin = 5
    MUON = 13
    charge = -1*np.sign(particle)
    mask = (-charge*x <= left_margin) & (-right_margin <= -charge*x) & (np.abs(y) <= y_margin) & ((np.abs(particle).astype(int))==MUON)
    x = x[mask]
    charge = charge[mask]
    return np.sqrt(1 + (charge*x-right_margin)/(left_margin+right_margin))


num_frames = 10
SC_mag = True

min_bound,max_bound = GetBounds()
relevant_parameters = magnets_params[4]+magnets_params[5]+magnets_params[6]#range(len(sc_v6))
with gzip.open(input_file, 'rb') as f:
    data = pickle.load(f)

def main(p, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    weight = []
    loss = []
    loss = []
    phi_range = np.linspace(min(min_bound[p],sc_v6[p]),max(max_bound[p],sc_v6[p]),num_frames)
    workloads = split_array(data,cores)
    for phi in phi_range:
        params = np.asarray(deepcopy(sc_v6))
        params[p] = phi
        name = os.path.join(out_dir,f'param_{p}_{phi:.0f}.png')
        
        with Pool(cores) as pool:
            result = pool.starmap(run, [(workload,params,z_bias,input_dist,True,True,sensitive_film_params,args.seed, False) for workload in workloads])
        all_results = []
        for rr in result:
            resulting_data,w = rr
            if resulting_data.size == 0: continue
            all_results += [resulting_data]
        
        all_results = np.concatenate(all_results, axis=0).T
        _,_,_,x,y,_,particle = all_results
        weight.append(np.mean(w))
        loss.append(muon_loss(x,y,particle).sum()+1)
        plot_args = {'output_file':name}
        with Pool(1) as pool:
            result = pool.starmap(construct_and_plot, [(all_results.T,params,z_bias,True,{'dz': 0.01, 'dx': 4, 'dy': 6,'position': 5},plot_args)])
    return weight,loss

def generate_animations(p):
        out_dir = f'plots/params/param_{p}'
        phi_range = np.linspace(min_bound[p],max_bound[p],num_frames)
        def update(frame):
            ax.clear()  # Clear the previous frame
            phi = phi_range[frame]
            img = plt.imread(os.path.join(out_dir,f'param_{p}_{phi:.0f}.png'))  # Read the current frame
            ax.imshow(img, aspect='auto')
            ax.axis('off')  # Hide axes
        fig, ax = plt.subplots(2,1)
        fig.tight_layout()
        ani = FuncAnimation(fig, update, frames=num_frames, repeat=True)
        ani.save(os.path.join(out_dir,f'animation_{p}.gif'), writer='pillow', fps=20)
        plt.plot(phi_range,weight)
        plt.xlabel(f'Param {p}')
        plt.ylabel('Weight [kg]')
        plt.grid()
        plt.savefig(os.path.join(out_dir,f'weight_{p}'))
        plt.close()

if __name__=='__main__':
    for p in [26,30]:#relevant_parameters:
        out_dir = f'plots/params/param_{p}'
        weight,loss= main(p,out_dir=out_dir)
        with gzip.open(os.path.join(out_dir,f'weight_{p}.pkl'), "wb") as f:
            pickle.dump(weight, f)
        with gzip.open(os.path.join(out_dir,f'loss_{p}.pkl'), "wb") as f:
            pickle.dump(loss, f)
        print('weight: ', weight)
        print('loss: ', loss)
    