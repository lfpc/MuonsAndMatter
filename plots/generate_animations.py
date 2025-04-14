import os
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
from matplotlib.animation import FuncAnimation

def GetBounds(zGap:float = 1.):
    magnet_lengths = [(170 + zGap, 300 + zGap)] * 6  
    dX_bounds = [(10, 100)] * 2
    dY_bounds = [(20, 200)] * 2 
    gap_bounds = [(2, 70)] * 2 
    bounds = magnet_lengths + 6*(dX_bounds + dY_bounds + gap_bounds)
    return np.array(bounds).T

min_bound,max_bound = GetBounds()


def load_pickle(file_path):
    """Utility function to load a gzip-compressed pickle file"""
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
def weight_loss(W,beta = 10, W0 = 1558731.375):
    return 1+np.exp(beta*(W-W0)/W0)

def generate_animations(p):
    # Directories and file paths
    out_dir = f'plots/params/param_{p}'
    
    # Load the arrays from the .pkl files
    vector_1 = np.asarray(load_pickle(os.path.join(out_dir, f'weight_{p}.pkl'))) 
    vector_1 = weight_loss(vector_1)
    vector_2 = np.asarray(load_pickle(os.path.join(out_dir, f'loss_{p}.pkl')))
    
    # Ensure both vectors have the same number of elements
    assert len(vector_1) == len(vector_2), "Both vectors must have the same length."
    
    # Number of frames must match the number of values in the array
    num_frames = len(vector_1)
    phi_range = np.linspace(min(min_bound[p],sc_v6[p]),max(max_bound[p],sc_v6[p]),num_frames)
    
    def update(frame):
        # Clear the previous frame in both subplots
        ax_image.clear()
        ax_plot_1.clear()
        ax_plot_2.clear()
        ax_plot_3.clear()
        
        # Load and display the image
        phi = phi_range[frame]
        img = plt.imread(os.path.join(out_dir, f'param_{p}_{phi:.0f}.png'))
        ax_image.imshow(img, aspect='auto')
        ax_image.axis('off')  # Hide axes for the image
        
        # Plot the first vector on ax_plot_1
        ax_plot_1.plot(phi_range,vector_1,'.-', label='Weight', color='b')
        ax_plot_3.axvline(x=phi, color='r', linestyle='--', label=rf'$\psi$')
        ax_plot_1.set_ylabel('Weight Loss', color='b')
        
        # Plot the second vector on ax_plot_2 with a different scale
        ax_plot_2.plot(phi_range,vector_2,'.-', label='Muon Loss', color='g')
        #ax_plot_2.axvline(x=frame, color='r', linestyle='--')  # Keep the vertical line
        ax_plot_2.set_ylabel('Muon Loss', color='g')
        ax_plot_1.tick_params(axis='y', labelcolor='b')  # Match the color of vector_1 plot
        ax_plot_2.tick_params(axis='y', labelcolor='g')  # Match the color of vector_2 plot
        ax_plot_3.axvline(x=sc_v6[p],color = 'k',linestyle = '--', label = 'Default parameter')
        ax_plot_3.plot(phi_range,vector_2*vector_1,'.-', label='Total Loss', color='red')
        ax_plot_3.set_yticks([])
        
        # Set plot title
        #ax_plot_1.set_title('Vector Plot with Current Frame Indicator')
        
        # Optional: add a legend for each plot
        ax_plot_3.legend(loc='upper left')
        #ax_plot_2.legend(loc='upper right')
    
    # Create the figure and subplots
    fig, (ax_image, ax_plot_1) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.tight_layout()
    
    # Create a secondary y-axis for ax_plot_2 (which shares the same x-axis with ax_plot_1)
    ax_plot_2 = ax_plot_1.twinx()
    ax_plot_3 = ax_plot_1.twinx()
    fig.subplots_adjust(left=0.15, right=0.95, top=0.8, bottom=0.05)
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, repeat=True)
    
    # Save the animation as a GIF
    ani.save(os.path.join(out_dir, f'animation_{p}.gif'), writer='pillow', fps=20)
    plt.close(fig)

from lib.reference_designs.params import *
relevant_parameters = magnets_params[4]+magnets_params[5]+magnets_params[6]
if __name__ == '__main__':
    for p in [26,30,36,38]:#relevant_parameters:
        print('PARAMETER: ', p)
        generate_animations(p)