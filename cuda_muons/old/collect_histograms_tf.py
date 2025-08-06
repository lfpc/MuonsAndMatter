import time

import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import os


import tensorflow as tf
from matplotlib.lines import Line2D


# Ensure all necessary directories exist
directories = [
    "data/multi",
    "plots/edges",
    "plots/hists_a"
]

for directory in directories:
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)



def compute_2d_histo(tensor_a, tensor_b, edges):
    t1 = time.time()
    tensor_a_np = tensor_a.numpy()
    tensor_b_np = tensor_b.numpy()
    edges_np = edges.numpy()

    hist2d,_,_ = np.histogram2d(tensor_a_np, tensor_b_np, bins=[edges_np, edges_np])
    print("Took", time.time() - t1, "seconds")
    print(hist2d.shape)

    return  hist2d

chunk_size = 100_000

# Define the feature description for parsing
feature_description = {
    'pz': tf.io.FixedLenFeature([chunk_size], tf.float32),  # Adjust size to match your chunk
    'px': tf.io.FixedLenFeature([chunk_size], tf.float32),
    'py': tf.io.FixedLenFeature([chunk_size], tf.float32),
    'initial_momenta': tf.io.FixedLenFeature([chunk_size], tf.float32),
    'step_length': tf.io.FixedLenFeature([chunk_size], tf.float32),
}

energy_segmentation = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

def parse_example(example_proto):
    """Parse the input tf.Example proto using the dictionary above."""
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    # Reshape each feature back to its original shape, e.g., (chunk_size, 10)
    parsed_features['pz'] = tf.reshape(parsed_features['pz'], [chunk_size])
    parsed_features['px'] = tf.reshape(parsed_features['px'], [chunk_size])
    parsed_features['py'] = tf.reshape(parsed_features['py'], [chunk_size])
    parsed_features['initial_momenta'] = tf.reshape(parsed_features['initial_momenta'], [chunk_size])
    parsed_features['step_length'] = tf.reshape(parsed_features['step_length'], [chunk_size])
    return parsed_features

# Load the TFRecord file
tfrecord_filename = 'data/multi/step_loss_data_joined.tfrecord'
# tfrecord_filename = 'data/muon_data_energy_loss_single_step_p3.tfrecord'
dataset = tf.data.TFRecordDataset(tfrecord_filename)


# # Apply parsing, batching, shuffling, prefetching
# dataset = (
#     dataset
#     .map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     .shuffle(buffer_size=10000)
#     .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# )

# Parse each example in the dataset
parsed_dataset = dataset.map(parse_example).prefetch(buffer_size=1_000_000)

max_size = 1_000_000*100

# Initialize large tensors with zeros as mutable variables
px_tensor = tf.Variable(tf.zeros([max_size], dtype=tf.float32))
py_tensor = tf.Variable(tf.zeros([max_size], dtype=tf.float32))
pz_tensor = tf.Variable(tf.zeros([max_size], dtype=tf.float32))
initial_momenta_tensor = tf.Variable(tf.zeros([max_size], dtype=tf.float32))  # Flattened for 3D vector
step_length_tensor = tf.Variable(tf.zeros([max_size], dtype=tf.float32))


total_elements = 0

# Iterate over the parsed TFRecord dataset
for iter, parsed_record in enumerate(parsed_dataset):
    # Ensure we don’t exceed the allocated array size
    if total_elements >= max_size:
        raise ValueError("Exceeded pre-allocated tensor size. Increase max_size.")

    print("Loading more, right now have", total_elements, 'elements...')
    indices = tf.range(max_size, max_size + len(parsed_record['px']), dtype=tf.int32)[:, tf.newaxis]

    # Directly update each tensor at the specified index
    px_tensor[total_elements:total_elements+len(parsed_record['px'])].assign(parsed_record['px'])
    py_tensor[total_elements:total_elements+len(parsed_record['px'])].assign(parsed_record['py'])
    pz_tensor[total_elements:total_elements+len(parsed_record['px'])].assign(parsed_record['pz'])
    initial_momenta_tensor[total_elements:total_elements+len(parsed_record['px'])].assign(parsed_record['initial_momenta'])
    step_length_tensor[total_elements:total_elements+len(parsed_record['px'])].assign(parsed_record['step_length'])
    total_elements += len(parsed_record['px'])

    if total_elements % 10_000_000 == 0:

        px = px_tensor[:total_elements]
        py = py_tensor[:total_elements]
        pz = pz_tensor[:total_elements]
        initial_momenta = initial_momenta_tensor[:total_elements]
        step_length = step_length_tensor[:total_elements]

        # Filtered values for delta_pz_f and delta_px_f
        delta_pz_f = tf.math.log(
            tf.abs((pz - initial_momenta) / (initial_momenta)))
        delta_pz_f = tf.where(tf.math.is_nan(delta_pz_f), -20, delta_pz_f)
        delta_pz_f = tf.where(delta_pz_f == float('-inf'), -20, delta_pz_f)

        step_length_f = tf.math.log((step_length))

        secondary_f = tf.math.log(
            tf.sqrt((px) ** 2 + (py) ** 2) / (initial_momenta))
        secondary_f = tf.where(tf.math.is_nan(secondary_f), -20, secondary_f)

        nbins = 100

        # Dynamic
        edges_dpz = tf.linspace(tf.reduce_min(delta_pz_f), tf.reduce_max(delta_pz_f), nbins + 1).numpy()
        edges_step_length = tf.linspace(tf.reduce_min(step_length_f), tf.reduce_max(step_length_f), nbins + 1).numpy()
        edges_secondary = tf.linspace(tf.reduce_min(secondary_f), tf.reduce_max(secondary_f), nbins + 1).numpy()

        # Static
        # edges_dpz = tf.linspace(-22, 0, nbins + 1).numpy()
        # edges_step_length = tf.linspace(-22, 0, nbins + 1).numpy()
        # edges_secondary = tf.linspace(-22, 0, nbins + 1).numpy()

        # Convert each array to a comma-separated string
        string1 = ','.join(map(str, edges_dpz))
        string2 = ','.join(map(str, edges_step_length))
        string3 = ','.join(map(str, edges_secondary))

        # Write the strings to a text file
        with open("plots/edges/edges.txt", "w") as file:
            file.write("edges z: " + string1 + "\n")
            file.write("edges step length: " + string2 + "\n")
            file.write("edges secondary: " + string3 + "\n")

        hists_dict = {

        }
        for i in range(len(energy_segmentation)-1):

            # Create a filter for the current segmentation
            filt = tf.logical_and(initial_momenta >= energy_segmentation[i], initial_momenta <= energy_segmentation[i + 1])

            # Filtered values for delta_pz_f and delta_px_f
            delta_pz_f = tf.math.log(tf.abs(tf.boolean_mask(pz - initial_momenta, filt) / tf.boolean_mask(initial_momenta, filt)))
            delta_pz_f = tf.where(tf.math.is_nan(delta_pz_f), -22, delta_pz_f)
            delta_pz_f = tf.where(delta_pz_f == float('-inf'), -22, delta_pz_f)


            step_length_f = tf.math.log(tf.boolean_mask(step_length, filt))

            secondary_f =  tf.math.log(tf.sqrt(tf.boolean_mask(px, filt)**2 + tf.boolean_mask(py, filt)**2)/tf.boolean_mask(initial_momenta, filt))
            secondary_f = tf.where(tf.math.is_nan(secondary_f), -22, secondary_f)

            total_loss_mag = tf.math.log(tf.sqrt(tf.boolean_mask(px, filt)**2 + tf.boolean_mask(py, filt)**2 + tf.boolean_mask(pz - initial_momenta, filt)**2)/tf.boolean_mask(initial_momenta, filt))
            nbins = 100

            # edges_dpz = tf.linspace(tf.reduce_min(delta_pz_f), tf.reduce_max(delta_pz_f), nbins + 1)
            # edges_step_length = tf.linspace(tf.reduce_min(step_length_f), tf.reduce_max(step_length_f), nbins + 1)
            # edges_secondary = tf.linspace(tf.reduce_min(secondary_f), tf.reduce_max(secondary_f), nbins + 1)

            edges_dpz = tf.linspace(-22, 0, nbins + 1)
            edges_step_length = tf.linspace(-22, 0, nbins + 1)
            edges_secondary = tf.linspace(-22, 0, nbins + 1)

            # # Compute histograms using the edges
            # hist_dpz = tf.histogram_fixed_width(delta_pz_f, [tf.reduce_min(delta_pz_f), tf.reduce_max(delta_pz_f)], nbins=nbins)
            # hist_secondary = tf.histogram_fixed_width(secondary_f, [tf.reduce_min(secondary_f), tf.reduce_max(secondary_f)], nbins=nbins)
            # hist_step_length = tf.histogram_fixed_width(step_length_f, [tf.reduce_min(step_length_f), tf.reduce_max(step_length_f)], nbins=nbins)
            #

            # Compute histograms using the edges
            hist_dpz = tf.histogram_fixed_width(delta_pz_f, [-22, 0], nbins=nbins)
            hist_secondary = tf.histogram_fixed_width(secondary_f, [-22, 0], nbins=nbins)

            hist_2d = compute_2d_histo(delta_pz_f, secondary_f, edges_dpz) # The edges are the same anyway
            hist_2d_step_length_vs_mag = compute_2d_histo(step_length_f, total_loss_mag, edges_dpz) # The edges are the same anyway
            hist_step_length = tf.histogram_fixed_width(step_length_f, [-22, 0], nbins=nbins)

            hists_dict[(energy_segmentation[i], energy_segmentation[i + 1])] = {
                'hist_dpz':hist_dpz,
                'hist_secondary':hist_secondary,
                'hist_2d':hist_2d,
                'hist_step_length':hist_step_length,
                'edges_dpz':edges_dpz,
                'edges_secondary':edges_secondary,
                'edges_step_length':edges_step_length,
                'hist_2d_step_length_vs_mag':hist_2d_step_length_vs_mag,
            }

            plotting_enabled = True
            if plotting_enabled:
                fig, ax = plt.subplots(1,3, figsize = (12,3.2))
                fig.subplots_adjust(
                    top=0.9,  # Adjust top margin (default is usually around 0.95)
                    bottom=0.1,  # Adjust bottom margin if needed
                    left=0.1,  # Adjust left margin if needed
                    right=0.9,  # Adjust right margin if needed
                    hspace=0.35,  # Space between rows
                    wspace=0.4  # Space between columns
                )
                ax[0].grid(True)
                ax[1].grid(True)
                ax[2].grid(True)

                fig.suptitle('        ')

                ax[0].stairs(hist_dpz.numpy(), edges_dpz.numpy(), label = '%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)
                ax[1].stairs(hist_secondary.numpy(), edges_secondary.numpy(), label = '%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)
                ax[2].stairs(hist_step_length.numpy(), edges_secondary.numpy(), label = '%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)

                ax[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
                ax[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
                ax[2].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))

                ax[0].set_ylabel('Frequency (arb.)')
                ax[1].set_ylabel('Frequency (arb.)')
                ax[2].set_ylabel('Frequency (arb.)')

                ax[0].set_xlabel(r'$\log\left(\frac{\Delta p_z}{\text{initial } p_z}\right)$')
                ax[1].set_xlabel(r'$\log\left(\frac{\sqrt{p_x^2 + p_y^2}}{\text{initial } p_z}\right)$')
                ax[2].set_xlabel('log (step length - $m$)')

                ax[0].set_yscale('log')
                ax[1].set_yscale('log')
                ax[2].set_yscale('log')

                print("Plotting...")
                plt.savefig('plots/hists_a/%d_dpz.pdf'%i, bbox_inches='tight')

                plot_no_log = False
                if plot_no_log:
                    secondary_f_no_log = tf.sqrt(
                        tf.boolean_mask(px, filt) ** 2 + tf.boolean_mask(py, filt) ** 2) / tf.boolean_mask(
                        initial_momenta,
                        filt)
                    delta_pz_f_no_log = tf.abs(
                        tf.boolean_mask(pz - initial_momenta, filt) / tf.boolean_mask(initial_momenta,
                                                                                      filt)).numpy()

                    secondary_f_no_log_no_frac = tf.sqrt(
                        tf.boolean_mask(px, filt) ** 2 + tf.boolean_mask(py, filt) ** 2).numpy()
                    delta_pz_f_no_log_no_frac = tf.abs(
                        tf.boolean_mask(pz - initial_momenta, filt)).numpy()

                    step_length_f_no_log = tf.boolean_mask(step_length, filt)

                    fig, ax = plt.subplots(2, 3, figsize=(9, 7))
                    fig.subplots_adjust(
                        top=0.9,  # Adjust top margin (default is usually around 0.95)
                        bottom=0.1,  # Adjust bottom margin if needed
                        left=0.1,  # Adjust left margin if needed
                        right=0.9,  # Adjust right margin if needed
                        hspace=0.65,  # Space between rows
                        wspace=0.4  # Space between columns
                    )

                    fig.suptitle('       ')


                    ax[0][0].grid(True)
                    ax[0][1].grid(True)
                    ax[0][2].grid(True)
                    ax[0][0].hist(delta_pz_f_no_log, histtype='step', bins=100, label = '%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)
                    ax[0][1].hist(secondary_f_no_log, histtype='step', bins=100, label = '%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)
                    ax[0][2].hist(step_length_f_no_log[step_length_f_no_log<100], histtype='step', bins=100, label = '%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)

                    ax[0][0].set_xlabel(r'$\frac{\Delta p_z}{\text{initial } p_z}$')
                    ax[0][1].set_xlabel(r'$\frac{\sqrt{p_x^2 + p_y^2}}{\text{initial } p_z}$')
                    ax[0][2].set_xlabel('step length - $m$')
                    ax[0][0].set_ylabel('Frequency (arb.)')
                    ax[0][1].set_ylabel('Frequency (arb.)')
                    ax[0][2].set_ylabel('Frequency (arb.)')

                    # custom_line = Line2D([0], [0], color='firebrick')  # Replace 'blue' with your desired color
                    ax[0][0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.17))
                    ax[0][1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.17))
                    ax[0][2].legend(loc='upper right', bbox_to_anchor=(1.0, 1.17))

                    ax[1][0].grid(True)
                    ax[1][1].grid(True)
                    ax[1][2].grid(True)

                    ax[1][0].hist(delta_pz_f_no_log_no_frac, histtype='step', bins=100, label = '%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)
                    ax[1][1].hist(secondary_f_no_log_no_frac, histtype='step', bins=100, label = '%.0f < $p_z$ [GeV] < %.0f'%(energy_segmentation[i], energy_segmentation[i+1]), color='firebrick', zorder=5)
                    ax[1][0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.17))
                    ax[1][1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.17))
                    ax[1][2].legend(loc='upper right', bbox_to_anchor=(1.0, 1.17))

                    ax[1][0].set_xlabel(r'$\Delta p_z$ [GeV]')
                    ax[1][1].set_xlabel(r'$\sqrt{p_x^2 + p_y^2}$ [GeV]')
                    # ax[0][2].set_xlabel('step length - $m$')

                    ax[1][0].set_yscale('log')
                    ax[1][1].set_yscale('log')
                    ax[1][0].set_ylabel('Frequency (arb.)')
                    ax[1][1].set_ylabel('Frequency (arb.)')
                    # ax[0][2].set_yscale('log')

                    print("Plotting...")
                    plt.savefig('plots/hists_a/%d_2.pdf' % i, bbox_inches='tight')
                    plt.close()

        with open("plots/hists_a/hists_n2.pkl", "wb") as file:
            pickle.dump(hists_dict, file)
        print("Histograms built.")




