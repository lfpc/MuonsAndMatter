import glob
import gzip
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import argparse


def _float_feature(value):
    """Returns a float_list from a list of floats."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_batch(pz_batch, px_batch, py_batch, initial_momenta_batch, step_length_batch):
    """Serializes a batch of 10-feature data with multiple fields."""
    features = {
        'pz': _float_feature(pz_batch.flatten()),
        'px': _float_feature(px_batch.flatten()),
        'py': _float_feature(py_batch.flatten()),
        'initial_momenta': _float_feature(initial_momenta_batch.flatten()),
        'step_length': _float_feature(step_length_batch.flatten())
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


parser = argparse.ArgumentParser(description="Convert pickled muon data to TFRecords.")
parser.add_argument('-dir', type=str, default='multi', help='Name of the folder containing the data')
args = parser.parse_args()

pickle_files = glob.glob(f'data/{args.dir}/muon_data_energy_loss_single_step.pkl')

# Initialize lists to accumulate data
pz_all = []
px_all = []
py_all = []
initial_momenta_all = []
step_length_all = []

print("Loading data from multiple files")
for pickle_file in tqdm(pickle_files):
    with gzip.open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        if not 'px' in data:
            continue
        pz_all.append(data['pz'])
        px_all.append(data['px'])
        py_all.append(data['py'])
        initial_momenta_all.append(data['initial_momenta'])
        step_length_all.append(data['step_length'])

# Concatenate all accumulated data
pz_all = np.concatenate(pz_all, axis=0)
px_all = np.concatenate(px_all, axis=0)
py_all = np.concatenate(py_all, axis=0)
initial_momenta_all = np.concatenate(initial_momenta_all, axis=0)
step_length_all = np.concatenate(step_length_all, axis=0)

print("Data loaded and concatenated")
print("Total size:", len(pz_all))

# Define chunk size and output file
chunk_size = 100_000
tfrecord_filename = 'data/multi/step_loss_data_joined.tfrecord'

with tf.io.TFRecordWriter(tfrecord_filename) as writer:
    for i in tqdm(range(0, len(pz_all), chunk_size)):
        # Select the chunk
        pz_batch = pz_all[i:i+chunk_size]
        px_batch = px_all[i:i+chunk_size]
        py_batch = py_all[i:i+chunk_size]
        initial_momenta_batch = initial_momenta_all[i:i+chunk_size]
        step_length_batch = step_length_all[i:i+chunk_size]

        # Serialize the batch and write to the TFRecord
        tf_example = serialize_batch(pz_batch, px_batch, py_batch, initial_momenta_batch, step_length_batch)
        writer.write(tf_example)

print("TFRecord file created successfully")
