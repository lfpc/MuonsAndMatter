# CUDA Muon Sampling Simulation

## Compile the CUDA package
Starting from the root of the repo
```
source env.sh
cd cuda_muons/faster_muons/faster_muons_torch/
```
build like this:
```
pip3 install -v --user --no-deps --no-build-isolation .
```


## Collect data from Geant4

Study the `collect_single_step_data.py` file to change paramterers etc.
```
cd cuda_muons/
python3 collect_single_step_data.py
```
Combine the data with:
```
python3 combine_geant4_data.py
```
And then convert to tfrecords:
```
python3 convert_to_tf_records.py
```
Finally, collect histograms from the tfrecords file:
```
python3 collect_histograms.py
```

Please change these files such that the arguments are taken from command line to make it more consistent.


I'll add more details later.