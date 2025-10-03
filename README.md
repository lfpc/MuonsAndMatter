# Muon Shield for the SHiP experiment 


<img src="plots/shield.png" alt="Muon Shield Visualization" width="400"/>


## Envrionment
For non-GUI access (such as on servers), download snoopy_geant_slurm.sif from the following location:

[Containers](https://uzh-my.sharepoint.com/:f:/g/personal/luis_felipe_cattelan_physik_uzh_ch/EjWSU34WfZRLiJQ98M3XD58B5BOe7T9fRzW2ffz93Bi9nQ?e=dfgTXF)

If you are using uzh-physik cluster, shell the container by executing `set_container.sh` (be sure to have Apptainer installed).

For other clusters, modify the commands accordingly. You should include every directory
you need access to from within the container with `-B` option.

Running on the MacBook is also easy, the default binary release from Geant4 works fine. And the rest of the packages can
be installed simply via pip3.

Inside the container, build the C++ code by executing the script `build_cpp.sh`.

### Data

The input muons are, ideally, the ones produced from simulations performed by the SHiP collaboration. The muons need to be send to run_simulation with the shape [px,py,pz,x,y,z,pdg_id,weight], where weight is optional. 

### Running
The main script to run simulation is:

```
python3 run_simulation.py
```

Be aware of the possible arguments (run `python3 run_simulation.py -h`). The default is to run in parallel with 45 CPU cores through multiprocessing library. Be sure to change accordingly with your computer limitations.
