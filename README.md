# Muon Shield Optimization for the SHiP experiment (2024 and onward)

Warning: The repository is still in very initial phases of development and can change really quickly.

<img src="images/shield.png" alt="Muon Shield Visualization" width="400"/>


## Envrionment
For non-GUI access (such as on servers), download snoopy_geant_slurm.sif from the following location:

[Containers](https://uzh-my.sharepoint.com/:f:/g/personal/luis_felipe_cattelan_physik_uzh_ch/EjWSU34WfZRLiJQ98M3XD58B5BOe7T9fRzW2ffz93Bi9nQ?e=dfgTXF)

If you are using uzh-physik cluster, shell the container by executing `set_container.sh` (be sure to have Apptainer installed). Alternatively, you can run the singularity container via the following commands:

```
. /disk/lhcb/scripts/lhcb_setup.sh
export SINGULARITY_TMPDIR=/disk/users/`whoami`/temp
export TMPDIR=/disk/users/`whoami`/tmp
export PROJECTS_DIR="$(dirname "$PWD")"
singularity shell --nv -B /cvmfs -B /disk/users/`whoami` -B /home/hep/`whoami` /disk/users/lprate/containers/snoopy_geant_slurm.sif
```

For other clusters, modify the commands accordingly. You should include every directory
you need access to from within the container with `-B` option.

Running on the MacBook is also easy, the default binary release from Geant4 works fine. And the rest of the packages can
be installed simply via pip3.


Inside the container, build the C++ code by executing the script `build_cpp.sh`.


## Data
To start with, a collection of muons data consisting of 4M samples can be found in `data/muons/subsample_4M.pkl`:

### Running
The main script to run simulation is:

```
python3 python/bin/run_simulation.py
```

Be aware of the possible arguments (run `python3 python/bin/run_simulation.py -h`). The default is to run in parallel with 45 CPU cores through multiprocessing library. Be sure to change accordingly with your computer limitations.
