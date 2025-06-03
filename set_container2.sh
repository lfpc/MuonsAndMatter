#. /disk/lhcb/scripts/lhcb_setup.sh
export PYTHONPATH="/opt/snoo.py/:$PYTHONPATH"  # Ensure this is first
export PYTHONPATH=$PYTHONPATH:`readlink -f python`:`readlink -f cpp/build`
export APPTAINER_TMPDIR=/disk/users/`whoami`/temp
export TMPDIR=/disk/users/`whoami`/tmp
export APPTAINER_CMD=disk/users/lprate/install-dir/bin/apptainer
export PROJECTS_DIR=/disk/users/gfrise/New_project/
export PYTHONNOUSERSITE=1

git submodule update --init --remote --recursive
apptainer shell --writable-tmpfs --nv -B /cvmfs -B /disk/users/`whoami` -B /home/hep/`whoami` /disk/users/gfrise/New_project/container/snoopy_geant_slurm2.sif
