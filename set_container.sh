#. /disk/lhcb/scripts/lhcb_setup.sh
export PYTHONPATH=$PYTHONPATH:`readlink -f python`:`readlink -f cpp/build`
# export APPTAINER_TMPDIR=/disk/users/`whoami`/temp
export APPTAINER_CMD=/disk/users/`whoami`/apptainer/bin/apptainer
export PROJECTS_DIR="$(dirname "$PWD")"
apptainer shell --nv -B /cvmfs -B /disk/users/`whoami` -B /home/hep/`whoami` /disk/users/lprate/containers/snoopy_geant_cuda.sif
