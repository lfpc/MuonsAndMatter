
. /disk/lhcb/scripts/lhcb_setup.sh
cd /disk/lhcb_data/sqasim/images
export APPTAINER_TMPDIR=/disk/users/`whoami`/temp
export TMPDIR=/disk/users/`whoami`/tmp
export APPTAINER_CMD=/disk/users/`whoami`/apptainer/bin/apptainer
export PROJECTS_DIR=/home/hep/`whoami`/projects
apptainer shell --nv -B /cvmfs -B /disk/users/`whoami` -B /home/hep/`whoami` ml4physics_v4.sif
