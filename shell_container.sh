export APPTAINER_CMD=/disk/users/`whoami`/apptainer/bin/apptainer
export PROJECTS_DIR="$(dirname "$PWD")"
export SAVEDIR=/scratch/lprate
apptainer exec --nv -B /cvmfs -B /scratch/`whoami` -B /disk/users/`whoami` -B /home/hep/`whoami` \
            /disk/users/lprate/containers/snoopy_geant_cuda.sif bash --rcfile set_env.sh
