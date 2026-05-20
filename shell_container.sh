export PROJECTS_DIR="$(dirname "$PWD")"
export SAVEDIR=/home/hep/`whoami`/MuonShieldProject
apptainer exec --nv -B /cvmfs -B /disk/users/`whoami` -B /home/hep/`whoami` \
            /disk/users/lprate/containers/MuonsAndMatterContainer.sif bash --rcfile set_env.sh
