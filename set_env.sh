USER_SITE=$(python3 -m site --user-site)
export PYTHONPATH="$USER_SITE:$PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:`readlink -f muons_and_matter`:`readlink -f cpp/build`:`readlink -f cuda_muons`
