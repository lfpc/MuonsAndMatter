cd faster_muons/faster_muons_torch
aux_ld_preload="$LD_PRELOAD"
export LD_PRELOAD=""
pip install --force-reinstall --user .
export LD_PRELOAD="$aux_ld_preload"
unset aux_ld_preload
export PATH="$HOME/.local/bin:$PATH"
USER_SITE=$(python3 -m site --user-site)
export PYTHONPATH="$USER_SITE:$PYTHONPATH"

cd ../..
