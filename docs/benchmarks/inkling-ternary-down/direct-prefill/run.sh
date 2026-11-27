set -euo pipefail
mkdir -p "$CARD"
for c in 1 2 3 4 5;do (
 if [[ $CARD == 5070Ti ]];then source "$(cat w54)/env5070.sh";else source "$(cat w54)/env.sh";fi
 export LEANSONIC_KV=fp16 LEANSONIC_PDL=1 LEANSONIC_PDL_ALL=1 LEANSONIC_SEQS=1 LEANSONIC_MXFP8_DEVICE_U4=1 LEANSONIC_MXFP8_DIRECT_ALL=1 LEANSONIC_CUBIN_PATH="$PWD/engine.cubin"
 if (( c%2 ));then arms=(prior candidate candidate prior);else arms=(candidate prior prior candidate);fi
 for l in 0 1 2 3;do arm=${arms[l]}; python3 leg.py "$CARD/$c-$l-$arm" "$arm";done
);done
