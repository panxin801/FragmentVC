#!/bin/bash

source path.sh
set -eu
set -x

start_stage=1
end_stage=99

wav_dir=$1
feat_dir=$2

if [ ${start_stage} -le 0 -a ${end_stage} -ge 0 ]; then
    bash local/preprocess.sh ${wav_dir} ${feat_dir} || exit 1
fi

if [ ${start_stage} -le 1 -a ${end_stage} -ge 1 ]; then
    python ${SRC_ROOT}/train.py ${feat_dir} --save-dir ./ckpts || exit1
fi

echo "All done!"
