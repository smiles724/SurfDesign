#!/bin/bash

DIR="$(dirname "$0")"
model_path="xxx/run/logs/fixedbb/cath_4.2/surfdesign_esm2_650m"

temperature=0.1
pdb_dir="xxx/data/pdb_samples"
out_dir="$pdb_dir/surfdesign_fasta"

python $DIR/design_pdb.py \
    --experiment_path $model_path --ckpt "best.ckpt" \
    --pdb_dir $pdb_dir --out_dir $out_dir \
    --seed 42 \
    --num_seqs 1 \
    --temperature $temperature \
    --max_iter 5
